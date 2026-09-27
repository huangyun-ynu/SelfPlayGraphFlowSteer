"""Relation audit remains trainable without becoming future Director input."""

import json
import math
import re
from dataclasses import replace

import pytest

from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.counterfactual import schedule_relation_decisions
from selfplay_graph_flowsteer.director import GraphDirector
from selfplay_graph_flowsteer.director_timeline import persist_context_policy
from selfplay_graph_flowsteer.llm import BinaryChoiceResponse, MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec, trace_from_canvas
from selfplay_graph_flowsteer.runtime import MultiAgentRuntime

from .helpers import NumericRecordingExecutor

MODES = ("snapshot_dedup", "append_only")
AUDIT_SENTINEL = "offline-relation-request-audit"
REASONING = "<think>Keep this fixture reasoning in incremental history and training.</think>\n"


class FixtureTokenizer:
    """Reversible fixture tokens, with single-token off/on and no model download."""

    def encode(self, text, **kwargs):
        return [
            token
            for part in re.split(r"(off|on)", text)
            for token in {"off": [101], "on": [102]}.get(part, [ord(c) + 1000 for c in part])
        ]

    def decode(self, ids, **kwargs):
        return "".join({101: "off", 102: "on"}[i] if i < 1000 else chr(i - 1000) for i in ids)

    def prompt_ids(self, messages):
        text = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}"
            + ("" if m["role"] == "assistant" else "<|im_end|>\n")
            for m in messages
        )
        return tuple(self.encode(text + "<|im_start|>assistant\n"))


class TokenBackend(MockBackend):
    tokenizer = FixtureTokenizer()

    def generate(self, messages, **kwargs):
        response = super().generate(messages, **kwargs)
        completion = tuple(self.tokenizer.encode(REASONING + response.text + "<|im_end|>\n"))
        return replace(
            response,
            raw_reasoning_text=REASONING,
            prompt_token_ids=self.tokenizer.prompt_ids(messages),
            completion_token_ids=completion,
            behavior_log_probs=(-1.0,) * len(completion),
        )

    def choose_binary(self, messages, **kwargs):
        return replace(
            super().choose_binary(messages, **kwargs),
            prompt_token_ids=self.tokenizer.prompt_ids(messages),
        )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("choices", [("off",), ("on",), ("on", "off")])
def test_director_input_omits_audit_but_events_and_counterfactual_keep_it(
    monkeypatch, tmp_path, mode, unified, choices
):
    monkeypatch.setenv("SPGFS_DIRECTOR_CONTEXT_MODE", mode)
    canvas = GraphCanvas(
        task="Find the requested result",
        dataset="aime",
        runtime=MultiAgentRuntime(NumericRecordingExecutor()),
        binary_relation_policy=True,
        config=CanvasConfig(
            submission_protocol="unified_task_result_v1" if unified else "legacy",
            submission_journal_dir=str(tmp_path),
            max_rounds=40,
        ),
    )
    actions = []
    for agent_id, objective in (("a", "Find evidence"), ("b", "Check the result")):
        actions.extend([
            {"action": "add_agent", "agent_id": agent_id},
            {
                "action": "set_prompt", "target": agent_id, "role": "Analyst",
                "objective": objective, "scope": "Reason independently",
                "expected_output": "The requested result",
                **({"result_scope": "task_result"} if unified else {}),
            },
        ])
    actions.extend(
        {"action": "consider_relation", "source": "a", "target": "b"} for _ in choices
    )
    probabilities = {"off": 0.37, "on": 0.63}
    binary_responses = [
        BinaryChoiceResponse(
            choice=choice, model="fixture-model",
            probabilities=probabilities,
            log_probabilities={key: math.log(value) for key, value in probabilities.items()},
            token_ids={"off": 101, "on": 102},
            metadata={"request_id": AUDIT_SENTINEL, "request_validation": {"verified": True}},
        )
        for choice in choices
    ]
    # Subsequent rejected actions exercise multiple rounds of replayed feedback
    # without deleting either endpoint needed by the counterfactual scheduler.
    backend = TokenBackend(
        [json.dumps(action) for action in actions] + ["not JSON"] * 4,
        binary_responses=binary_responses,
    )
    run = GraphDirector(backend=backend, canvas=canvas, tokenizer=backend.tokenizer).run()
    relation_turns = [turn for turn in run.turns if turn.turn_kind == "relation_choice"]
    assert [turn.model_action for turn in relation_turns] == list(choices)
    assert all(turn.accepted and turn.trainable for turn in relation_turns)
    assert len(run.turns) == len(backend.calls)
    for index, (turn, call) in enumerate(zip(run.turns, backend.calls, strict=True)):
        assert turn.prompt_messages == call["messages"]
        content = "\n".join(m["content"] for m in turn.prompt_messages)
        for audit_field in (
            AUDIT_SENTINEL, "probabilities", "log_probabilities", "tokenizer_attestation",
            "request_validation", '"policy":',
        ):
            assert audit_field not in content
        assert (REASONING in content) is bool(index)
        if turn.turn_kind != "relation_choice":
            assert turn.raw_reasoning_text == REASONING
            assert REASONING in backend.tokenizer.decode(turn.completion_token_ids)
        if mode != "snapshot_dedup":
            audit = turn.action_diagnostics["timeline_prefix_audit"]
            assert audit["timeline_merge_candidate"]
            if index:
                assert audit["previous_policy_is_exact_prefix"] is True

    facts = canvas.control_snapshot()["graph_state"]["last_relation_decision"]
    assert facts["choice"] == choices[-1]
    assert facts["chosen_present"] is (choices[-1] == "on")
    assert facts["previously_present"] is (choices == ("on", "off"))
    assert facts["source"] == "a" and facts["target"] == "b"
    assert "policy" not in facts
    assert bool(canvas.graph.bidirectional_edges) is (choices[-1] == "on")
    for step in canvas.history:
        assert AUDIT_SENTINEL not in step.feedback
        assert AUDIT_SENTINEL not in json.dumps(step.control_snapshot)

    trace = trace_from_canvas(run_id="test", task=TaskSpec("task", canvas.task), canvas=canvas)
    events = [e for e in trace.events if e.payload.get("relation_decision", {}).get("phase") == "choice"]
    for event, turn, response in zip(events, relation_turns, binary_responses, strict=True):
        # Exercise the persisted JSON representation, as well as in-memory data.
        saved_policy = json.loads(json.dumps(event.payload))["relation_decision"]["policy"]
        assert saved_policy == turn.to_dict()["relation_decision"]["policy"]
        assert saved_policy == turn.action_diagnostics["binary_policy_audit"]
        for key, value in response.to_policy_audit().items():
            assert saved_policy[key] == value
        assert turn.behavior_log_probs == (response.log_probabilities[response.choice],)
        assert turn.completion_token_ids == (response.token_ids[response.choice],)
        assert "tokenizer_attestation" in saved_policy
    decision, = schedule_relation_decisions(trace)
    assert decision.probability_present == probabilities["on"]
    assert decision.chosen_present is (choices[-1] == "on")
    assert decision.policy_audit["metadata"]["request_id"] == AUDIT_SENTINEL
    assert decision.policy_audit["choice_token_id"] == binary_responses[-1].token_ids[choices[-1]]


@pytest.mark.parametrize("mode", MODES)
def test_resume_cannot_mix_old_relation_audit_context(monkeypatch, tmp_path, mode):
    monkeypatch.setenv("SPGFS_DIRECTOR_CONTEXT_MODE", mode)
    persist_context_policy(tmp_path, resume=False)
    persist_context_policy(tmp_path, resume=True)
    marker = tmp_path / "director_context_policy.json"
    saved = json.loads(marker.read_text())
    assert saved.pop("relation_audit_visibility") == "offline_only_v1"
    marker.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="differs"):
        persist_context_policy(tmp_path, resume=True)
