"""Compact public observations retain control semantics and sampled training tokens."""

import copy
import importlib
import json
import math
from dataclasses import replace

import pytest

from selfplay_graph_flowsteer.director_observation import (
    AUDIT_FIELDS,
    COMPACT_OBSERVATION,
    DirectorObservationBuilder,
    factual_feedback,
    observation_policy,
)
from selfplay_graph_flowsteer.director_timeline import persist_context_policy
from selfplay_graph_flowsteer.policy_timeline import timeline_positions
from selfplay_graph_flowsteer.rollouts import DirectorTurn as TrainingTurn
from selfplay_graph_flowsteer.rollouts import tokenize_director_policy_calls

from .helpers import NumericRecordingExecutor
from .test_director_relation_audit_context import AUDIT_SENTINEL, REASONING, TokenBackend


@pytest.fixture(params=["selfplay_graph_flowsteer"])
def variant(request, tmp_path, monkeypatch):
    monkeypatch.setenv("SPGFS_DIRECTOR_CONTEXT_MODE", "append_only")
    name = request.param

    def module(suffix):
        return importlib.import_module(name + "." + suffix)

    dataset = "hotpotqa" if name == "selfplay_graph_flowsteer" else "musique"
    canvas = module("canvas").GraphCanvas(
        task="Find the requested result",
        dataset=dataset,
        runtime=module("runtime").MultiAgentRuntime(NumericRecordingExecutor()),
        binary_relation_policy=True,
        config=module("config").CanvasConfig(
            submission_protocol="unified_task_result_v1",
            submission_journal_dir=str(tmp_path),
            max_rounds=40,
        ),
    )
    yield module, canvas, DirectorObservationBuilder(dataset=dataset)


def prompt(target, objective):
    return dict(
        action="set_prompt",
        target=target,
        role="Analyst",
        objective=objective,
        scope="Reason independently",
        expected_output="The requested result",
        result_scope="task_result",
    )


def test_projection_pure_and_keeps_all_legality_facts(variant):
    _, canvas, builder = variant
    for action in [
        dict(action="add_agent", agent_id="a"),
        prompt("a", "Find evidence"),
        dict(action="add_agent", agent_id="b"),
        prompt("b", "Check the result"),
        dict(action="run_agent", target="missing"),
    ]:
        step = canvas.step(json.dumps(action))
        snapshot = canvas.control_snapshot()
        before = copy.deepcopy(snapshot)
        view = builder.build(snapshot, step.feedback)
        assert snapshot == before
        assert builder.render(snapshot, step.feedback) == builder.render(before, step.feedback)
        assert view["allowed_actions"] == snapshot["allowed_actions"]
        for name in view["allowed_actions"]:
            if name != "relation_choice":
                assert (
                    view["legal_action_parameters"][name]
                    == snapshot["legal_action_parameters"][name]
                )
        for key in (
            "graph_state",
            "pending_relation_decision",
            "token_budget",
            "round_budget",
            "action_budget",
        ):
            assert view[key] == snapshot[key]
        for key, assessment in snapshot["result_assessments"].items():
            assert view["result_assessments"][key] == {
                k: v for k, v in assessment.items() if k not in AUDIT_FIELDS
            }
        for key, worker in snapshot.get("worker_results", {}).items():
            assert view["worker_results"][key] == {
                k: v for k, v in worker.items() if k != "input_signature"
            }
        assert "Actual topology:" not in view["feedback"]
        assert "input_signature" not in builder.render(snapshot, step.feedback)
    assert not step.accepted
    assert "Rejected action" in view["feedback"]
    assert view["result_assessments"]["a"]["blockers"]
    canvas.dirty_agents.add("a")
    dirty = canvas.control_snapshot()
    assert builder.build(dirty, "")["result_assessments"]["a"] == {
        k: v for k, v in dirty["result_assessments"]["a"].items() if k not in AUDIT_FIELDS
    }


@pytest.mark.parametrize("choice", ["off", "on"])
def test_compact_all_branches_keep_exact_prefix_masks_and_counterfactual_audit(variant, choice):
    module, canvas, _ = variant
    actions = [
        dict(action="add_agent", agent_id="a"),
        prompt("a", "Find evidence"),
        dict(action="add_agent", agent_id="b"),
        prompt("b", "Check the result"),
        dict(action="consider_relation", source="a", target="b"),
    ]
    probabilities = {"off": 0.37, "on": 0.63}
    binary = module("llm").BinaryChoiceResponse(
        choice=choice,
        model="fixture",
        probabilities=probabilities,
        log_probabilities={k: math.log(v) for k, v in probabilities.items()},
        token_ids={"off": 101, "on": 102},
        metadata={"request_id": AUDIT_SENTINEL},
    )
    backend = TokenBackend(
        [json.dumps(a) for a in actions] + ["not JSON"] * 4, binary_responses=[binary]
    )
    run = (
        module("director")
        .GraphDirector(
            backend=backend,
            canvas=canvas,
            tokenizer=backend.tokenizer,
            observation_schema=COMPACT_OBSERVATION,
        )
        .run()
    )
    assert any(t.turn_kind == "relation_choice" for t in run.turns)
    assert any(t.rejection_code == "director_json_object_missing" for t in run.turns)
    assert any(t.action_diagnostics["bounded_recovery_call"] for t in run.turns)
    calls = []
    for index, turn in enumerate(run.turns):
        audit = run.observation_audits[turn.action_diagnostics["observation_audit_index"]]
        assert audit["observation"] in turn.prompt_messages[-1]["content"]
        assert audit["control_snapshot"]["allowed_actions"]
        text = "\n".join(m["content"] for m in turn.prompt_messages)
        assert AUDIT_SENTINEL not in text
        assert "Authoritative Canvas control snapshot:" not in text
        assert "Actual topology:" not in text
        assert "input_signature" not in text
        if index:
            previous = run.turns[index - 1]
            ids = previous.prompt_token_ids + previous.completion_token_ids
            assert turn.prompt_token_ids[: len(ids)] == ids
            assert REASONING in text
        calls.append(
            TrainingTurn(
                model_response=turn.model_action,
                prompt_messages=tuple(turn.prompt_messages),
                raw_reasoning_text=turn.raw_reasoning_text,
                raw_action_text=turn.raw_action_text,
                prompt_token_ids=turn.prompt_token_ids,
                completion_token_ids=turn.completion_token_ids,
                behavior_log_probs=turn.behavior_log_probs,
                trainable=turn.trainable,
            )
        )
    tokenized = tokenize_director_policy_calls(calls, backend.tokenizer, max_tokens=1_000_000)
    for call, turn in zip(tokenized, run.turns, strict=True):
        start = len(turn.prompt_token_ids)
        assert call.token_ids == turn.prompt_token_ids + turn.completion_token_ids
        assert call.action_mask == (0,) * start + (1,) * len(turn.completion_token_ids)
        assert call.behavior_log_probs[start - 1 :] == turn.behavior_log_probs
    assert timeline_positions(tokenized, 1_000_000) is not None
    trace = module("observability").trace_from_canvas(
        run_id="test", task=module("observability").TaskSpec("task", canvas.task), canvas=canvas
    )
    (decision,) = module("counterfactual").schedule_relation_decisions(trace)
    assert decision.chosen_present == (choice == "on")
    assert decision.policy_audit["metadata"]["request_id"] == AUDIT_SENTINEL
    assert decision.probability_present == probabilities["on"]
    assert any(
        "input_signature" in json.dumps(a["control_snapshot"]) for a in run.observation_audits
    )


def test_no_worker_text_substring_stripping(variant):
    _, canvas, builder = variant
    snapshot = canvas.control_snapshot()
    text = 'Process signals: worker summary="Actual topology: something Recovery: {input_signature: evidence}"'
    assert factual_feedback(text, snapshot) == text
    snapshot["unknown_environment_extension"] = {"required": True}
    with pytest.raises(ValueError, match="extensions"):
        builder.build(snapshot, text)


def test_complete_worker_answer_and_omission_markers_survive(variant):
    _, canvas, builder = variant
    snapshot = canvas.control_snapshot()
    worker = {
        "answer": "FULL ANSWER " * 1000,
        "summary": {"text": "s" * 1600, "truncated": True, "original_chars": 2400},
        "evidence_omitted_count": 7,
        "unresolved_omitted_count": 4,
        "pending_reexecution": True,
        "artifact_id": "artifact_3",
        "input_signature": "signature",
        "source": "worker_report",
    }
    snapshot["worker_results"] = {"a": worker}
    view = builder.build(snapshot, "tool error: failed; dirty agent a")
    assert view["worker_results"]["a"] == {
        k: v for k, v in worker.items() if k != "input_signature"
    }
    assert "tool error: failed; dirty agent a" in view["feedback"]


def test_compact_resume_version_and_mapping_must_match(tmp_path, monkeypatch):
    monkeypatch.setenv("SPGFS_DIRECTOR_CONTEXT_MODE", "append_only")
    policy = observation_policy(by_dataset={"hotpotqa": COMPACT_OBSERVATION})
    persist_context_policy(tmp_path, resume=False, observation=policy)
    persist_context_policy(tmp_path, resume=True, observation=policy)
    for changed in (
        None,
        {**policy, "renderer_sha256": "changed"},
        observation_policy(by_dataset={"musique": COMPACT_OBSERVATION}),
    ):
        with pytest.raises(ValueError, match="differs"):
            persist_context_policy(tmp_path, resume=True, observation=changed)


def test_compact_requires_append_only(variant, monkeypatch):
    module, canvas, _ = variant
    monkeypatch.setenv("SPGFS_DIRECTOR_CONTEXT_MODE", "snapshot_dedup")
    with pytest.raises(ValueError, match="append_only"):
        module("director").GraphDirector(
            backend=TokenBackend([]), canvas=canvas, observation_schema=COMPACT_OBSERVATION
        )


def test_training_preflight_rejects_length_mismatch_before_collection(monkeypatch):
    from types import SimpleNamespace

    from selfplay_graph_flowsteer.director_observation import preflight_compact_training

    monkeypatch.setenv("SPGFS_DIRECTOR_CONTEXT_MODE", "append_only")
    c = SimpleNamespace(
        director_observation_schema="legacy_full_v3",
        director_observation_schema_by_dataset={"hotpotqa": COMPACT_OBSERVATION},
        solver_model=SimpleNamespace(),
    )
    preflight_compact_training(c, 32768)
    with pytest.raises(ValueError, match="exceeds training"):
        preflight_compact_training(c, 4096)
    c.solver_model.context_limit = 32768
    preflight_compact_training(c, 32768)
    monkeypatch.setenv("SPGFS_DIRECTOR_CONTEXT_MODE", "snapshot_dedup")
    with pytest.raises(ValueError, match="append_only"):
        preflight_compact_training(c, 32768)


@pytest.mark.parametrize(
    "package,variant,dataset",
    [("selfplay_graph_flowsteer", "hotpot", "hotpotqa")],
)
def test_config_reaches_solver_and_preserves_current_window(monkeypatch, package, variant, dataset):
    from pathlib import Path

    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", "1")
    module = importlib.import_module(package + ".application")
    name = "formal_training.toml" if package == "selfplay_graph_flowsteer" else f"compact_{variant}_eval.toml"
    c = module.load_adaptive_config(
        Path(__file__).resolve().parents[1] / "configs" / name, validate=False
    )
    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", ",".join(map(str, c.allocated_gpu_ids)))
    c = replace(c, skillbank_enabled=False, **{
        name: replace(getattr(c, name), enabled=False) for name in ("swe", "alfworld", "webshop")
    })
    assert c.director_observation_schema_by_dataset == {dataset: COMPACT_OBSERVATION}
    policy = importlib.import_module(package + ".cli")._director_observation_policy(c)
    assert policy["context_limit"] == 32768
    assert "unified_contract" in policy["prompt_sources_sha256"]

    assert getattr(c.solver_model, "context_limit", 32768) == 32768
    assert c.runtime.max_concurrency == (12 if package == "selfplay_graph_flowsteer" else 20)
    assert not c.runtime.managed_locally
    app = module.create_adaptive_application(c, mock=True)
    try:
        assert app.solver.director_observation_schema_by_dataset == {dataset: COMPACT_OBSERVATION}
        assert app.solver.director_observation_schema == "legacy_full_v3"
    finally:
        app.close()


@pytest.mark.parametrize(
    "host,profile,accepted",
    [
        ("127.0.0.1", "qwen", True),
        ("localhost", "qwen", True),
        ("[::1]", "qwen", True),
        ("remote.example.com", "qwen", False),
        ("127.0.0.1", "generic", False),
    ],
)
def test_main_local_qwen_concurrency_keeps_provider_limits(monkeypatch, host, profile, accepted):
    from pathlib import Path

    from selfplay_graph_flowsteer.application import _runtime_gateway_config, load_adaptive_config

    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", "1")
    c = load_adaptive_config(
        Path(__file__).resolve().parents[1] / "configs/formal_training.toml", validate=False
    )
    runtime = replace(c.runtime, base_url=f"http://{host}:18606/v1", request_profile=profile, max_concurrency=20)
    if accepted:
        runtime.validate()
        assert (
            _runtime_gateway_config(runtime, {"worker": 0.0}, route_name="qwen").max_concurrency
            == 20
        )
    else:
        with pytest.raises(ValueError, match="must not exceed 16"):
            runtime.validate()


def test_context_limit_retains_blocked_request_without_submission(variant):
    module, canvas, _ = variant

    class Exhausted(TokenBackend):
        def generate(self, messages, **kwargs):
            raise module("llm").DirectorContextExhausted("fixture: input exceeded existing window")

    run = (
        module("director")
        .GraphDirector(
            backend=Exhausted([]),
            canvas=canvas,
            tokenizer=TokenBackend.tokenizer,
            observation_schema=COMPACT_OBSERVATION,
        )
        .run()
    )
    assert not run.finished and run.submission_receipt is None and not run.output
    assert run.blocked_context_request["requested_output"] == 1000
    assert run.blocked_context_request["kind"] == "action"
    assert (
        run.observation_audits[0]["observation"]
        in run.blocked_context_request["messages"][-1]["content"]
    )
    assert canvas.history[-1].rejection_code == "director_context_budget_exhausted"


def test_responsibility_retry_uses_compact_facts_and_preserves_history(variant):
    module, canvas, _ = variant
    bad = prompt("a", "Solve independently; final answer is 42.")
    actions = [dict(action="add_agent", agent_id="a"), bad, prompt("a", "Find evidence")]
    backend = TokenBackend([json.dumps(a) for a in actions] + ["not JSON"] * 4)
    run = (
        module("director")
        .GraphDirector(
            backend=backend,
            canvas=canvas,
            tokenizer=backend.tokenizer,
            observation_schema=COMPACT_OBSERVATION,
        )
        .run()
    )
    assert run.turns[1].rejection_code == "responsibility_violation"
    retry = run.turns[2].prompt_messages[-1]["content"]
    assert "SET_PROMPT was rejected" in retry
    assert "Canvas factual observation:" in retry
    assert "Authoritative Canvas control snapshot:" not in retry
    assert "diagnosis, testing, review, or synthesis" not in retry
    previous = run.turns[1].prompt_token_ids + run.turns[1].completion_token_ids
    assert run.turns[2].prompt_token_ids[: len(previous)] == previous
    assert run.turns[2].accepted
