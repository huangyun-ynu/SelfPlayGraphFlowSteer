"""Thinking remains in incremental inputs and exact training calls."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.director import GraphDirector
from selfplay_graph_flowsteer.director_timeline import persist_context_policy
from selfplay_graph_flowsteer.llm import LLMResponse, MockBackend
from selfplay_graph_flowsteer.policy_timeline import timeline_positions
from selfplay_graph_flowsteer.rollouts import (
    DirectorTurn,
    TokenizedPolicyCall,
    tokenize_director_policy_calls,
)
from selfplay_graph_flowsteer.runtime import MultiAgentRuntime
from selfplay_graph_flowsteer.training import TransformersGRPOTrainer

from .helpers import RecordingExecutor
from .test_director_relation_audit_context import MODES, FixtureTokenizer

THOUGHT = "historical-reasoning-sentinel"
ACTION = "not JSON"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("reasoning", "action"), [
    (THOUGHT, ACTION),
    ("", f"<think>{THOUGHT}</think>{ACTION}"),
    ("", f"{THOUGHT}</think>{ACTION}"),
    ("", f"<think>{THOUGHT}"),
    (THOUGHT, ""),
    ("", ACTION),
])
def test_incremental_thinking_preserves_exact_training_calls(monkeypatch, mode, reasoning, action):
    monkeypatch.setenv("SPGFS_DIRECTOR_CONTEXT_MODE", mode)
    tokenizer = FixtureTokenizer()

    class Backend(MockBackend):
        def generate(self, messages, **kwargs):
            self.calls.append({"messages": [dict(m) for m in messages]})
            completion = tuple(tokenizer.encode(reasoning + action + "<|im_end|>"))
            return LLMResponse(
                text=reasoning + action, model="fixture",
                raw_reasoning_text=reasoning, raw_action_text=action,
                prompt_token_ids=tokenizer.prompt_ids(messages),
                completion_token_ids=completion,
                behavior_log_probs=tuple(-0.1 - i / 1000 for i in range(len(completion))),
                training_eligible=True,
            )

    backend = Backend([])
    canvas = GraphCanvas(task="test task", runtime=MultiAgentRuntime(RecordingExecutor()))
    run = GraphDirector(backend=backend, canvas=canvas, tokenizer=tokenizer).run()
    assert len(run.turns) == 4
    training_turns = []
    for index, turn in enumerate(run.turns):
        assert turn.prompt_messages == backend.calls[index]["messages"]
        expected = reasoning + action + ("<|im_end|>" if mode == "append_only" else "")
        assert [m["content"] for m in turn.prompt_messages if m["role"] == "assistant"] == (
            [expected] * index
        )
        if index and mode == "append_only":
            previous = run.turns[index - 1]
            assert turn.prompt_messages[:len(previous.prompt_messages)] == previous.prompt_messages
            previous_ids = previous.prompt_token_ids + previous.completion_token_ids
            assert turn.prompt_token_ids[:len(previous_ids)] == previous_ids
            audit = turn.action_diagnostics["timeline_prefix_audit"]
            assert audit["previous_policy_is_exact_prefix"]
            assert audit["timeline_merge_candidate"]
        assert turn.raw_reasoning_text == reasoning
        assert turn.raw_action_text == action
        assert tokenizer.decode(turn.completion_token_ids) == reasoning + action + "<|im_end|>"
        assert turn.trainable
        saved = json.loads(json.dumps(turn.to_dict()))
        assert saved["completion_token_ids"] == list(turn.completion_token_ids)
        assert saved["raw_reasoning_text"] == reasoning
        training_turns.append(DirectorTurn(
            model_response=turn.model_action, prompt_messages=tuple(turn.prompt_messages),
            raw_reasoning_text=turn.raw_reasoning_text, raw_action_text=turn.raw_action_text,
            prompt_token_ids=turn.prompt_token_ids, completion_token_ids=turn.completion_token_ids,
            behavior_log_probs=turn.behavior_log_probs, trainable=turn.trainable,
        ))

    calls = tokenize_director_policy_calls(training_turns, tokenizer, max_tokens=1_000_000)
    for call, turn in zip(calls, run.turns, strict=True):
        start = len(turn.prompt_token_ids)
        assert call.token_ids == turn.prompt_token_ids + turn.completion_token_ids
        assert call.action_mask == (0,) * start + (1,) * len(turn.completion_token_ids)
        assert call.behavior_log_probs[start - 1:] == turn.behavior_log_probs
    positions = timeline_positions(calls, 1_000_000)
    if mode == "append_only":
        assert positions is not None
        assert [len(row) for row in positions] == [len(t.completion_token_ids) for t in run.turns]
    else:
        assert positions is None  # Snapshot rewriting still needs per-call training.


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("old_visibility", [None, "offline_only_v1"])
def test_resume_rejects_changed_or_unmarked_thinking_policy(
    monkeypatch, tmp_path, mode, old_visibility
):
    monkeypatch.setenv("SPGFS_DIRECTOR_CONTEXT_MODE", mode)
    with pytest.raises(ValueError, match="unmarked"):
        persist_context_policy(tmp_path, resume=True)
    persist_context_policy(tmp_path, resume=False)
    persist_context_policy(tmp_path, resume=True)
    marker = tmp_path / "director_context_policy.json"
    policy = json.loads(marker.read_text())
    assert policy.pop("history_thinking_visibility") == "online_and_training_v1"
    if old_visibility is not None:
        policy["history_thinking_visibility"] = old_visibility
    marker.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="differs"):
        persist_context_policy(tmp_path, resume=True)


def test_formal_environment_enables_incremental_director_without_changing_worker_template():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("SPGFS_DIRECTOR_CONTEXT_MODE", None)
    env["PYTHONPATH"] = str(root / "src")
    code = (
        "import json; "
        "from selfplay_graph_flowsteer.director_timeline import director_context_mode; "
        "from selfplay_graph_flowsteer.qwen_compat import qwen_vllm_server_args; "
        "print(json.dumps({'mode': director_context_mode(), "
        "'director_template': '--chat-template' in qwen_vllm_server_args('Qwen3.5-9B', native_tools=False), "
        "'worker_template': '--chat-template' in qwen_vllm_server_args('Qwen3.5-9B')}))"
    )
    result = subprocess.run(
        ["bash", "-c", 'source scripts/formal/environment.sh; exec "$1" -c "$2"',
         "incremental-context-test", sys.executable, code],
        cwd=root, env=env, capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout) == {
        "mode": "append_only", "director_template": True, "worker_template": False,
    }


def test_timeline_training_matches_per_call_probabilities_and_gradients():
    torch = pytest.importorskip("torch")
    # Two exact causal calls, each with reasoning and action targets; observation
    # tokens have mask=0. No downloaded model or GPU is involved.
    calls = (
        TokenizedPolicyCall("first", (1, 2, 3, 4, 5), (0, 0, 1, 1, 1)),
        TokenizedPolicyCall("second", (1, 2, 3, 4, 5, 6, 7, 8, 9),
                            (0, 0, 0, 0, 0, 0, 0, 1, 1)),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = torch.nn.ModuleDict({
            "embed": torch.nn.Embedding(16, 4), "head": torch.nn.Linear(4, 16),
        }).double()
    trainer = object.__new__(TransformersGRPOTrainer)
    trainer.torch = torch
    trainer.model = model
    trainer.config = SimpleNamespace(device="cpu", max_sequence_length=32, entropy_coefficient=0)
    trainer._activation_cpu_offload_enabled = lambda length: False

    def logits(inputs, attention_mask, positions):
        hidden = model["embed"](inputs).cumsum(dim=1)
        return model["head"](hidden)[:, positions, :]

    trainer._selected_sequence_logits = logits
    merged = trainer._timeline_policy_outputs(calls, include_entropy=False)
    assert merged is not None
    merged_values = [values for values, _ in merged]
    (-torch.cat(merged_values).mean()).backward()
    merged_gradients = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    separate_values = [
        trainer._masked_policy_call_log_probs(call, require_grad=True) for call in calls
    ]
    for actual, expected in zip(merged_values, separate_values, strict=True):
        torch.testing.assert_close(actual, expected)
    (-torch.cat(separate_values).mean()).backward()
    for actual, parameter in zip(merged_gradients, model.parameters(), strict=True):
        torch.testing.assert_close(actual, parameter.grad)
