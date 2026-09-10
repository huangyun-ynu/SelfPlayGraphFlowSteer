from __future__ import annotations

from pathlib import Path

import pytest

from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.director import GraphDirector
from selfplay_graph_flowsteer.director_timeline import (
    director_context_mode,
    persist_context_policy,
    timeline_assistant_content,
    timeline_prefix_audit,
)
from selfplay_graph_flowsteer.llm import LLMResponse, MockBackend
from selfplay_graph_flowsteer.qwen_compat import qwen_request_extra, qwen_vllm_server_args
from selfplay_graph_flowsteer.runtime import MultiAgentRuntime

from .helpers import RecordingExecutor

MODEL = Path("models/Qwen3.5-9B")
TEMPLATE = Path(__file__).resolve().parents[1] / "configs/templates/director_append_only.jinja"


@pytest.fixture(scope="module")
def tokenizer():
    if not MODEL.is_dir():
        pytest.skip("local tokenizer unavailable")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)


def render(tokenizer, messages, *, thinking=True, timeline=True):
    encoded = tokenizer.apply_chat_template(
        messages,
        chat_template=TEMPLATE.read_text(),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=thinking,
        director_append_only=timeline,
    )
    return encoded["input_ids"] if hasattr(encoded, "keys") else encoded


def test_template_default_is_identical_to_original(tokenizer):
    messages = [
        {"role": "system", "content": "System\n"},
        {"role": "user", "content": "Task"},
        {"role": "assistant", "content": "<think>reason</think>\n\nanswer"},
        {"role": "user", "content": "next"},
    ]
    for thinking in (False, True):
        original = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=thinking
        )
        if hasattr(original, "keys"):
            original = original["input_ids"]
        assert render(tokenizer, messages, thinking=thinking, timeline=False) == original


def test_timeline_preserves_reasoning_whitespace_and_binary_prefix(tokenizer):
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Task + Canvas 0"},
    ]
    previous = ()
    # Includes length-truncated thinking and a one-token off/on response with no EOS.
    cases = [
        (True, '  think\n\n</think>\n\n{"action":"add_agent"}<|im_end|>'),
        (False, "on"),
        (True, "unfinished thinking  "),
        (True, '</think>\n\n{"action":"set_output"}<|im_end|>'),
    ]
    for index, (thinking, completion) in enumerate(cases):
        prompt_ids = render(tokenizer, messages, thinking=thinking)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        audit = timeline_prefix_audit(previous, prompt_ids, completion_ids)
        assert audit["timeline_merge_candidate"]
        if previous:
            assert audit["previous_policy_is_exact_prefix"] is True
        content = timeline_assistant_content(tokenizer, prompt_ids, completion_ids)
        assert content.endswith(completion)
        previous = tuple(prompt_ids) + tuple(completion_ids)
        messages.extend(
            [
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"Feedback {index} + new Canvas"},
            ]
        )


def test_prefix_audit_does_not_accept_missing_or_rewritten_tokens():
    assert not timeline_prefix_audit((1, 2, 3), (1, 4), (5,))["timeline_merge_candidate"]
    assert not timeline_prefix_audit((), (), (5,))["timeline_merge_candidate"]
    with pytest.raises(ValueError, match="provider token IDs"):
        timeline_assistant_content(None, (), ())


def test_runtime_tokenizer_wrapper_is_supported(tokenizer):
    from types import SimpleNamespace

    prompt = render(tokenizer, [{"role": "user", "content": "Task"}])
    completion = tokenizer.encode("reason</think>\n\nanswer", add_special_tokens=False)
    assert timeline_assistant_content(SimpleNamespace(tokenizer=tokenizer), prompt, completion) == (
        timeline_assistant_content(tokenizer, prompt, completion)
    )


def test_collection_cannot_mix_context_modes_or_template_versions(monkeypatch, tmp_path):
    import json

    monkeypatch.setenv("SPGFS_DIRECTOR_CONTEXT_MODE", "append_only")
    with pytest.raises(ValueError, match="unmarked"):
        persist_context_policy(tmp_path, resume=True)
    persist_context_policy(tmp_path, resume=False)
    persist_context_policy(tmp_path, resume=True)
    marker = tmp_path / "director_context_policy.json"
    saved = json.loads(marker.read_text())
    monkeypatch.setenv("SPGFS_DIRECTOR_CONTEXT_MODE", "snapshot_dedup")
    with pytest.raises(ValueError, match="differs"):
        persist_context_policy(tmp_path, resume=True)
    monkeypatch.setenv("SPGFS_DIRECTOR_CONTEXT_MODE", "append_only")
    saved["template_sha256"] = "changed"
    marker.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="differs"):
        persist_context_policy(tmp_path, resume=True)


@pytest.mark.parametrize("mode", ["append_only", "delta_timeline"])
def test_graph_director_preserves_each_actual_request_and_audits_tokens(
    monkeypatch, tokenizer, mode
):
    monkeypatch.setenv("SPGFS_DIRECTOR_CONTEXT_MODE", mode)

    class Backend(MockBackend):
        def generate(self, messages, **kwargs):
            self.calls.append({"messages": [dict(m) for m in messages]})
            prompt = render(tokenizer, messages)
            completion = tokenizer.encode(
                "still thinking\n</think>\n\nnot JSON<|im_end|>", add_special_tokens=False
            )
            return LLMResponse(
                text="not JSON",
                model="test",
                raw_reasoning_text="still thinking\n</think>\n\n",
                raw_action_text="not JSON",
                prompt_token_ids=tuple(prompt),
                completion_token_ids=tuple(completion),
                behavior_log_probs=tuple(-1.0 for _ in completion),
                training_eligible=True,
            )

    backend = Backend([])
    canvas = GraphCanvas(task="test", runtime=MultiAgentRuntime(RecordingExecutor()))
    run = GraphDirector(backend=backend, canvas=canvas, tokenizer=tokenizer).run()
    assert len(run.turns) == 4
    for i, turn in enumerate(run.turns):
        audit = turn.action_diagnostics["timeline_prefix_audit"]
        assert audit["timeline_merge_candidate"]
        assert (
            turn.action_diagnostics["director_context_schema"]
            == f"{mode}_action_feedback_history_v1"
        )
        messages = backend.calls[i]["messages"]
        assert sum("Task:\n" in m["content"] for m in messages) == 1
        assert (
            sum("Authoritative Canvas control snapshot:" in m["content"] for m in messages) == i + 1
        )
        if i:
            assert (
                messages[: len(backend.calls[i - 1]["messages"])]
                == backend.calls[i - 1]["messages"]
            )
            assert audit["previous_policy_is_exact_prefix"]
    assert "Bounded recovery" in backend.calls[-1]["messages"][-1]["content"]


def test_opt_in_template_does_not_change_worker_or_proposer_requests(monkeypatch):
    monkeypatch.delenv("SPGFS_DIRECTOR_CONTEXT_MODE", raising=False)
    assert director_context_mode() == "snapshot_dedup"
    assert "--chat-template" not in qwen_vllm_server_args("Qwen3.5-9B", native_tools=False)
    monkeypatch.setenv("SPGFS_DIRECTOR_CONTEXT_MODE", "append_only")
    assert "--chat-template" in qwen_vllm_server_args("Qwen3.5-9B", native_tools=False)
    assert "--chat-template" not in qwen_vllm_server_args("Qwen3.5-9B")
    assert "director_append_only" not in qwen_request_extra()["chat_template_kwargs"]
    assert qwen_request_extra(director_timeline=True)["chat_template_kwargs"][
        "director_append_only"
    ]
    monkeypatch.setenv("SPGFS_DIRECTOR_CONTEXT_MODE", "typo")
    with pytest.raises(ValueError):
        director_context_mode()


def test_timeline_accepts_vllm_text_content_blocks(tokenizer):
    plain = [
        {"role": "user", "content": "Task\n  content"},
        {"role": "assistant", "content": "reason</think>\n\nanswer<|im_end|>"},
    ]
    blocks = [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in plain
    ]
    assert render(tokenizer, blocks) == render(tokenizer, plain)
