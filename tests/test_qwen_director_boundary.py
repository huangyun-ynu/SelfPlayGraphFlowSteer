"""Regression coverage for the tool-only vLLM parser swallowing </think>."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.config import ModelGatewayConfig, ModelRoleConfig
from selfplay_graph_flowsteer.llm import OpenAICompatibleBackend
from selfplay_graph_flowsteer.qwen_compat import recover_qwen_policy_parts, response_policy_parts


def entries(*pieces):
    return [SimpleNamespace(bytes=list(piece), logprob=-0.25) for piece in pieces]


def recover(content, pieces, *, token_ids=None, **message_fields):
    return recover_qwen_policy_parts(
        SimpleNamespace(content=content, **message_fields),
        completion_token_ids=list(range(len(pieces))) if token_ids is None else token_ids,
        content_logprobs=pieces,
    )


def test_restore_only_sampled_boundary_and_preserve_unicode_and_whitespace():
    thought = '  分析中引用 {"action":"delete_agent"}\n'
    action = '\n\n{"action":"finish"}  '
    # Token bytes may divide a Unicode code point; decode the whole stream.
    thought_bytes = thought.encode()
    pieces = entries(thought_bytes[:3], thought_bytes[3:], b"</think>", action.encode(), b"<|im_end|>")
    assert recover(thought + action, pieces) == (thought + "</think>", action)


@pytest.mark.parametrize("stop", [b"", b"<|im_end|>", b"<|endoftext|>"])
def test_known_terminal_stop_does_not_enter_action(stop):
    pieces = entries(b"thought\n", b"</think>", b'\n{"action":"finish"}')
    if stop:
        pieces.extend(entries(stop))
    assert recover('thought\n\n{"action":"finish"}', pieces) == (
        "thought\n</think>", '\n{"action":"finish"}'
    )


@pytest.mark.parametrize("fault", [
    "no_delimiter", "literal_delimiter", "duplicate_delimiter", "missing_bytes",
    "invalid_bytes", "invalid_utf8", "missing_logprob", "wrong_token_count",
    "no_token_ids", "message_mismatch", "unknown_stop", "nonterminal_stop",
])
def test_unattested_boundaries_are_not_recovered(fault):
    content = 'thought\n{"action":"finish"}'
    pieces = entries(b"thought", b"</think>", b'\n{"action":"finish"}')
    token_ids = None
    if fault == "no_delimiter":
        pieces.pop(1)
    elif fault == "literal_delimiter":
        pieces[1:2] = entries(b"</", b"think>")
    elif fault == "duplicate_delimiter":
        pieces.insert(1, entries(b"</think>")[0])
    elif fault == "missing_bytes":
        pieces[0].bytes = None
    elif fault == "invalid_bytes":
        pieces[0].bytes = [300]
    elif fault == "invalid_utf8":
        pieces[0].bytes = [255]
    elif fault == "missing_logprob":
        pieces[0].logprob = None
    elif fault == "wrong_token_count":
        token_ids = [1]
    elif fault == "no_token_ids":
        token_ids = []
    elif fault == "message_mismatch":
        content = 'different\n{"action":"finish"}'
    elif fault == "unknown_stop":
        pieces.extend(entries(b"<unknown_stop>"))
    elif fault == "nonterminal_stop":
        pieces.insert(2, entries(b"<|im_end|>")[0])
    assert recover(content, pieces, token_ids=token_ids) is None


def test_provider_reasoning_channel_and_inline_boundary_remain_authoritative():
    pieces = entries(b"thought", b"</think>", b'\n{"action":"finish"}')
    assert recover('thought\n{"action":"finish"}', pieces, reasoning="provider reasoning") is None
    assert recover('thought</think>\n{"action":"finish"}', pieces) is None


def test_unfinished_reasoning_with_json_is_never_promoted_to_an_action():
    thought = 'I could use {"action":"finish"}'
    assert recover(thought, entries(thought.encode())) is None
    assert response_policy_parts(SimpleNamespace(content=thought), thinking_prefilled=True) == (thought, "")


def fake_completion(content, pieces, *, reasoning=None):
    return SimpleNamespace(
        model="Qwen3.5-9B", prompt_token_ids=[17, 18],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=len(pieces)),
        choices=[SimpleNamespace(
            finish_reason="stop", token_ids=list(range(len(pieces))),
            logprobs=SimpleNamespace(content=pieces),
            message=SimpleNamespace(content=content, reasoning=reasoning, tool_calls=None),
        )],
    )


def gateway(monkeypatch, completions, *, thinking=True, profile="qwen"):
    import selfplay_graph_flowsteer.llm as llm

    requests = []

    def create(**request):
        requests.append(request)
        assert len(requests) <= len(completions), "Unexpected Director regeneration"
        return completions[len(requests) - 1]

    class Counted:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"count":100}'

    monkeypatch.setattr(llm, "urlopen", lambda *args, **kwargs: Counted())
    monkeypatch.delenv("SPGFS_DIRECTOR_DYNAMIC_BUDGET", raising=False)
    monkeypatch.delenv("SPGFS_QWEN_DIRECTOR_ACTION_RETRY", raising=False)
    backend = object.__new__(OpenAICompatibleBackend)
    backend.config = ModelGatewayConfig(
        base_url="http://qwen.test/v1", request_profile=profile,
        roles={"graph-director": ModelRoleConfig(enable_thinking=thinking)},
    )
    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    backend.rollout_deadline = None
    return backend, requests


def test_actual_vllm_response_executes_first_action_with_one_contiguous_sample(monkeypatch):
    from openai.types.chat import ChatCompletion

    recorded = json.loads((Path(__file__).parent / "fixtures/qwen_director_parser_response.json").read_text())
    completion = ChatCompletion.model_validate(recorded)
    assert completion.choices[0].message.content.count("</think>") == 0
    backend, requests = gateway(monkeypatch, [completion])
    response = backend.generate([{"role": "user", "content": "Finish"}], role="graph-director")
    assert len(requests) == len(response.metadata["generation_attempts"]) == 1
    assert response.text == '{"action":"finish"}'
    assert response.raw_reasoning_text.endswith("</think>")
    assert response.raw_action_text == '\n\n{"action":"finish"}'
    assert response.metadata["qwen_thinking_boundary_restored"] is True
    assert response.metadata["split_director_action_retry"] is False
    assert response.training_eligible is True
    assert response.completion_token_ids == tuple(completion.choices[0].token_ids)
    assert len(response.behavior_log_probs) == len(response.completion_token_ids)
    assert response.prompt_token_ids == tuple(completion.prompt_token_ids)
    assert response.token_in == completion.usage.prompt_tokens
    assert response.token_out == completion.usage.completion_tokens


def test_malformed_visible_action_is_not_regenerated_or_replaced_with_thought_json(monkeypatch):
    from selfplay_graph_flowsteer.actions import ActionParser

    thought = 'Example {"action":"finish"}\n'
    action = '\n{"action":'
    pieces = entries(thought.encode(), b"</think>", action.encode())
    backend, requests = gateway(monkeypatch, [fake_completion(thought + action, pieces)])
    response = backend.generate([{"role": "user", "content": "Decide"}], role="graph-director")
    assert len(requests) == 1
    assert response.raw_action_text == action
    assert not ActionParser().parse_policy_output(response.raw_action_text).action_text


def test_true_unfinished_reasoning_keeps_bounded_nontrainable_retry_and_all_usage(monkeypatch):
    thought = 'I could use {"action":"finish"}'
    action = '{"action":"finish"}'
    first = fake_completion(thought, entries(thought.encode()))
    second = fake_completion(action, entries(action.encode()))
    backend, requests = gateway(monkeypatch, [first, second])
    response = backend.generate([{"role": "user", "content": "Decide"}], role="graph-director")
    assert len(requests) == len(response.metadata["generation_attempts"]) == 2
    assert requests[1]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert response.metadata["qwen_thinking_boundary_restored"] is False
    assert response.metadata["split_director_action_retry"] is True
    assert response.training_eligible is False
    assert response.raw_reasoning_text == thought
    assert response.raw_action_text == action
    assert (response.token_in, response.token_out) == (4, 2)


@pytest.mark.parametrize("thinking,profile", [(False, "qwen"), (True, "generic")])
def test_recovery_is_scoped_to_qwen_thinking_director(monkeypatch, thinking, profile):
    content = 'thought\n{"action":"finish"}'
    completion = fake_completion(content, entries(b"thought", b"</think>", b'\n{"action":"finish"}'))
    backend, requests = gateway(monkeypatch, [completion], thinking=thinking, profile=profile)
    response = backend.generate([{"role": "user", "content": "Decide"}], role="graph-director")
    assert len(requests) == 1
    assert response.metadata["qwen_thinking_boundary_restored"] is False
    assert response.text == content


def test_proper_separate_reasoning_channel_does_not_trigger_recovery_or_retry(monkeypatch):
    action = '{"action":"finish"}'
    completion = fake_completion(action, entries(b"thought", b"</think>", action.encode()), reasoning="thought")
    backend, requests = gateway(monkeypatch, [completion])
    response = backend.generate([{"role": "user", "content": "Decide"}], role="graph-director")
    assert len(requests) == 1
    assert response.raw_reasoning_text == "thought"
    assert response.raw_action_text == action
    assert response.metadata["qwen_thinking_boundary_restored"] is False
    assert response.training_eligible is True
