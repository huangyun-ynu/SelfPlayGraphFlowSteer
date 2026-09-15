from __future__ import annotations

import inspect
import json
from collections import deque
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.backend_failures import BackendRequestError
from selfplay_graph_flowsteer.config import ModelGatewayConfig, ModelRoleConfig
from selfplay_graph_flowsteer.llm import OpenAICompatibleBackend

SCHEMA = {
    "type": "object",
    "properties": {"decision": {"type": "string", "enum": ["KEEP"]}},
    "required": ["decision"],
    "additionalProperties": False,
}
MESSAGES = [{"role": "user", "content": "Return the review decision."}]


def completion(text='{"decision":"KEEP"}', *, finish_reason="stop"):
    return SimpleNamespace(
        model="Qwen3.5-9B",
        prompt_token_ids=[10, 11],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None),
                finish_reason=finish_reason,
                token_ids=[12],
                logprobs=SimpleNamespace(content=[SimpleNamespace(logprob=-0.2)]),
            )
        ],
    )


def fake_backend(*responses, surface="chat_completions"):
    pending = deque(responses or [completion()])
    requests = []

    def create(**request):
        requests.append(request)
        response = pending.popleft()
        if isinstance(response, Exception):
            raise response
        return response

    backend = object.__new__(OpenAICompatibleBackend)
    backend.config = ModelGatewayConfig(
        base_url="http://skill-refiner.test/v1",
        sampling_seed=13,
        roles={"worker": ModelRoleConfig(api_surface=surface, max_tokens=1024, top_k=20)},
    )
    backend.rollout_deadline = None
    backend.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return backend, requests


def test_refiner_schema_reaches_client_without_changing_regular_generation():
    backend, requests = fake_backend(completion(), completion())

    response = backend.generate_json(MESSAGES, role="skill-distiller", schema=SCHEMA)
    regular = backend.generate(MESSAGES, role="skill-distiller")

    assert json.loads(response.text) == {"decision": "KEEP"}
    assert response.metadata["backend_request_events"]
    assert regular.text == response.text
    assert requests[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "pats_skill_review", "strict": True, "schema": SCHEMA},
    }
    assert "response_format" not in requests[1]
    assert requests[0]["seed"] == requests[1]["seed"] == 13
    assert requests[0]["extra_body"] == requests[1]["extra_body"]
    assert "logprobs" not in requests[0]
    assert "return_token_ids" not in requests[0]["extra_body"]
    # The guard preserves the expanded signature for capability inspection.
    assert "response_json_schema" in inspect.signature(backend.generate).parameters


def test_proposer_generation_keeps_original_probability_request_and_response():
    backend, requests = fake_backend()

    response = backend.generate(MESSAGES, role="proposer")

    assert "response_format" not in requests[0]
    assert requests[0]["logprobs"] is True
    assert requests[0]["top_logprobs"] == 0
    assert requests[0]["extra_body"]["return_token_ids"] is True
    assert response.prompt_token_ids == (10, 11)
    assert response.completion_token_ids == (12,)
    assert response.behavior_log_probs == (-0.2,)
    assert response.token_provenance == "provider_prompt_and_completion_token_ids_and_logprobs"


@pytest.mark.parametrize("role", ["graph-director", "proposer", "worker"])
def test_schema_capability_rejects_other_roles_before_request_or_director_guard(
    role, tmp_path, monkeypatch
):
    backend, requests = fake_backend()
    guard_dir = tmp_path / "guard"
    monkeypatch.setenv("SPGFS_DIRECTOR_CONNECTION_GUARD_DIR", str(guard_dir))

    with pytest.raises(ValueError, match="limited to skill-distiller"):
        backend.generate_json(MESSAGES, role=role, schema=SCHEMA)

    assert requests == []
    assert not guard_dir.exists()


@pytest.mark.parametrize("surface", ["responses", "unsupported_surface"])
def test_unsupported_surface_reports_capability_without_sending(surface):
    backend, requests = fake_backend(surface=surface)

    with pytest.raises(NotImplementedError, match="chat-completions"):
        backend.generate_json(MESSAGES, role="skill-distiller", schema=SCHEMA)

    assert requests == []
    if surface == "responses":
        sentinel = object()
        backend._generate_response = lambda *args, **kwargs: sentinel
        assert backend.generate(MESSAGES, role="skill-distiller") is sentinel


def test_truncated_refiner_retry_retains_the_exact_schema():
    backend, requests = fake_backend(
        completion('{"decision":', finish_reason="length"), completion()
    )

    response = backend.generate_json(MESSAGES, role="skill-distiller", schema=SCHEMA)

    assert json.loads(response.text) == {"decision": "KEEP"}
    assert len(requests) == 2
    assert requests[0]["response_format"] == requests[1]["response_format"]
    assert len(response.metadata["generation_attempts"]) == 2


def test_schema_request_rejection_propagates_without_free_text_fallback():
    class InvalidSchemaRequest(RuntimeError):
        status_code = 400

    error = InvalidSchemaRequest("provider rejected response schema")
    backend, requests = fake_backend(error)

    with pytest.raises(BackendRequestError) as caught:
        backend.generate_json(MESSAGES, role="skill-distiller", schema=SCHEMA)

    assert caught.value.__cause__ is error
    assert caught.value.classification.kind == "invalid_request"
    assert len(requests) == 1
    assert requests[0]["response_format"]["type"] == "json_schema"


def test_none_schema_cannot_silently_request_free_text():
    backend, requests = fake_backend()

    with pytest.raises(TypeError, match="must be an object"):
        backend.generate_json(MESSAGES, role="skill-distiller", schema=None)

    assert requests == []
