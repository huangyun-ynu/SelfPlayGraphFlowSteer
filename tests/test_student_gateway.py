from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer import llm
from selfplay_graph_flowsteer.action_protocol import ActionSpec
from selfplay_graph_flowsteer.application import _api_key
from selfplay_graph_flowsteer.config import ModelGatewayConfig, ModelRoleConfig


def test_private_key_file(tmp_path):
    path = tmp_path / "key"
    path.write_text("test-key\n")
    assert _api_key({"api_key_file": str(path)}) == "test-key"


def test_responses_text_omits_tools_and_serializes_history(monkeypatch):
    sent = []
    def create(client, config, request, deadline):
        sent.append(request)
        return SimpleNamespace(output_text='{"action_calls": []}', output=[], usage=None)
    monkeypatch.setattr(llm, "_openai_response_create", create)
    backend = object.__new__(llm.OpenAICompatibleBackend)
    backend.config = ModelGatewayConfig(request_profile="responses_text")
    backend._client_for_request = lambda: None
    action = ActionSpec(name="echo", description="Echo", parameters={"type": "object"})
    backend._generate_response(
        [{"role": "tool", "tool_call_id": "c1", "content": "OK"}],
        role="worker", role_config=ModelRoleConfig(api_surface="responses"),
        actions=[action], deadline=None,
    )
    assert "tools" not in sent[0]
    assert "tool_choice" not in sent[0]
    assert "stream" not in sent[0]
    assert "echo" in sent[0]["input"][0]["content"]
    assert sent[0]["input"][1]["role"] == "user"
    assert "OK" in sent[0]["input"][1]["content"]


def test_text_gateway_submits_only_once(monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("test failure")
    monkeypatch.setattr(llm, "_openai_response_attempt", fail)
    with pytest.raises(RuntimeError):
        llm._openai_response_create(None, ModelGatewayConfig(request_profile="responses_text"), {}, None)
    assert len(calls) == 1
