"""Native Qwen NQ transport and interrupted-request audit regression tests."""
import io
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer import llm
from selfplay_graph_flowsteer.application import GraphEvaluationIncompleteError
from selfplay_graph_flowsteer.config import ModelGatewayConfig, ModelRoleConfig
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor
from selfplay_graph_flowsteer.worker_usage_ledger import WorkerUsageLedger, worker_usage_scope

from .test_hotpot_usage_budget import gateway_slot as gateway_slot
from .test_nq_corpus_evidence import SearchTool, context, run, supported
from .test_nq_formal_promotion import branch_app
from .test_nq_usage_budget import formal_config

pytestmark = pytest.mark.usefixtures("gateway_slot")


def native(call_id="native-1", query="Curie", arguments=None):
    return SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
        id=call_id, type="function", function=SimpleNamespace(
            name="search", arguments=json.dumps(arguments if arguments is not None else {"query": query})))])


def final(text=None):
    return SimpleNamespace(content=supported() if text is None else text, tool_calls=None)


def qwen(monkeypatch, responses, *, input_count=200, context_limit=32768):
    sent, counted = [], []
    replies = iter(responses)

    def create(**request):
        sent.append(request)
        message, token_in, token_out, *finish = next(replies)
        return SimpleNamespace(id=f"response-{len(sent)}", model="Qwen3.5-9B",
                               usage=SimpleNamespace(prompt_tokens=token_in, completion_tokens=token_out),
                               choices=[SimpleNamespace(message=message,
                                                        finish_reason=finish[0] if finish else "stop")])

    def tokenize(request, **kwargs):
        assert request.full_url == "http://qwen.test/tokenize"
        counted.append(json.loads(request.data))
        return io.StringIO(json.dumps({"count": input_count, "max_model_len": context_limit}))

    monkeypatch.setattr(llm, "urlopen", tokenize)
    backend = object.__new__(llm.OpenAICompatibleBackend)
    backend.config = ModelGatewayConfig(base_url="http://qwen.test/v1", api_key="EMPTY",
        request_profile="qwen", route_name="qwen", network_path="direct", timeout_s=120,
        roles={"worker": ModelRoleConfig(model="Qwen3.5-9B", max_tokens=2048, enable_thinking=False)})
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    backend._client_for_request = lambda: client
    backend.close = lambda: None
    backend.rollout_deadline = None
    return backend, sent, counted


@pytest.fixture
def account(tmp_path):
    ledger = WorkerUsageLedger(tmp_path / "native.sqlite3", question_attempt_id="qwen-native",
                               dataset="nq_open", threshold=240000)
    with llm.request_dataset("nq_open"), worker_usage_scope(ledger, agent_id="a", execution_id="1"):
        yield ledger
    ledger.close()


def test_native_search_continuation_preserves_call_ids_actual_observations_and_usage(monkeypatch, account):
    backend, sent, counted = qwen(monkeypatch, [(native(), 200, 20), (final(), 500, 50)])
    tool = SearchTool()
    result = run(ModelAgentExecutor(backend, tools={"search": tool}, nq_evidence_context=context()))
    assert result.answer == "Polonium" and result.runtime_tool_evidence["nq_corpus"]["valid"]
    assert len(sent) == len(counted) == 2 and len(tool.calls) == 1
    assert account.status()["confirmed_used"] == 770 and account.status()["attempt_count"] == 2
    for request in sent:
        assert request["tools"][0]["function"]["name"] == "search"
        assert request["tool_choice"] == "auto" and request["parallel_tool_calls"] is False
        assert request["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        assert request["max_tokens"] == 2048
        assert 'Alternatively, return one or more Action calls' not in request["messages"][0]["content"]
    assert sent[1]["messages"][-2]["tool_calls"][0]["id"] == "native-1"
    observation = sent[1]["messages"][-1]
    assert observation["role"] == "tool" and observation["tool_call_id"] == "native-1"
    assert "Marie Curie named polonium after Poland." in observation["content"]
    assert counted[1]["tools"] == sent[1]["tools"]
    assert counted[1]["messages"] == sent[1]["messages"]
    assert result.react_trace[0]["action"]["call_id"] == "native-1"
    events = result.backend_request_events
    assert len(events) == 2
    assert events[0]["request_protocol"] == "qwen_nq_native_v1"
    assert events[0]["native_response_call_ids"] == ["native-1"]
    assert events[1]["tool_response_ids"] == events[1]["assistant_tool_call_ids"] == ["native-1"]


def test_text_tool_calls_are_rejected_without_executing_fake_observation(monkeypatch, account):
    fake = '{"action_calls":[{"name":"search","arguments":{"query":"fabricated"}}]}'
    backend, sent, _ = qwen(monkeypatch, [(final(fake), 100, 10), (native(), 200, 20), (final(), 300, 30)])
    tool = SearchTool()
    result = run(ModelAgentExecutor(backend, tools={"search": tool}, nq_evidence_context=context()))
    assert len(sent) == 3 and len(tool.calls) == 1
    assert tool.calls[0]["query"] == "Curie"
    assert any(d.get("rejection_reason") == "native_tool_call_required" and d["actions_executed"] == 0
               for d in result.protocol_diagnostics)
    assert result.runtime_tool_evidence["nq_corpus"]["valid"]
    assert account.status()["confirmed_used"] == 660


def test_invalid_native_arguments_return_tool_error_then_allow_bounded_correction(monkeypatch, account):
    backend, sent, _ = qwen(monkeypatch, [
        (native(arguments={"query": 123}), 100, 10),
        (native("native-2"), 200, 20), (final(), 300, 30),
    ])
    tool = SearchTool()
    result = run(ModelAgentExecutor(backend, tools={"search": tool}, nq_evidence_context=context()))
    assert len(tool.calls) == 1 and len(sent) == 3
    assert sent[1]["messages"][-1]["tool_call_id"] == "native-1"
    assert "invalid_action_arguments" in sent[1]["messages"][-1]["content"]
    assert result.runtime_tool_evidence["nq_corpus"]["valid"]
    assert account.status()["confirmed_used"] == 660


def test_exact_context_capacity_is_separate_from_question_budget(monkeypatch, account):
    backend, sent, counted = qwen(monkeypatch, [(native(), 239000, 0), (final(), 4000, 1000)],
                                 input_count=4090, context_limit=5000)
    result = run(ModelAgentExecutor(backend, tools={"search": SearchTool()}, nq_evidence_context=context()))
    assert result.runtime_tool_evidence["nq_corpus"]["valid"]
    assert account.status()["confirmed_used"] == 244000
    assert [request["max_tokens"] for request in sent] == [846, 846]
    assert len(counted) == 2


def test_context_failure_precedes_generation_and_does_not_create_unknown_usage(monkeypatch, account):
    backend, sent, counted = qwen(monkeypatch, [], input_count=4990, context_limit=5000)
    with pytest.raises(ValueError, match="exact model context"):
        run(ModelAgentExecutor(backend, tools={"search": SearchTool()}, nq_evidence_context=context()))
    assert len(counted) == 1 and not sent
    assert account.status()["attempt_count"] == 0


def test_runtime_owns_truncated_final_recovery_without_text_action_gateway_retry(monkeypatch, account):
    backend, sent, _ = qwen(monkeypatch, [(native(), 100, 10),
                                        (final('{"answer":'), 200, 20, "length"), (final(), 300, 30)])
    result = run(ModelAgentExecutor(backend, tools={"search": SearchTool()}, nq_evidence_context=context()))
    assert result.runtime_tool_evidence["nq_corpus"]["valid"]
    assert len(sent) == 3 and sent[-1]["max_tokens"] == 2048
    assert "tools" not in sent[-1]
    assert account.status()["confirmed_used"] == 660


def test_interrupted_execution_exports_every_original_physical_request_event(tmp_path, monkeypatch):
    app, task, graph, _ = branch_app([])
    app.config = replace(app.config, canvas=formal_config(tmp_path).canvas)
    backend, sent, _ = qwen(monkeypatch, [(native(), None, None),
                                        (native("native-2", "Polonium"), None, None)])
    app.runtime.executor.backend = backend
    try:
        with pytest.raises(GraphEvaluationIncompleteError, match="Worker dispatch stopped"):
            app.evaluate_graph(task, graph, seed=0)
        usage = app.last_graph_evaluation["worker_usage"]
        stopped = app.last_graph_evaluation["execution"]["execution_events"][-1]
        events = stopped["backend_request_events"]
        assert stopped["reason"] == "worker_usage_unsettled_limit"
        assert len(events) == len(sent) == usage["attempt_count"] == 2
        assert usage["unsettled_attempt_count"] == 2
        assert len({event["worker_usage_attempt_id"] for event in events}) == 2
        assert all(event["request_role"] == "worker" for event in events)
        assert all(event["completion_usage"]["token_in"] is None
                   and event["completion_usage"]["token_out"] is None for event in events)
    finally:
        app.close()
