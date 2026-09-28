"""Reported SWE usage is charged per physical request and shared per question."""

from contextlib import contextmanager
from types import SimpleNamespace
import json

import pytest

from selfplay_graph_flowsteer import llm
from selfplay_graph_flowsteer.endpoint_pool import EndpointPoolBackend
from selfplay_graph_flowsteer.worker_usage_ledger import (
    UsageDispatchStopped, WorkerUsageLedger, worker_usage_scope,
)


def _ledger(tmp_path, *, threshold=350000):
    return WorkerUsageLedger(
        tmp_path / "usage.sqlite3", question_attempt_id="run:question:seed:attempt",
        threshold=threshold, max_unsettled_attempts=2,
    )


def test_last_legal_request_can_cross_threshold_and_recovery_preserves_it(tmp_path):
    ledger = _ledger(tmp_path)
    first = ledger.begin(route="gpt", agent_id="a", execution_id="1", request={"model": "gpt"})
    ledger.settle(first, input_tokens=300000, output_tokens=40000)
    second = ledger.begin(route="student", agent_id="b", execution_id="2", request={"model": "student"})
    ledger.settle(second, input_tokens=14000, output_tokens=6000)
    assert ledger.status()["confirmed_used"] == 360000
    assert ledger.status()["confirmed_overshoot"] == 10000
    assert ledger.dispatches_valid()
    with pytest.raises(UsageDispatchStopped, match="worker_usage_threshold_reached"):
        ledger.begin(route="gpt", agent_id="c", execution_id="3", request={})
    ledger.close()
    reopened = _ledger(tmp_path)
    assert reopened.status()["confirmed_used"] == 360000
    assert reopened.status()["attempt_count"] == 2
    reopened.close()


def test_unknown_usage_bounded_failover_and_reconciliation(tmp_path):
    ledger = _ledger(tmp_path, threshold=100)
    first = ledger.begin(route="gpt", agent_id="a", execution_id="1", request={})
    ledger.mark_unknown(first)
    second = ledger.begin(route="student", agent_id="a", execution_id="1", request={})
    ledger.settle(second, input_tokens=30, output_tokens=10)
    assert ledger.status()["confirmed_used"] == 40
    assert not ledger.status()["usage_complete"]
    third = ledger.begin(route="gpt", agent_id="a", execution_id="2", request={})
    ledger.mark_unknown(third)
    with pytest.raises(UsageDispatchStopped, match="worker_usage_unsettled_limit"):
        ledger.begin(route="student", agent_id="a", execution_id="2", request={})
    ledger.reconcile(first, input_tokens=20, output_tokens=5)
    ledger.reconcile(first, input_tokens=20, output_tokens=5)
    assert ledger.status()["confirmed_used"] == 65
    assert ledger.status()["unsettled_attempt_count"] == 1
    with pytest.raises(ValueError, match="conflicting"):
        ledger.reconcile(first, input_tokens=21, output_tokens=5)
    ledger.close()


@pytest.mark.parametrize("usage", [None, SimpleNamespace(prompt_tokens=None, completion_tokens=4)])
def test_gateway_missing_usage_is_unknown_not_zero(tmp_path, monkeypatch, usage):
    ledger = _ledger(tmp_path, threshold=100)
    calls = []

    @contextmanager
    def slot(*args, **kwargs):
        yield SimpleNamespace(route="gpt", timeout_s=10, request_started_monotonic=0,
                              request_budget_s=10, queue_wait_s=0, priority="primary")

    monkeypatch.setattr(llm, "_request_slot", slot)

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id="response-1", usage=usage)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    config = SimpleNamespace(route_name="gpt", stream=False)
    with worker_usage_scope(ledger, agent_id="a", execution_id="1"):
        llm._openai_completion_attempt(client, config, {"model": "gpt", "messages": []},
                                       None, attempt=1, request_budget_cap_s=10)
    assert len(calls) == 1
    assert ledger.status()["unsettled_attempt_count"] == 1
    assert ledger.status()["confirmed_used"] == (4 if usage else 0)
    ledger.close()


def test_gateway_stops_before_provider_call_and_pool_does_not_failover(tmp_path, monkeypatch):
    ledger = _ledger(tmp_path, threshold=1)
    attempt = ledger.begin(route="gpt", agent_id="a", execution_id="1", request={})
    ledger.settle(attempt, input_tokens=1, output_tokens=0)
    calls = []

    @contextmanager
    def slot(*args, **kwargs):
        yield SimpleNamespace(route="gpt", timeout_s=10, request_started_monotonic=0,
                              request_budget_s=10, queue_wait_s=0, priority="primary")

    monkeypatch.setattr(llm, "_request_slot", slot)

    class Backend:
        def __init__(self, name):
            self.config = SimpleNamespace(base_url="https://" + name, timeout_s=10,
                                          route_name=name)

        def generate(self, messages, **kwargs):
            return llm._openai_completion_attempt(
                SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                    create=lambda **request: calls.append(request)))),
                SimpleNamespace(route_name=self.config.route_name, stream=False),
                {"model": self.config.route_name, "messages": messages}, None,
                attempt=1, request_budget_cap_s=10,
            )

    pool = EndpointPoolBackend("gpt", {"gpt": Backend("gpt"), "student": Backend("student")}, tmp_path)
    with worker_usage_scope(ledger, agent_id="a", execution_id="2"):
        with pytest.raises(llm.WorkerUsageDispatchStopped):
            pool.generate([{"role": "user", "content": "hello"}], role="worker")
    assert not calls
    assert ledger.status()["attempt_count"] == 1
    ledger.close()


def test_gemini_native_dispatch_uses_same_reported_usage_account(tmp_path, monkeypatch):
    ledger = _ledger(tmp_path, threshold=100)
    calls = []

    @contextmanager
    def slot(*args, **kwargs):
        yield SimpleNamespace(route="gemini", timeout_s=10, request_started_monotonic=0,
                              request_budget_s=10, queue_wait_s=0, priority="primary")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({
                "candidates": [{"content": {"parts": [{"text": "done"}]}}],
                "usageMetadata": {"promptTokenCount": 6, "candidatesTokenCount": 4},
            }).encode()

    monkeypatch.setattr(llm, "_request_slot", slot)
    monkeypatch.setattr(llm, "urlopen", lambda *args, **kwargs: (calls.append(args), Response())[1])
    config = SimpleNamespace(
        route_name="gemini", timeout_s=10, base_url="https://gemini.example",
        api_key="fixture", roles={"worker": SimpleNamespace(
            model="gemini", temperature=0, max_tokens=20, system_prompt="",
        )},
    )
    backend = llm.GeminiNativeBackend(config)
    with worker_usage_scope(ledger, agent_id="a", execution_id="1"):
        assert backend.generate([{"role": "user", "content": "hello"}], role="worker").text == "done"
    assert len(calls) == 1
    assert ledger.status()["confirmed_used"] == 10
    assert ledger.status()["usage_complete"]
    ledger.close()
