from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.application import load_adaptive_config
from selfplay_graph_flowsteer.endpoint_pool import EndpointPoolBackend


class Backend:
    def __init__(self, name):
        self.config = SimpleNamespace(base_url="https://" + name, timeout_s=180.0, route_name=name)
        self.deadline = None

    def generate(self, messages, **kwargs):
        return SimpleNamespace(metadata={}, text=messages[0]["content"])

    def set_deadline_context(self, deadline):
        self.deadline = deadline


def test_pool_balances_independent_instances_and_keeps_logical_route(tmp_path):
    members = {"a": Backend("a"), "b": Backend("b")}
    pools = [EndpointPoolBackend("gpt", members, tmp_path) for _ in range(20)]
    with ThreadPoolExecutor(10) as executor:
        responses = list(
            executor.map(lambda pool: pool.generate([{"content": "same"}], role="worker"), pools)
        )
    assert sum(r.metadata["endpoint_pool_member"] == "a" for r in responses) == 10
    assert all(r.metadata["logical_route"] == "gpt" and r.text == "same" for r in responses)
    deadline = object()
    pools[0].set_deadline_context(deadline)
    assert all(backend.deadline is deadline for backend in members.values())


def test_cross_channel_replay_strips_opaque_provider_state(tmp_path):
    class CapturingBackend(Backend):
        def generate(self, messages, **kwargs):
            self.received = messages
            return SimpleNamespace(
                metadata={}, assistant_message={"role": "assistant", "content": "OK"}
            )

    a, b = CapturingBackend("a"), CapturingBackend("b")
    pool = EndpointPoolBackend("gemini", {"a": a, "b": b}, tmp_path)
    message = {
        "role": "assistant",
        "content": "",
        "action_calls": [{"call_id": "c", "name": "probe", "arguments": {}}],
        "_provider_payloads": {
            "openai_responses": [{"type": "reasoning", "encrypted_content": "opaque"}]
        },
        "_endpoint_pool_member": "a",
    }
    pool.generate([message], role="worker")
    result = pool.generate([message], role="worker")
    assert "_provider_payloads" in a.received[0]
    assert "_provider_payloads" not in b.received[0]
    assert b.received[0]["action_calls"] == message["action_calls"]
    assert "_provider_payloads" in message
    assert result.assistant_message["_endpoint_pool_member"] == "b"


def _transient_failure():
    from selfplay_graph_flowsteer.backend_failures import (
        BackendFailureClassification,
        BackendRequestError,
    )

    return BackendRequestError(
        BackendFailureClassification(
            backend_failure=True,
            origin="upstream",
            kind="upstream_5xx",
            retryable=True,
            counts_toward_route_circuit=True,
            status_code=503,
        )
    )


def _permanent_endpoint_failure():
    from selfplay_graph_flowsteer.backend_failures import (
        BackendFailureClassification,
        BackendRequestError,
    )

    return BackendRequestError(
        BackendFailureClassification(
            backend_failure=True,
            origin="route_configuration",
            kind="auth_failure",
            retryable=False,
            counts_toward_route_circuit=True,
            disable_route=True,
            status_code=401,
        )
    )


@pytest.mark.parametrize("capacity_failure", [False, True])
def test_failover_retries_same_request_and_reuses_recovered_endpoint(tmp_path, capacity_failure):
    calls = []

    class FailOnce(Backend):
        def generate(self, messages, **kwargs):
            calls.append((self.config.base_url, messages, kwargs))
            if len(calls) == 1:
                if capacity_failure:
                    error = RuntimeError("All available accounts exhausted")
                    error.status_code = 403
                    raise error
                raise _transient_failure()
            return super().generate(messages, **kwargs)

    a, b = FailOnce("a"), FailOnce("b")
    pool = EndpointPoolBackend("gpt", {"a": a, "b": b}, tmp_path)
    messages = [{"role": "tool", "content": "already executed action result"}]
    result = pool.generate(messages, role="worker")
    assert [c[0] for c in calls] == ["https://a", "https://b"]
    assert calls[0][1:] == calls[1][1:]
    assert result.metadata["endpoint_pool_failovers"] == 1
    pool.generate(messages, role="worker")  # normal round-robin starts at b
    recovered = pool.generate(messages, role="worker")
    assert recovered.metadata["endpoint_pool_member"] == "a"
    assert len(pool.members) == 2
    import json

    audit = [
        json.loads(line)
        for line in pool.counter_path.with_suffix(".events.jsonl").read_text().splitlines()
    ]
    assert audit[0]["status"] == "failed" and audit[1]["status"] == "success"
    assert audit[0]["request_id"] == audit[1]["request_id"]


def test_failover_is_bounded_and_preserves_permanent_errors(tmp_path):
    calls = []

    class Failing(Backend):
        def generate(self, messages, **kwargs):
            calls.append(self.config.base_url)
            raise _transient_failure()

    pool = EndpointPoolBackend("gpt", {k: Failing(k) for k in ["a", "b", "c"]}, tmp_path)
    with pytest.raises(type(_transient_failure())) as error:
        pool.generate([{"content": "same"}], role="worker")
    assert len(calls) == 3
    assert len(error.value.endpoint_pool_attempts) == 3
    calls.clear()

    class Invalid(Failing):
        def generate(self, messages, **kwargs):
            calls.append("invalid")
            raise ValueError("local invalid request")

    pool = EndpointPoolBackend("other", {"a": Invalid("invalid"), "b": Failing("b")}, tmp_path)
    with pytest.raises(ValueError, match="local invalid"):
        pool.generate([{"content": "same"}], role="worker")
    assert calls == ["invalid"]


def test_pool_failover_skips_one_permanently_failed_endpoint(tmp_path):
    class PermanentlyFailing(Backend):
        def generate(self, messages, **kwargs):
            raise _permanent_endpoint_failure()

    pool = EndpointPoolBackend(
        "gpt",
        {"a": PermanentlyFailing("a"), "b": Backend("b")},
        tmp_path,
        pool_retry_attempts=2,
    )

    result = pool.generate([{"content": "same"}], role="worker")

    assert result.metadata["endpoint_pool_member"] == "b"
    assert result.metadata["endpoint_pool_failovers"] == 1
    assert result.metadata["endpoint_pool_retries"] == 0


def test_pool_retries_all_members_then_reports_one_logical_failure(tmp_path):
    from selfplay_graph_flowsteer.backend_failures import BackendRequestError

    calls = []

    class Failing(Backend):
        def generate(self, messages, **kwargs):
            calls.append(self.config.route_name)
            raise _transient_failure()

    pool = EndpointPoolBackend(
        "gpt",
        {"a": Failing("a"), "b": Failing("b")},
        tmp_path,
        pool_retry_attempts=2,
        retry_backoff_s=0,
    )

    with pytest.raises(BackendRequestError) as captured:
        pool.generate([{"content": "same"}], role="worker")

    error = captured.value
    assert calls == ["a", "b", "b", "a", "a", "b"]
    assert error.classification.route == "gpt"
    assert error.classification.disable_route
    assert not error.classification.retryable
    assert len(error.endpoint_pool_attempts) == 6
    assert len(error.request_events) == 1
    assert error.request_events[0]["route"] == "gpt"


def test_failover_cannot_extend_request_budget(tmp_path, monkeypatch):
    import selfplay_graph_flowsteer.endpoint_pool as module

    clock = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    calls = []

    class SlowFailure(Backend):
        def generate(self, messages, **kwargs):
            calls.append(self.config.base_url)
            clock[0] += 5
            raise _transient_failure()

    a, b = SlowFailure("a"), SlowFailure("b")
    a.config.timeout_s = b.config.timeout_s = 5
    pool = EndpointPoolBackend("gpt", {"a": a, "b": b}, tmp_path)
    with pytest.raises(type(_transient_failure())):
        pool.generate([{"content": "same"}], role="worker")
    assert calls == ["https://a"]


def test_failover_scope_disables_nested_retries_and_restores_context():
    import time

    from selfplay_graph_flowsteer.llm import (
        _ENDPOINT_FAILOVER_ACTIVE,
        _logical_request_budget_s,
        _retry_backend_request,
        endpoint_failover_scope,
    )

    with endpoint_failover_scope(time.monotonic() + 2):
        assert not _retry_backend_request(
            _transient_failure(),
            attempt=1,
            sequence_started=time.monotonic(),
            sequence_budget_s=100,
        )
        assert 0 < _logical_request_budget_s(SimpleNamespace(timeout_s=100), None) <= 2
    assert not _ENDPOINT_FAILOVER_ACTIVE.get()


def test_pool_caps_member_queue_wait_and_fails_over_immediately(tmp_path):
    from selfplay_graph_flowsteer.backend_failures import (
        BackendFailureClassification,
        BackendRequestError,
    )
    from selfplay_graph_flowsteer.llm import _ENDPOINT_QUEUE_WAIT_CAP_S

    observed_caps = []

    class SaturatedThenHealthy(Backend):
        def generate(self, messages, **kwargs):
            observed_caps.append((self.config.route_name, _ENDPOINT_QUEUE_WAIT_CAP_S.get()))
            if self.config.route_name == "a":
                raise BackendRequestError(
                    BackendFailureClassification(
                        backend_failure=True,
                        origin="local_queue",
                        kind="queue_timeout",
                        retryable=True,
                        counts_toward_route_circuit=False,
                        stage="backend_queue",
                        route="a",
                    )
                )
            return super().generate(messages, **kwargs)

    pool = EndpointPoolBackend(
        "gpt",
        {"a": SaturatedThenHealthy("a"), "b": SaturatedThenHealthy("b")},
        tmp_path,
        member_queue_wait_s=0.5,
    )

    result = pool.generate([{"content": "same"}], role="worker")

    assert observed_caps == [("a", 0.5), ("b", 0.5)]
    assert result.metadata["endpoint_pool_member"] == "b"
    assert result.metadata["endpoint_pool_failovers"] == 1
    assert _ENDPOINT_QUEUE_WAIT_CAP_S.get() is None


def test_cancelled_rollout_does_not_try_another_endpoint(tmp_path):
    from selfplay_graph_flowsteer.deadline import WorkerWallClockLimitExceeded

    cancelled = [False]
    calls = []

    class Deadline:
        def check(self, stage):
            if cancelled[0]:
                raise WorkerWallClockLimitExceeded(
                    "cancelled", reason="backend_circuit_open", stage=stage, elapsed_s=1, idle_s=0
                )

        def request_budget_s(self, *args, **kwargs):
            return 100

    class Cancelling(Backend):
        def generate(self, messages, **kwargs):
            calls.append(self.config.base_url)
            cancelled[0] = True
            raise _transient_failure()

    pool = EndpointPoolBackend("gpt", {"a": Cancelling("a"), "b": Cancelling("b")}, tmp_path)
    pool.set_deadline_context(Deadline())
    with pytest.raises(WorkerWallClockLimitExceeded):
        pool.generate([{"content": "same"}], role="worker")
    assert calls == ["https://a"]


def test_failed_channel_gets_fresh_bounded_attempt_without_charging_task(tmp_path, monkeypatch):
    import time

    from selfplay_graph_flowsteer.deadline import RolloutDeadline

    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    calls = []

    class SlowFirst(Backend):
        def generate(self, messages, **kwargs):
            calls.append(self.config.base_url)
            if self.config.base_url == "https://a":
                clock[0] += 5
                raise _transient_failure()
            clock[0] += 2
            return super().generate(messages, **kwargs)

    a, b = SlowFirst("a"), SlowFirst("b")
    a.config.timeout_s = b.config.timeout_s = 5
    pool = EndpointPoolBackend("gpt", {"a": a, "b": b}, tmp_path)
    deadline = RolloutDeadline(30, 30, 5, started_monotonic=0, exclude_failed_request_time=True)
    pool.set_deadline_context(deadline)
    result = pool.generate([{"content": "existing tool result"}], role="worker")
    assert calls == ["https://a", "https://b"]
    assert result.metadata["endpoint_pool_failovers"] == 1
    assert deadline.diagnostics()["elapsed_s"] == 7
    assert deadline.hard_remaining_s("test") == 28


def test_application_installs_deadline_on_pool_wrapper(monkeypatch, tmp_path):
    from dataclasses import replace

    import selfplay_graph_flowsteer.application as app_module
    from selfplay_graph_flowsteer.deadline import RolloutDeadline

    root = Path(__file__).resolve().parents[1]
    base = load_adaptive_config(root / "configs/mock.toml")
    config = replace(
        base,
        additional_runtimes={
            "secondary": replace(base.runtime, base_url="http://secondary.invalid/v1")
        },
        runtime_endpoint_pools={"default": ("default", "secondary")},
        route_health_path=tmp_path / "health.json",
    )
    monkeypatch.setattr(
        app_module,
        "_create_runtime_backend",
        lambda runtime, **kwargs: Backend(kwargs.get("route_name", "test")),
    )
    app = app_module.create_adaptive_application(
        config, director_backend=Backend("director"), distiller_backend=Backend("support")
    )
    try:
        deadline = RolloutDeadline(900, 300, 120, exclude_failed_request_time=True)
        app.set_rollout_deadline(deadline)
        pools = [b for b in app.owned_backends if isinstance(b, EndpointPoolBackend)]
        assert len(pools) == 1
        assert pools[0]._deadline.get() is deadline
        assert all(b.deadline is deadline for _, b in pools[0].members)
    finally:
        app.close()


def test_deepseek_continuation_keeps_reasoning_on_original_member(tmp_path):
    class CapturingBackend(Backend):
        def generate(self, messages, **kwargs):
            self.received = messages
            return SimpleNamespace(
                metadata={}, assistant_message={"role": "assistant", "content": "OK"}
            )

    a, b = CapturingBackend("a"), CapturingBackend("b")
    pool = EndpointPoolBackend("deepseek", {"a": a, "b": b}, tmp_path)
    message = {
        "role": "assistant",
        "content": "",
        "_endpoint_pool_member": "a",
        "_provider_payloads": {"openai": {"reasoning_text": "provider state"}},
    }
    for _ in range(2):
        result = pool.generate([message], role="worker")
        assert result.metadata["endpoint_pool_member"] == "a"
        assert a.received[0]["_provider_payloads"]["openai"]["reasoning_text"] == "provider state"
    assert not hasattr(b, "received")
