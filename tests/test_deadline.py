from __future__ import annotations

import threading
import time

import pytest

from selfplay_graph_flowsteer.deadline import (
    RolloutDeadline,
    WorkerWallClockLimitExceeded,
)


def test_rollout_deadline_honors_collection_cancellation() -> None:
    cancelled = threading.Event()
    deadline = RolloutDeadline(
        total_timeout_s=60.0,
        no_progress_timeout_s=60.0,
        request_timeout_s=60.0,
        cancellation_event=cancelled,
    )

    cancelled.set()

    with pytest.raises(WorkerWallClockLimitExceeded) as raised:
        deadline.check("after_worker_request")
    assert raised.value.reason == "backend_circuit_open"


def test_no_progress_deadline_reports_reason_and_stage() -> None:
    deadline = RolloutDeadline(
        total_timeout_s=100.0,
        no_progress_timeout_s=1.0,
        request_timeout_s=1.0,
        started_monotonic=time.monotonic() - 2.0,
    )

    with pytest.raises(WorkerWallClockLimitExceeded) as caught:
        deadline.check("director_turn_start")

    assert caught.value.reason == "no_progress"
    assert caught.value.stage == "director_turn_start"


def test_effective_progress_refreshes_only_the_idle_budget() -> None:
    deadline = RolloutDeadline(
        total_timeout_s=100.0,
        no_progress_timeout_s=1.0,
        request_timeout_s=0.5,
        started_monotonic=time.monotonic() - 2.0,
    )
    deadline.mark_progress("canvas_set_prompt")

    assert deadline.request_budget_s("backend_request") <= 0.5
    assert deadline.diagnostics()["last_progress_kind"] == "canvas_set_prompt"


def test_active_request_pauses_only_the_no_progress_clock(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr("selfplay_graph_flowsteer.deadline.time.monotonic", lambda: clock[0])
    deadline = RolloutDeadline(
        total_timeout_s=100.0,
        no_progress_timeout_s=1.0,
        request_timeout_s=10.0,
        started_monotonic=0.0,
    )

    clock[0] = 0.75
    with deadline.pause_no_progress():
        clock[0] = 5.0
        deadline.check("backend_request")
        assert deadline.diagnostics()["no_progress_paused"] is True

    assert deadline.diagnostics()["idle_s"] == pytest.approx(0.75)
    assert deadline.diagnostics()["elapsed_s"] == pytest.approx(5.0)
    clock[0] = 5.5
    with pytest.raises(WorkerWallClockLimitExceeded) as caught:
        deadline.check("director_turn_start")
    assert caught.value.reason == "no_progress"


def test_request_budget_supports_narrow_route_override() -> None:
    deadline = RolloutDeadline(
        total_timeout_s=900.0,
        no_progress_timeout_s=120.0,
        request_timeout_s=240.0,
        request_timeout_overrides_s={"grok": 300.0},
    )

    assert deadline.request_budget_s("request", route="gpt") <= 240.0
    assert 299.0 < deadline.request_budget_s("request", route="grok") <= 300.0


def test_director_request_view_does_not_change_worker_or_judge_limits(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("selfplay_graph_flowsteer.deadline.time.monotonic", lambda: clock[0])
    deadline = RolloutDeadline(2400, 300, 300, {"grok": 240}, started_monotonic=0)
    director = deadline.for_request("graph-director", 600)
    assert director.request_budget_s("request") == 600
    assert deadline.for_request("worker", 600) is deadline
    assert deadline.for_request("healthbench-grader", 600) is deadline
    assert deadline.request_budget_s("request") == 300
    assert deadline.request_budget_s("request", route="grok") == 240
    with director.pause_no_progress():
        clock[0] = 599
        director.check("active_generation")
    deadline.check("director_response")
    assert deadline.diagnostics()["idle_s"] == 0
    assert deadline.diagnostics()["elapsed_s"] == 599


def test_director_request_view_keeps_hard_deadline_and_cancellation(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("selfplay_graph_flowsteer.deadline.time.monotonic", lambda: clock[0])
    cancelled = threading.Event()
    deadline = RolloutDeadline(100, 300, 300, started_monotonic=0, cancellation_event=cancelled)
    director = deadline.for_request("graph-director", 600)
    assert director.request_budget_s("request") == 100
    with director.pause_no_progress():
        clock[0] = 101
        with pytest.raises(
            WorkerWallClockLimitExceeded, match="exceeded its effective execution budget"
        ):
            director.check("active_generation")
    clock[0] = 1
    cancelled.set()
    with pytest.raises(WorkerWallClockLimitExceeded) as exc:
        director.check("request")
    assert exc.value.reason == "backend_circuit_open"


def test_failed_wait_credit_is_union_and_preserves_wall_clock(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    deadline = RolloutDeadline(100, 100, 20, started_monotonic=0, exclude_failed_request_time=True)
    clock[0] = 70
    deadline.record_failed_request(10, 40)
    deadline.record_failed_request(20, 50)
    deadline.record_failed_request(10, 40)  # nested gateway/pool accounting
    assert deadline.hard_remaining_s("test") == 70
    diagnostics = deadline.diagnostics()
    assert diagnostics["elapsed_s"] == 70
    assert diagnostics["failed_request_wait_s"] == 40
    assert diagnostics["effective_elapsed_s"] == 30
    clock[0] = 900
    with pytest.raises(WorkerWallClockLimitExceeded) as error:
        deadline.check("test")
    assert error.value.reason == "absolute_wall_deadline"


def test_gateway_refunds_only_failed_requests(monkeypatch):
    from selfplay_graph_flowsteer.llm import _failed_request_clock

    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    deadline = RolloutDeadline(100, 100, 20, started_monotonic=0, exclude_failed_request_time=True)
    with _failed_request_clock(deadline):
        clock[0] = 10
    with pytest.raises(TimeoutError), _failed_request_clock(deadline):
        clock[0] = 30
        raise TimeoutError("upstream timeout")
    with pytest.raises(ValueError), _failed_request_clock(deadline):
        clock[0] = 35
        raise ValueError("local code failure")
    assert deadline.diagnostics()["failed_request_wait_s"] == 20
    assert deadline.hard_remaining_s("test") == 85


def test_failed_wait_cannot_extend_absolute_900_seconds(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    deadline = RolloutDeadline(900, 900, 120, started_monotonic=0, exclude_failed_request_time=True)
    clock[0] = 899
    deadline.record_failed_request(0, 120)
    assert deadline.diagnostics()["effective_elapsed_s"] == 779
    assert deadline.hard_remaining_s("test") == 1
    clock[0] = 900
    with pytest.raises(WorkerWallClockLimitExceeded) as error:
        deadline.check("test")
    assert error.value.reason == "absolute_wall_deadline"


def test_channel_timeout_caps_larger_task_request_budget():
    from types import SimpleNamespace

    from selfplay_graph_flowsteer.llm import _logical_request_budget_s, _request_slot

    config = SimpleNamespace(
        timeout_s=60,
        route_name="gpt_nexus_pro",
        base_url="http://test-channel-cap",
        api_key="test",
        max_concurrency=1,
        max_concurrency_by_dataset={},
        api_keys_by_dataset={},
    )
    deadline = RolloutDeadline(900, 300, 120, request_timeout_overrides_s={"gpt_nexus_pro": 240})
    assert _logical_request_budget_s(config, deadline) == 60
    with _request_slot(config, deadline) as slot:
        assert slot.request_budget_s == 60
        assert 0 < slot.timeout_s <= 60
