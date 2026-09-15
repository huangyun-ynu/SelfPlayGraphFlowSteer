from __future__ import annotations

import threading
import time

import pytest

from selfplay_graph_flowsteer import benchmark as benchmark_module
from selfplay_graph_flowsteer.benchmark import BenchmarkRunner
from selfplay_graph_flowsteer.benchmark_tracking import BenchmarkRunTracker, benchmark_aggregate
from selfplay_graph_flowsteer.evaluation import EvaluationRecord
from selfplay_graph_flowsteer.learning import FixedDatasetExample
from selfplay_graph_flowsteer.static_director_skills import StaticDirectorSkillBank


def _example(name: str, dataset: str = "aime") -> FixedDatasetExample:
    return FixedDatasetExample(name, f"task {name}", metadata={"dataset": dataset})


def test_benchmark_runner_rolls_slots_and_closes_each_application(monkeypatch):
    active = 0
    peak = 0
    lock = threading.Lock()
    closed = []

    class Application:
        def solve(self, task, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return kwargs["task_id"]

        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        benchmark_module,
        "from_adaptive_result",
        lambda result, **kwargs: EvaluationRecord(
            task_id=result, system="test", answer="ok", score=1.0, passed=True
        ),
    )
    completed = []
    records = BenchmarkRunner(lambda seed: Application()).run(
        [_example(str(index)) for index in range(7)],
        workers=3,
        on_record=lambda example, seed, record: completed.append(example.example_id),
    )
    assert len(records) == len(closed) == len(completed) == 7
    assert peak == 3


def test_benchmark_runner_persists_errors_and_continues(monkeypatch):
    class Application:
        def solve(self, task, **kwargs):
            if kwargs["task_id"] == "bad":
                raise RuntimeError("failed")
            return kwargs["task_id"]

        def close(self):
            pass

    monkeypatch.setattr(
        benchmark_module,
        "from_adaptive_result",
        lambda result, **kwargs: EvaluationRecord(
            task_id=result, system="test", answer="ok", score=1.0, passed=True
        ),
    )
    errors = []
    records = BenchmarkRunner(lambda seed: Application()).run(
        [_example("bad"), _example("good")],
        workers=2,
        on_error=lambda example, seed, exc: errors.append((example.example_id, type(exc))),
        continue_on_error=True,
    )
    assert [record.task_id for record in records] == ["good"]
    assert errors == [("bad", RuntimeError)]


def test_benchmark_runner_stops_refilling_after_systemic_failures():
    attempts = []

    def broken_factory(seed):
        attempts.append(seed)
        raise ValueError("invalid shared configuration")

    with pytest.raises(RuntimeError, match="3 consecutive ValueError"):
        BenchmarkRunner(broken_factory).run(
            [_example(str(index)) for index in range(100)],
            workers=4,
            continue_on_error=True,
            systemic_error_limit=3,
        )
    assert len(attempts) <= 6


def test_benchmark_tracker_is_resumable_and_keeps_full_trajectory(tmp_path):
    tracker = BenchmarkRunTracker(tmp_path, wandb_mode="disabled")
    tracker.start_wandb({"evaluation_only": True})
    example = _example("one", "hotpotqa")
    record = EvaluationRecord(
        task_id="one",
        system="test",
        answer="answer",
        score=0.5,
        passed=True,
        token_cost=12,
        duration_s=3.0,
        trajectory={"events": [{"event": "backend_request_success", "route": "gpt"}]},
    )
    tracker.record(example, 0, record)
    assert tracker.is_complete(example, 0)
    assert tracker.records() == [record]
    assert list((tmp_path / "trajectories").glob("*.json"))
    summary = benchmark_aggregate(tracker.records(), tracker.dataset_by_task())
    assert summary["hotpotqa"]["mean_score"] == 0.5


def test_static_director_skills_are_dataset_scoped_and_budgeted(tmp_path):
    (tmp_path / "aime.md").write_text("orchestrate exact math", encoding="utf-8")
    bank = StaticDirectorSkillBank(tmp_path, prompt_token_budget=10)

    class Tokenizer:
        def encode(self, text, *, add_special_tokens=False):
            return text.split()

    selected, context, manifest = bank.select_context(
        "private prompt",
        task_type="math",
        tokenizer=Tokenizer(),
        task_metadata={"dataset": "aime"},
    )
    assert selected[0].skill_id == "static_director_aime"
    assert context == "orchestrate exact math"
    assert manifest["dataset"] == "aime"
    with pytest.raises(ValueError, match="above budget"):
        StaticDirectorSkillBank(tmp_path, prompt_token_budget=2).select_context(
            "private prompt",
            task_type="math",
            tokenizer=Tokenizer(),
            task_metadata={"dataset": "aime"},
        )
