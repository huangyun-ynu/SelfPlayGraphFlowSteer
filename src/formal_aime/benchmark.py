from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .application import AdaptiveSolverApplication
from .evaluation import EvaluationRecord, from_adaptive_result, from_flowsteer_trajectory
from .learning import FixedDatasetExample


@dataclass(frozen=True)
class BenchmarkSummary:
    system: str
    examples: int
    mean_score: float | None
    pass_rate: float | None
    score_std: float | None
    mean_token_cost: float
    mean_duration_s: float
    unclipped_mean_score: float | None = 0.0
    known_examples: int = 0
    unknown_examples: int = 0
    submitted_examples: int = 0
    policy_failure_examples: int = 0
    known_coverage: float = 0.0
    submission_rate: float = 0.0
    successful_submission_rate_all: float = 0.0
    submitted_answer_pass_rate: float | None = None
    answer_em: float | None = None
    answer_f1: float | None = None
    answer_metric_examples: int = 0


@dataclass(frozen=True)
class PairedSummary:
    pairs: int
    mean_score_delta: float | None
    wins: int
    ties: int
    losses: int
    matched_pairs: int = 0
    candidate_unknown: int = 0
    baseline_unknown: int = 0


class BenchmarkRunner:
    def __init__(
        self,
        application_factory: Callable[[int], AdaptiveSolverApplication],
        *,
        checkpoint: str = "",
    ) -> None:
        self.application_factory = application_factory
        self.checkpoint = checkpoint

    def run(
        self,
        examples: Iterable[FixedDatasetExample],
        *,
        seeds: Iterable[int] = (0,),
        workers: int = 1,
        should_run: Callable[[FixedDatasetExample, int], bool] | None = None,
        on_record: Callable[[FixedDatasetExample, int, EvaluationRecord], None] | None = None,
        on_error: Callable[[FixedDatasetExample, int, BaseException], None] | None = None,
        continue_on_error: bool = False,
        systemic_error_limit: int = 3,
    ) -> list[EvaluationRecord]:
        jobs = [
            (example, int(seed))
            for example in examples
            for seed in seeds
            if should_run is None or should_run(example, int(seed))
        ]

        def execute(job: tuple[FixedDatasetExample, int]) -> EvaluationRecord:
            example, seed = job
            application = self.application_factory(seed)
            try:
                started = time.monotonic()
                result = application.solve(
                    example.task,
                    task_id=example.example_id,
                    task_type=str((example.metadata or {}).get("task_type", "general")),
                    reference=example.reference,
                    run_id=f"benchmark-{example.example_id}-{seed}",
                    metadata=example.metadata,
                )
                duration = time.monotonic() - started
                return replace(
                    from_adaptive_result(result, seed=seed, checkpoint=self.checkpoint),
                    duration_s=duration,
                )
            finally:
                application.close()

        if workers <= 0:
            raise ValueError("benchmark workers must be positive")
        if systemic_error_limit <= 0:
            raise ValueError("systemic error limit must be positive")
        if workers == 1:
            records = []
            for example, seed in jobs:
                try:
                    record = execute((example, seed))
                except BaseException as exc:
                    if on_error is not None:
                        on_error(example, seed, exc)
                    if not continue_on_error:
                        raise
                else:
                    records.append(record)
                    if on_record is not None:
                        on_record(example, seed, record)
            return records
        records_by_index: dict[int, EvaluationRecord] = {}
        indexed_jobs = iter(enumerate(jobs))
        consecutive_signature: tuple[type[BaseException], str] | None = None
        consecutive_errors = 0
        terminal_error: BaseException | None = None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: dict[Future[EvaluationRecord], tuple[int, FixedDatasetExample, int]] = {}

            def submit_next() -> bool:
                try:
                    index, (example, seed) = next(indexed_jobs)
                except StopIteration:
                    return False
                futures[pool.submit(execute, (example, seed))] = (index, example, seed)
                return True

            for _ in range(min(workers, len(jobs))):
                submit_next()
            while futures:
                completed, _pending = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    index, example, seed = futures.pop(future)
                    if future.cancelled():
                        continue
                    try:
                        record = future.result()
                    except BaseException as exc:
                        if on_error is not None:
                            on_error(example, seed, exc)
                        if terminal_error is None:
                            if not continue_on_error:
                                terminal_error = exc
                            else:
                                signature = (type(exc), str(exc))
                                if signature == consecutive_signature:
                                    consecutive_errors += 1
                                else:
                                    consecutive_signature = signature
                                    consecutive_errors = 1
                                if consecutive_errors >= systemic_error_limit:
                                    terminal_error = RuntimeError(
                                        f"benchmark stopped after {consecutive_errors} consecutive "
                                        f"{type(exc).__name__} failures: {str(exc)[:500]}"
                                    )
                                    terminal_error.__cause__ = exc
                        if terminal_error is not None:
                            # Stop admission, but save every outcome from work
                            # that is already running before raising the error.
                            for pending in futures:
                                pending.cancel()
                    else:
                        consecutive_signature = None
                        consecutive_errors = 0
                        records_by_index[index] = record
                        if on_record is not None:
                            on_record(example, seed, record)
                    if terminal_error is None:
                        submit_next()
        if terminal_error is not None:
            raise terminal_error
        return [records_by_index[index] for index in sorted(records_by_index)]


def summarize(records: Iterable[EvaluationRecord]) -> BenchmarkSummary:
    items = list(records)
    if not items:
        raise ValueError("cannot summarize an empty benchmark")
    known = [item for item in items if item.score is not None and math.isfinite(item.score)]
    submitted = [item for item in items if item.submission_status == "submitted"]
    scored_submitted = [item for item in submitted if item.score is not None]
    scores = [item.score for item in known]
    raw_mean = sum(scores) / len(scores) if scores else None
    mean = max(0.0, min(1.0, raw_mean)) if raw_mean is not None else None
    variance = sum((score - raw_mean) ** 2 for score in scores) / len(scores) if scores else None
    em = [item.answer_metrics.get("answer_em", item.answer_metrics.get("em")) for item in items]
    f1 = [item.answer_metrics.get("answer_f1", item.answer_metrics.get("f1")) for item in items]
    em = [float(value) for value in em if isinstance(value, (int, float)) and math.isfinite(value)]
    f1 = [float(value) for value in f1 if isinstance(value, (int, float)) and math.isfinite(value)]
    return BenchmarkSummary(
        system=items[0].system,
        examples=len(items),
        mean_score=mean,
        pass_rate=sum(item.passed is True for item in known) / len(known) if known else None,
        score_std=math.sqrt(variance) if variance is not None else None,
        mean_token_cost=sum(item.token_cost for item in items) / len(items),
        mean_duration_s=sum(item.duration_s for item in items) / len(items),
        unclipped_mean_score=raw_mean,
        known_examples=len(known), unknown_examples=len(items) - len(known),
        submitted_examples=len(submitted),
        policy_failure_examples=sum(item.outcome_status == "policy_failure" for item in items),
        known_coverage=len(known) / len(items), submission_rate=len(submitted) / len(items),
        successful_submission_rate_all=sum(item.passed is True for item in submitted) / len(items),
        submitted_answer_pass_rate=(sum(item.passed is True for item in scored_submitted)
                                    / len(scored_submitted) if scored_submitted else None),
        answer_em=sum(em) / len(em) if em else None,
        answer_f1=sum(f1) / len(f1) if f1 else None,
        answer_metric_examples=len(em),
    )


def paired_summary(
    candidate: Iterable[EvaluationRecord], baseline: Iterable[EvaluationRecord]
) -> PairedSummary:
    left = {(item.task_id, item.seed): item for item in candidate}
    right = {(item.task_id, item.seed): item for item in baseline}
    keys = sorted(left.keys() & right.keys())
    if not keys:
        raise ValueError("candidate and baseline have no matching task_id/seed pairs")
    valid_keys = [key for key in keys if left[key].score is not None and right[key].score is not None]
    deltas = [left[key].score - right[key].score for key in valid_keys]
    return PairedSummary(
        pairs=len(valid_keys), matched_pairs=len(keys),
        candidate_unknown=sum(left[key].score is None for key in keys),
        baseline_unknown=sum(right[key].score is None for key in keys),
        mean_score_delta=sum(deltas) / len(deltas) if deltas else None,
        wins=sum(delta > 0 for delta in deltas),
        ties=sum(delta == 0 for delta in deltas),
        losses=sum(delta < 0 for delta in deltas),
    )


def load_flowsteer_records(path: str | Path) -> list[EvaluationRecord]:
    with Path(path).open(encoding="utf-8") as handle:
        return [from_flowsteer_trajectory(json.loads(line)) for line in handle if line.strip()]


def write_benchmark(
    directory: str | Path,
    records: list[EvaluationRecord],
    *,
    baseline: list[EvaluationRecord] | None = None,
) -> dict[str, Any]:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    payload: dict[str, Any] = {"candidate": asdict(summarize(records))}
    if baseline:
        payload["baseline"] = asdict(summarize(baseline))
        payload["paired"] = asdict(paired_summary(records, baseline))
    path = destination / "summary.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return payload
