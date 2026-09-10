from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Iterable
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
    mean_score: float
    pass_rate: float
    score_std: float
    mean_token_cost: float
    mean_duration_s: float
    unclipped_mean_score: float = 0.0


@dataclass(frozen=True)
class PairedSummary:
    pairs: int
    mean_score_delta: float
    wins: int
    ties: int
    losses: int


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
    ) -> list[EvaluationRecord]:
        records: list[EvaluationRecord] = []
        for example in examples:
            for seed in seeds:
                application = self.application_factory(int(seed))
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
                records.append(
                    replace(
                        from_adaptive_result(result, seed=int(seed), checkpoint=self.checkpoint),
                        duration_s=duration,
                    )
                )
        return records


def summarize(records: Iterable[EvaluationRecord]) -> BenchmarkSummary:
    items = list(records)
    if not items:
        raise ValueError("cannot summarize an empty benchmark")
    scores = [item.score for item in items]
    raw_mean = sum(scores) / len(scores)
    mean = max(0.0, min(1.0, raw_mean))
    variance = sum((score - raw_mean) ** 2 for score in scores) / len(scores)
    return BenchmarkSummary(
        system=items[0].system,
        examples=len(items),
        mean_score=mean,
        pass_rate=sum(item.passed for item in items) / len(items),
        score_std=math.sqrt(variance),
        mean_token_cost=sum(item.token_cost for item in items) / len(items),
        mean_duration_s=sum(item.duration_s for item in items) / len(items),
        unclipped_mean_score=raw_mean,
    )


def paired_summary(
    candidate: Iterable[EvaluationRecord], baseline: Iterable[EvaluationRecord]
) -> PairedSummary:
    left = {(item.task_id, item.seed): item for item in candidate}
    right = {(item.task_id, item.seed): item for item in baseline}
    keys = sorted(left.keys() & right.keys())
    if not keys:
        raise ValueError("candidate and baseline have no matching task_id/seed pairs")
    deltas = [left[key].score - right[key].score for key in keys]
    return PairedSummary(
        pairs=len(keys),
        mean_score_delta=sum(deltas) / len(deltas),
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
