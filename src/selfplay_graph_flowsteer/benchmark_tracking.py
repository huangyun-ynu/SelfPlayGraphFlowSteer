"""Durable local benchmark telemetry with an optional best-effort W&B mirror."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import time
import traceback
import uuid
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .evaluation import EvaluationRecord
from .learning import FixedDatasetExample


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:100] or "unknown"


def _dataset(example: FixedDatasetExample) -> str:
    return str((example.metadata or {}).get("dataset", "unknown"))


def _job_key(example: FixedDatasetExample, seed: int) -> str:
    identity = json.dumps(
        [_dataset(example), example.example_id, int(seed), example.task],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


def _request_metrics(value: Any) -> tuple[Counter[str], Counter[str]]:
    routes: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    if isinstance(value, dict):
        if value.get("event") in {"backend_request_success", "backend_request_failure"}:
            routes[_safe_name(str(value.get("route") or "unknown"))] += 1
            roles[_safe_name(str(value.get("request_role") or "unknown"))] += 1
        for item in value.values():
            child_routes, child_roles = _request_metrics(item)
            routes.update(child_routes)
            roles.update(child_roles)
    elif isinstance(value, list):
        for item in value:
            child_routes, child_roles = _request_metrics(item)
            routes.update(child_routes)
            roles.update(child_roles)
    return routes, roles


class BenchmarkRunTracker:
    """Stores every sample before attempting telemetry transport."""

    def __init__(self, root: str | Path, *, wandb_mode: str = "disabled") -> None:
        self.root = Path(root)
        self.wandb_mode = wandb_mode
        self.run = None
        self.wandb_failed = False
        self.started_at = time.time()
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("samples", "trajectories", "errors", "wandb_outbox"):
            (self.root / name).mkdir(exist_ok=True)
        state_path = self.root / "run_state.json"
        self.state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.is_file()
            else {"run_id": uuid.uuid4().hex[:16], "completed": 0, "failed": 0}
        )
        _write_json(state_path, self.state)

    def start_wandb(self, config: dict[str, Any]) -> None:
        _write_json(self.root / "run_manifest.json", config)
        if self.wandb_mode == "disabled":
            return
        try:
            wandb = importlib.import_module("wandb")
            for name in ("NO_PROXY", "no_proxy"):
                os.environ[name] = ",".join(
                    filter(
                        None,
                        (
                            os.environ.get(name),
                            "api.wandb.ai,wandb.ai,.wandb.ai,storage.googleapis.com,.storage.googleapis.com",
                        ),
                    )
                )
            run_dir = self.root / "wandb"
            run_dir.mkdir(exist_ok=True)
            self.run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "selfplay-graph-flowsteer"),
                entity=os.environ.get("WANDB_ENTITY") or None,
                id=self.state["run_id"],
                resume="allow" if self.wandb_mode == "online" else None,
                name=os.environ.get("WANDB_NAME", self.root.name),
                mode=self.wandb_mode,
                dir=str(run_dir),
                config=config,
                settings=wandb.Settings(
                    console="off",
                    disable_code=True,
                    disable_git=True,
                    x_disable_meta=True,
                    x_save_requirements=False,
                    # Direct W&B occasionally needs one transport retry before
                    # GraphQL setup succeeds; do not abandon live telemetry while
                    # its core process is still establishing the same run.
                    init_timeout=120,
                ),
            )
            self.run.define_metric("sample_step")
            self.run.define_metric("sample/*", step_metric="sample_step")
            self.run.define_metric("summary/*")
            (self.root / "wandb_error.json").unlink(missing_ok=True)
            self.state["wandb_url"] = getattr(self.run, "url", None)
            _write_json(self.root / "run_state.json", self.state)
            self._drain_wandb()
        except Exception as exc:
            self._wandb_error("init", exc)

    def is_complete(self, example: FixedDatasetExample, seed: int) -> bool:
        return (self.root / "samples" / f"{_job_key(example, seed)}.json").is_file()

    def record(self, example: FixedDatasetExample, seed: int, record: EvaluationRecord) -> None:
        key = _job_key(example, seed)
        trajectory_path = self.root / "trajectories" / f"{key}.json"
        sample_path = self.root / "samples" / f"{key}.json"
        _write_json(trajectory_path, record.trajectory)
        payload = {
            "job_key": key,
            "dataset": _dataset(example),
            "example_id": example.example_id,
            "seed": int(seed),
            "trajectory_path": str(trajectory_path.relative_to(self.root)),
            "record": record.to_dict(),
        }
        _write_json(sample_path, payload)
        (self.root / "errors" / f"{key}.json").unlink(missing_ok=True)
        self.state["completed"] = len(list((self.root / "samples").glob("*.json")))
        self.state["failed"] = len(list((self.root / "errors").glob("*.json")))
        self.state["updated_at"] = time.time()
        _write_json(self.root / "run_state.json", self.state)
        self._queue_wandb_sample(payload)

    def record_error(self, example: FixedDatasetExample, seed: int, exc: BaseException) -> None:
        key = _job_key(example, seed)
        payload = {
            "job_key": key,
            "dataset": _dataset(example),
            "example_id": example.example_id,
            "seed": int(seed),
            "exception_type": type(exc).__name__,
            "message": str(exc)[:2000],
            "traceback": "".join(traceback.format_exception(exc))[-20000:],
            "recorded_at": time.time(),
        }
        _write_json(self.root / "errors" / f"{key}.json", payload)
        self.state["failed"] = len(list((self.root / "errors").glob("*.json")))
        self.state["updated_at"] = time.time()
        _write_json(self.root / "run_state.json", self.state)
        row = {
            "sample_step": self.state.get("completed", 0) + self.state["failed"],
            "sample/error": 1,
            "sample/dataset": payload["dataset"],
            "sample/exception_type": payload["exception_type"],
        }
        self._queue_wandb(f"error-{key}", row)

    def records(self) -> list[EvaluationRecord]:
        records = []
        for path in sorted((self.root / "samples").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))["record"]
            records.append(EvaluationRecord(**payload))
        return records

    def dataset_by_task(self) -> dict[tuple[str, int], str]:
        output = {}
        for path in (self.root / "samples").glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            record = payload["record"]
            output[(str(record["task_id"]), int(record["seed"]))] = str(payload["dataset"])
        return output

    def finish(self, summary: dict[str, Any], *, failed: bool = False) -> None:
        _write_json(self.root / "aggregate_summary.json", summary)
        numeric = {"summary/failed_jobs": int(self.state.get("failed", 0))}
        for scope, values in summary.items():
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    numeric[f"summary/{_safe_name(scope)}/{_safe_name(key)}"] = value
        if self.run is not None and not self.wandb_failed:
            try:
                self._drain_wandb()
                self.run.log(numeric)
                wandb = importlib.import_module("wandb")
                artifact = wandb.Artifact(f"sota-eval-{self.state['run_id']}", type="evaluation")
                for name in ("run_manifest.json", "run_state.json", "aggregate_summary.json"):
                    artifact.add_file(str(self.root / name), name=name)
                artifact.add_dir(str(self.root / "samples"), name="samples")
                artifact.add_dir(str(self.root / "trajectories"), name="trajectories")
                artifact.add_dir(str(self.root / "errors"), name="errors")
                self.run.log_artifact(artifact)
            except Exception as exc:
                self._wandb_error("artifact", exc)
        if self.run is not None:
            try:
                self.run.finish(exit_code=int(failed))
            except Exception as exc:
                self._wandb_error("finish", exc)

    def _queue_wandb_sample(self, payload: dict[str, Any]) -> None:
        record = payload["record"]
        routes, roles = _request_metrics(record.get("trajectory", {}))
        row: dict[str, Any] = {
            "sample_step": int(self.state["completed"]) + int(self.state.get("failed", 0)),
            "sample/error": 0,
            "sample/exception_type": "",
            "sample/dataset": payload["dataset"],
            "sample/task_id": record["task_id"],
            "sample/score": record["score"],
            "sample/passed": int(record["passed"]),
            "sample/token_cost": record["token_cost"],
            "sample/duration_s": record["duration_s"],
        }
        row.update({f"sample/routes/{key}/calls": value for key, value in routes.items()})
        row.update({f"sample/roles/{key}/calls": value for key, value in roles.items()})
        self._queue_wandb(f"sample-{payload['job_key']}", row)

    def _queue_wandb(self, name: str, row: dict[str, Any]) -> None:
        path = self.root / "wandb_outbox" / f"{name}.json"
        _write_json(path, row)
        self._drain_wandb()

    def _drain_wandb(self) -> None:
        if self.run is None or self.wandb_failed:
            return
        sent = self.state.setdefault("wandb_sent", {})
        for path in sorted((self.root / "wandb_outbox").glob("*.json")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if sent.get(path.name) == digest:
                continue
            try:
                self.run.log(json.loads(path.read_text(encoding="utf-8")))
                sent[path.name] = digest
                _write_json(self.root / "run_state.json", self.state)
            except Exception as exc:
                self._wandb_error("log", exc)
                return

    def _wandb_error(self, stage: str, exc: BaseException) -> None:
        self.wandb_failed = True
        _write_json(
            self.root / "wandb_error.json",
            {"stage": stage, "exception_type": type(exc).__name__, "local_outbox_preserved": True},
        )
        warnings.warn(
            f"W&B {stage} failed ({type(exc).__name__}); local benchmark data preserved",
            stacklevel=2,
        )


def benchmark_aggregate(
    records: list[EvaluationRecord], dataset_by_task: dict[tuple[str, int], str]
) -> dict[str, Any]:
    grouped: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        grouped[dataset_by_task.get((record.task_id, record.seed), "unknown")].append(record)

    def summarize(items: list[EvaluationRecord]) -> dict[str, Any]:
        if not items:
            return {"examples": 0}
        return {
            "examples": len(items),
            "mean_score": sum(item.score for item in items) / len(items),
            "pass_rate": sum(item.passed for item in items) / len(items),
            "mean_token_cost": sum(item.token_cost for item in items) / len(items),
            "mean_duration_s": sum(item.duration_s for item in items) / len(items),
        }

    return {"overall": summarize(records), **{name: summarize(rows) for name, rows in grouped.items()}}
