from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def classify_judge_exception(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    # Typed backend failures keep their normalized HTTP status and retryability
    # inside ``classification`` so request wrappers do not have to expose
    # provider-specific exception attributes.  Consult that envelope before
    # falling back to the raw exception chain/string.
    classification = getattr(exc, "classification", None)
    if status is None and classification is not None:
        status = getattr(classification, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        status_code = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_code = None
    if status_code == 429:
        return "rate_limit"
    if status_code is not None and 400 <= status_code < 500:
        return "http_4xx"
    if status_code is not None and 500 <= status_code < 600:
        return "http_5xx"
    rendered = f"{type(exc).__name__} {exc}".casefold()
    if any(
        marker in rendered
        for marker in (
            "timeout",
            "timed out",
            "wallclocklimitexceeded",
            "wall-clock budget",
            "no effective progress",
        )
    ):
        return "timeout"
    return "unknown_exception"


def retryable_judge_error(category: str) -> bool:
    return category in {
        "rate_limit",
        "timeout",
        "http_5xx",
        "empty_response",
        "invalid_json",
        "invalid_schema",
    }


@dataclass
class HealthBenchJudgeAuditStore:
    """Private append-only Judge evidence plus an aggregate public-safe manifest."""

    root: Path
    _lock = threading.Lock()

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if "private" not in {part.casefold() for part in self.root.parts}:
            raise ValueError("HealthBench Judge audit path must be under a private directory")

    @property
    def events_path(self) -> Path:
        return self.root / "judge_events.private.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def append(self, event: dict[str, Any]) -> None:
        record = {
            "schema_version": 1,
            "recorded_at": datetime.now(UTC).isoformat(),
            **event,
        }
        with self._lock:
            self._prepare_directory()
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            os.chmod(self.events_path, 0o600)
            manifest = self._load_manifest()
            self._update_manifest(manifest, record)
            self._write_manifest(manifest)

    def load_criterion(self, key: str) -> dict[str, Any] | None:
        path = self.root / "criteria" / (key + ".json")
        with self._lock:
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))

    def save_criterion(self, key: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._prepare_directory()
            directory = self.root / "criteria"
            directory.mkdir(mode=0o700, exist_ok=True)
            path = directory / (key + ".json")
            temporary = directory / f"{key}.{os.getpid()}.{threading.get_ident()}.tmp"
            temporary.write_text(
                json.dumps(result, default=lambda value: sorted(value)), encoding="utf-8"
            )
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)

    def _prepare_directory(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def _load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {
            "schema_version": 1,
            "privacy": "private_judge_evidence",
            "events_file": self.events_path.name,
            "total_events": 0,
            "events_by_type": {},
            "errors_by_category": {},
            "judge_calls": 0,
            "token_in": 0,
            "token_out": 0,
            "latency_s": 0.0,
            "priced_calls": 0,
            "unpriced_calls": 0,
            "priced_estimated_cost_usd": 0.0,
            "estimated_cost_usd": 0.0,
            "cache_hits": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
        }

    @staticmethod
    def _update_manifest(manifest: dict[str, Any], record: dict[str, Any]) -> None:
        manifest.setdefault(
            "priced_estimated_cost_usd",
            float(manifest.get("estimated_cost_usd") or 0.0)
            if not manifest.get("unpriced_calls")
            else 0.0,
        )
        event_type = str(record.get("event", "unknown"))
        manifest["total_events"] += 1
        event_counts = manifest["events_by_type"]
        event_counts[event_type] = int(event_counts.get(event_type, 0)) + 1
        if event_type == "judge_attempt":
            manifest["judge_calls"] += 1
            manifest["token_in"] += int(record.get("token_in", 0) or 0)
            manifest["token_out"] += int(record.get("token_out", 0) or 0)
            manifest["latency_s"] += float(record.get("latency_s", 0.0) or 0.0)
            cost = record.get("estimated_cost_usd")
            if cost is None:
                manifest["unpriced_calls"] += 1
            else:
                manifest["priced_calls"] += 1
                manifest["priced_estimated_cost_usd"] = float(
                    manifest.get("priced_estimated_cost_usd", 0.0)
                ) + float(cost)
            category = str(record.get("error_category", "")).strip()
            if category:
                errors = manifest["errors_by_category"]
                errors[category] = int(errors.get(category, 0)) + 1
        elif event_type == "cache_hit":
            manifest["cache_hits"] += 1
        elif event_type == "task_complete":
            manifest["tasks_completed"] += 1
        elif event_type == "task_failed":
            manifest["tasks_failed"] += 1
        manifest["estimated_cost_usd"] = (
            None
            if manifest["unpriced_calls"]
            else float(manifest.get("priced_estimated_cost_usd", 0.0))
        )
        manifest["updated_at"] = record["recorded_at"]

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        temporary = self.root / ".manifest.tmp"
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.manifest_path)
        os.chmod(self.manifest_path, 0o600)
