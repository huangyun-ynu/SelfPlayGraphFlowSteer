from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import canonical_dataset_name

VALIDATOR_VERSION = "fixed_task_pool_validator_v2"
VERIFIER_CONTRACT_VERSION = "trusted_outcome_verifier_v1"
ADAPTER_CONTRACT_VERSION = "dataset_adapter_v1"
PRIVATE_PAYLOAD_SCHEMA_VERSION = "task_spec_private_payload_v1"


@dataclass(frozen=True)
class PoolAuditResult:
    validated_path: Path
    manifest_path: Path
    accepted: int
    rejected: int


def audit_fixed_task_pool(
    source_paths: Iterable[str | Path],
    *,
    output_dir: str | Path,
) -> PoolAuditResult:
    """Validate immutable task rows and write a new pool plus public manifest."""

    paths = [Path(value).resolve() for value in source_paths]
    if not paths:
        raise ValueError("at least one source task-pool path is required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = path.read_bytes()
        source_hash.update(path.name.encode("utf-8"))
        source_hash.update(b"\0")
        source_hash.update(payload)
        # JSON strings can contain literal U+2028/U+2029; those are not JSONL
        # record delimiters. Split only physical LF lines (CRLF is whitespace).
        for line_number, line in enumerate(payload.decode("utf-8").split("\n"), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: task row must be an object")
            rows.append(dict(row))

    accepted: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    datasets: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        task_id = str(row.get("id", row.get("source_id", index))).strip()
        reasons = _validate_pool_row(row, task_id=task_id, seen_ids=seen_ids)
        if reasons:
            failures.append({"task_id": task_id, "reason": "; ".join(reasons)})
            continue
        seen_ids.add(task_id)
        dataset = canonical_dataset_name(row.get("dataset", "")) or str(row.get("dataset", ""))
        datasets[dataset] += 1
        accepted.append(row)

    validated_path = output / "validated_task_pool.jsonl"
    serialized = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in accepted
    ).encode("utf-8")
    validated_path.write_bytes(serialized)
    manifest_path = output / "validated_task_pool_manifest.json"
    manifest = {
        "schema_version": 1,
        "validator_version": VALIDATOR_VERSION,
        "source_pool_sha256": source_hash.hexdigest(),
        "validated_pool_sha256": hashlib.sha256(serialized).hexdigest(),
        "source_paths": [path.name for path in paths],
        "total_task_count": len(rows),
        "validated_task_count": len(accepted),
        "rejected_task_count": len(failures),
        "dataset_counts": dict(sorted(datasets.items())),
        "verifier_contract_version": VERIFIER_CONTRACT_VERSION,
        "adapter_contract_version": ADAPTER_CONTRACT_VERSION,
        "private_payload_schema_version": PRIVATE_PAYLOAD_SCHEMA_VERSION,
        "failed_tasks": failures,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PoolAuditResult(validated_path, manifest_path, len(accepted), len(failures))


def _validate_pool_row(
    row: dict[str, Any],
    *,
    task_id: str,
    seen_ids: set[str],
) -> list[str]:
    reasons: list[str] = []
    if not task_id:
        reasons.append("missing task id")
    elif task_id in seen_ids:
        reasons.append("duplicate task id")
    if str(row.get("split", "train")).casefold() != "train":
        reasons.append("split is not train")
    prompt = str(
        row.get("prompt", row.get("task", row.get("problem", row.get("question", ""))))
    ).strip()
    if not prompt:
        reasons.append("missing prompt")
    dataset = canonical_dataset_name(row.get("dataset", ""))
    if not dataset:
        reasons.append("missing or unsupported dataset")
    verifier = str(row.get("verifier", (row.get("metadata") or {}).get("verifier", ""))).strip()
    if not verifier:
        reasons.append("missing verifier contract")
    reference = _reference(row)
    if dataset in {"aime", "nq_open", "hotpotqa"} and reference in (None, "", [], {}):
        reasons.append("missing reference")
    # HotpotQA comparison questions can legitimately name the answer entity in
    # the question (for example, asking which of two named people was born
    # first).  Presence alone is not answer leakage for that dataset.
    if (
        dataset != "hotpotqa"
        and isinstance(reference, str)
        and len(reference.strip()) >= 4
        and reference.casefold() in prompt.casefold()
    ):
        reasons.append("reference appears verbatim in prompt")
    if dataset == "healthbench_professional":
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        rubrics = row.get("rubric_items", metadata.get("rubric_items"))
        if not isinstance(rubrics, list) or not rubrics:
            reasons.append("missing HealthBench rubric_items")
        else:
            positive = sum(
                max(0.0, float(item.get("points", 0.0)))
                for item in rubrics
                if isinstance(item, dict)
            )
            if positive <= 0.0:
                reasons.append("HealthBench rubric has no positive points")
    if dataset == "swe_bench":
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else row
        for key in ("repo", "base_commit"):
            if not str(row.get(key, metadata.get(key, ""))).strip():
                reasons.append(f"missing SWE {key}")
    if dataset in {"webshop", "alfworld"}:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if not (row.get("task_id") or metadata.get("task_id") or task_id):
            reasons.append("missing environment task identity")
    return reasons


def _reference(row: dict[str, Any]) -> Any:
    answers = row.get("target_answers")
    if isinstance(answers, list) and answers:
        return answers[0] if len(answers) == 1 else answers
    return row.get("reference", row.get("answer"))
