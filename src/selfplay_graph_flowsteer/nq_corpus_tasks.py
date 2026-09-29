"""Prepare NQ question-only inputs for the fixed local corpus search Action."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .config import canonical_dataset_name


_FORBIDDEN_METADATA = frozenset(
    {
        "context_documents", "evidence_source", "evidence_top_k", "evidence_cache_key",
        "gold_page", "gold_answer", "ground_truth", "has_answer", "all_answers",
        "target_answers", "reference", "source_annotations",
    }
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def validate_corpus_task(row: dict[str, Any]) -> None:
    """Reject frozen evidence and hidden labels before a corpus-mode NQ rollout."""

    metadata = row.get("metadata")
    if not isinstance(metadata, dict) or canonical_dataset_name(metadata.get("dataset")) != "nq_open":
        raise ValueError("corpus_tool requires an NQ-open task with metadata")
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("corpus_tool requires a nonempty original question")
    if metadata.get("evidence_mode") != "corpus_tool":
        raise ValueError("NQ task requires evidence_mode=corpus_tool")
    if metadata.get("original_question") != prompt:
        raise ValueError("NQ corpus prompt must equal its original public question")
    forbidden = _FORBIDDEN_METADATA.intersection(metadata)
    if forbidden:
        raise ValueError("NQ corpus task contains forbidden evidence/labels: " + ", ".join(sorted(forbidden)))


def prepare_pool(
    source: Path,
    output: Path,
    *,
    profile: str = "nq-dense8-v1",
    identity_sha256: str = "",
    expected_nq_count: int | None = None,
) -> dict[str, Any]:
    """Preserve every row and reference while marking only NQ prompts for live search."""

    source = Path(source)
    output = Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("corpus task output must differ from source")
    if not profile.strip():
        raise ValueError("corpus profile cannot be empty")
    if identity_sha256 and (len(identity_sha256) != 64 or any(c not in "0123456789abcdef" for c in identity_sha256)):
        raise ValueError("identity_sha256 must be lowercase SHA-256")
    source_bytes = source.read_bytes()
    prepared: list[dict[str, Any]] = []
    nq_ids: list[str] = []
    seen_nq_ids: set[str] = set()
    question_hashes: list[dict[str, str]] = []
    # JSONL uses LF separators. Unicode line separators inside a JSON string
    # are valid content in the existing mixed training pool.
    for number, line in enumerate(source_bytes.decode("utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid task JSON at line {number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"task line {number} must be an object")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"task line {number} metadata must be an object")
        if canonical_dataset_name(metadata.get("dataset", row.get("dataset"))) == "nq_open":
            question = row.get("prompt")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"NQ task line {number} has no public question")
            if "evidence_mode" in metadata and metadata["evidence_mode"] != "corpus_tool":
                raise ValueError("NQ source already contains frozen or supplied evidence")
            converted = dict(row)
            converted_metadata = dict(metadata)
            converted_metadata["dataset"] = "nq_open"
            converted_metadata["evidence_mode"] = "corpus_tool"
            converted_metadata["original_question"] = question
            converted_metadata["corpus_profile"] = profile
            converted["metadata"] = converted_metadata
            validate_corpus_task(converted)
            identifier = str(converted.get("id", ""))
            if not identifier or identifier in seen_nq_ids:
                raise ValueError(f"duplicate or empty NQ task ID at line {number}")
            seen_nq_ids.add(identifier)
            nq_ids.append(identifier)
            question_hashes.append({"id": identifier, "sha256": _sha256(question.encode("utf-8"))})
            row = converted
        prepared.append(row)
    if not nq_ids:
        raise ValueError("source task pool contains no NQ-open questions")
    if expected_nq_count is not None and len(nq_ids) != expected_nq_count:
        raise ValueError(f"expected {expected_nq_count} NQ tasks; found {len(nq_ids)}")
    output_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in prepared
    ).encode("utf-8")
    manifest = {
        "schema": "nq_corpus_tool_task_pool_v1",
        "source": str(source.resolve()),
        "source_sha256": _sha256(source_bytes),
        "output": str(output.resolve()),
        "output_sha256": _sha256(output_bytes),
        "rows": len(prepared),
        "nq_count": len(nq_ids),
        "nq_ids": nq_ids,
        "question_hashes_sha256": _sha256(json.dumps(question_hashes, sort_keys=True).encode("utf-8")),
        "profile": profile,
        "identity_sha256": identity_sha256 or None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(output_bytes)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest_path = output.with_name(output.name + ".manifest.json")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    try:
        temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary_manifest.replace(manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="nq-dense8-v1")
    parser.add_argument("--identity-sha256", default="")
    parser.add_argument("--expected-nq-count", type=int)
    args = parser.parse_args()
    print(json.dumps(prepare_pool(
        args.input, args.output, profile=args.profile,
        identity_sha256=args.identity_sha256, expected_nq_count=args.expected_nq_count,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
