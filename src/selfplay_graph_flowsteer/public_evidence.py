"""Public retrieval projection, independent of task labels and candidate correctness."""

from __future__ import annotations

import math
from typing import Any


def public_document(document: object) -> dict[str, Any] | None:
    if not isinstance(document, dict):
        return None
    if any(document.get(key) is True for key in ("private", "is_private", "verifier_only")):
        return None
    if str(document.get("visibility", "")).casefold() in {
        "private",
        "hidden",
        "verifier_only",
        "oracle",
        "gold",
    }:
        return None
    if str(document.get("source_type", "")).casefold() in {
        "reference_answer",
        "gold_answer",
        "teacher_solution",
        "oracle",
    }:
        return None
    # Public passages may legitimately contain the correct answer. Never redact
    # based on a reference value or filter/reorder by gold relevance annotations.
    return {
        key: value
        for key in ("id", "title", "text", "contents")
        if isinstance(value := document.get(key), str)
        or (key == "id" and isinstance(value, int) and not isinstance(value, bool))
    }


def public_search_results(results: list) -> list[list[dict[str, Any]]]:
    projected = []
    for group in results:
        if not isinstance(group, list):
            raise RuntimeError("local retrieval service returned an invalid query result")
        hits = []
        for hit in group:
            if not isinstance(hit, dict):
                raise RuntimeError("local retrieval service returned an invalid document hit")
            if public_document(hit) is None:
                continue
            document = public_document(hit.get("document"))
            if document is None:
                continue
            item: dict[str, Any] = {"document": document}
            score = hit.get("score")
            if (
                isinstance(score, (float, int))
                and not isinstance(score, bool)
                and math.isfinite(score)
            ):
                item["score"] = score
            hits.append(item)
        projected.append(hits)
    return projected
