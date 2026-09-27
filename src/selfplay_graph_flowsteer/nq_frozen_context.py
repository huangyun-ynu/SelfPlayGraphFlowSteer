"""Reference-blind, reusable NQ evidence preparation for formal task pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .config import canonical_dataset_name


def _dataset(row: dict[str, Any]) -> str:
    return canonical_dataset_name(row.get("metadata", {}).get("dataset", row.get("dataset")))


def validate_frozen_context(row: dict[str, Any], *, top_k: int) -> None:
    metadata = row.get("metadata", {})
    docs = metadata.get("context_documents", [])
    question = str(metadata.get("original_question", "")).strip()
    prompt = str(row.get("prompt", ""))
    if (
        metadata.get("evidence_mode") != "provided_context_inline"
        or not isinstance(docs, list)
        or len(docs) != top_k
        or not question
        or question not in prompt
        or any(
            not isinstance(doc, dict)
            or not str(doc.get("text", "")).strip()
            or str(doc["text"]) not in prompt
            for doc in docs
        )
    ):
        raise ValueError(
            f"NQ task {row.get('id', '')!r} requires {top_k} frozen inline passages; "
            "prepare the task pool with python -m selfplay_graph_flowsteer.nq_frozen_context"
        )


def fetch_passages(question: str, *, service_url: str, top_k: int) -> list[dict[str, Any]]:
    # Only the public question is sent. References and other row fields stay local.
    request = Request(
        service_url,
        data=json.dumps({"queries": [question], "topk": top_k, "return_scores": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        hits = json.load(response)["result"][0]
    docs = []
    for rank, hit in enumerate(hits[:top_k], 1):
        document = hit.get("document", hit)
        docs.append(
            {
                "id": str(document.get("id", "")),
                "title": str(document.get("title", "")),
                "text": str(document.get("text", document.get("contents", ""))).strip(),
                "rank": rank,
                "score": hit.get("score"),
                "source": "local_searchr1_e5_faiss_cache",
            }
        )
    return docs


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_pool(
    source: Path,
    output: Path,
    *,
    service_url: str,
    top_k: int = 8,
    workers: int = 8,
    fetcher=fetch_passages,
) -> dict[str, Any]:
    if source.resolve() == output.resolve():
        raise ValueError("frozen evidence output must differ from the original task pool")
    if top_k <= 0 or workers <= 0:
        raise ValueError("top_k and workers must be positive")
    source_bytes = source.read_bytes()
    rows = []
    for number, line in enumerate(source_bytes.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"invalid source task pool {source}, line {number}") from exc
    cache = output.parent / (output.name + ".evidence-cache")
    cache.mkdir(parents=True, exist_ok=True)

    def prepare(row: dict[str, Any]) -> dict[str, Any]:
        if _dataset(row) != "nq_open":
            return row
        metadata = dict(row.get("metadata", {}))
        if metadata.get("evidence_mode") == "provided_context_inline":
            validate_frozen_context(row, top_k=top_k)
            return row
        question = str(metadata.get("original_question") or row["prompt"]).strip()
        key_payload = {
            "schema": "formal_nq_frozen_v1",
            "service_url": service_url,
            "top_k": top_k,
            "question": question,
        }
        key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()
        cache_path = cache / f"{key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("request") != key_payload:
                raise ValueError("NQ evidence cache request mismatch")
            docs = cached["documents"]
        else:
            docs = fetcher(question, service_url=service_url, top_k=top_k)
        evidence = "\n\n".join(
            f"[{index}] {doc.get('title', '')}\n{doc.get('text', '')}"
            for index, doc in enumerate(docs, 1)
        )
        metadata.update(
            evidence_mode="provided_context_inline",
            evidence_source="local_searchr1_e5_faiss_cache",
            evidence_top_k=len(docs),
            context_documents=docs,
            original_question=question,
            evidence_cache_key=key,
        )
        prepared = {
            **row,
            "prompt": (
                "Based on the following passages, answer the question.\n\n"
                f"{evidence}\n\nQuestion: {question}\nAnswer:"
            ),
            "metadata": metadata,
        }
        validate_frozen_context(prepared, top_k=top_k)
        if not cache_path.exists():
            # Per-row ID keeps concurrent duplicate public questions from sharing
            # a temporary path; completed caches contain no reference answers.
            temp = cache_path.with_name(f".{key}.{id(row)}.tmp")
            temp.write_text(
                json.dumps({"request": key_payload, "documents": docs}, ensure_ascii=False),
                encoding="utf-8",
            )
            temp.replace(cache_path)
        return prepared

    with ThreadPoolExecutor(max_workers=workers) as pool:
        prepared_rows = list(pool.map(prepare, rows))
    # Publish only a complete pool, in its original ordering. A failed fetch
    # leaves any previous output intact and completed evidence caches reusable.
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prepared_rows)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(output)
    manifest = {
        "schema": "formal_nq_frozen_v1",
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output": str(output.resolve()),
        "output_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "rows": len(rows),
        "datasets": dict(Counter(_dataset(row) for row in rows)),
        "top_k": top_k,
        "service_url": service_url,
        "references_used_for_retrieval": False,
    }
    _atomic_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--service-url", default="http://127.0.0.1:18010/retrieve")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.config:
        import tomllib

        retrieval = tomllib.loads(args.config.read_text(encoding="utf-8")).get("retrieval", {})
        args.service_url = retrieval.get("service_url", args.service_url)
        args.top_k = int(retrieval.get("nq_frozen_top_k", 0))
        if not args.top_k:
            raise ValueError("configuration does not enable frozen NQ evidence")
    print(json.dumps(prepare_pool(
        args.input, args.output, service_url=args.service_url,
        top_k=args.top_k, workers=args.workers,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
