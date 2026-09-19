#!/usr/bin/env python3
"""Create an offline NQ evidence cache from the local retrieval service.

The resulting JSONL keeps the canonical ``nq_open`` dataset name, but marks
``evidence_mode=frozen_retrieval`` so the runtime exposes no search Action.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--service-url", default="http://127.0.0.1:18010/retrieve")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    with args.input.open(encoding="utf-8") as handle:
        source_rows = [json.loads(line) for line in handle if line.strip()]

    def prepare(row: dict) -> dict:
        query = str(row["prompt"])
        payload = json.dumps(
            {"queries": [query], "topk": args.top_k, "return_scores": True}
        ).encode()
        req = Request(args.service_url, data=payload, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=180) as response:
            result = json.load(response).get("result", [[]])[0]
        docs = []
        for rank, hit in enumerate(result[: args.top_k], 1):
            document = hit.get("document", hit) if isinstance(hit, dict) else {}
            if not isinstance(document, dict):
                continue
            text = str(document.get("text", document.get("contents", ""))).strip()
            if not text:
                continue
            docs.append(
                {
                    "id": str(document.get("id", "")),
                    "title": str(document.get("title", "")),
                    "text": text,
                    "rank": rank,
                    "score": hit.get("score") if isinstance(hit, dict) else None,
                    "source": "local_searchr1_e5_faiss_cache",
                }
            )
        if not docs:
            raise RuntimeError(f"no evidence returned for {row.get('id', query)!r}")
        row = dict(row)
        metadata = dict(row.get("metadata") or {})
        metadata.update(
            {
                "evidence_mode": "frozen_retrieval",
                "evidence_source": "local_searchr1_e5_faiss_cache",
                "evidence_top_k": len(docs),
                "context_documents": docs,
                "original_question": query,
            }
        )
        evidence = "\n\n".join(
            f"[{index}] {doc['title']}\n{doc['text']}" for index, doc in enumerate(docs, 1)
        )
        row["prompt"] = (
            "Based on the following passages, answer the question.\n\n"
            f"{evidence}\n\nQuestion: {query}\nAnswer:"
        )
        metadata["evidence_mode"] = "provided_context_inline"
        row["metadata"] = metadata
        return row

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool, args.output.open("w", encoding="utf-8") as handle:
        futures = [pool.submit(prepare, row) for row in source_rows]
        for future in as_completed(futures):
            row = future.result()
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            completed += 1
            if completed % 4 == 0 or completed == len(source_rows):
                print(json.dumps({"completed": completed, "total": len(source_rows), "output": str(args.output)}), flush=True)
    print(json.dumps({"rows": completed, "output": str(args.output), "top_k": args.top_k}), flush=True)


if __name__ == "__main__":
    main()
