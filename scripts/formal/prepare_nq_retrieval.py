#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from selfplay_graph_flowsteer.retrieval_service import create_index

_RESULT = re.compile(
    r'<div class="mw-search-result-heading"><a href="([^"]+)" title="([^"]+)"[^>]*>.*?'
    r'</a>\s*</div><div class="searchresult">(.*?)</div>',
    re.DOTALL,
)
_PARAGRAPH = re.compile(r"<p(?:\s[^>]*)?>(.*?)</p>", re.DOTALL)
_REMOVE = re.compile(
    r"<(?:script|style|table|figure|sup)\b.*?</(?:script|style|table|figure|sup)>",
    re.DOTALL | re.IGNORECASE,
)
_TAG = re.compile(r"<[^>]+>")


def clean_markup(value: str) -> str:
    value = _REMOVE.sub(" ", value)
    value = _TAG.sub(" ", value)
    return " ".join(html.unescape(value).split())


def fetch(url: str, *, attempts: int = 3, timeout_s: float = 30.0) -> str:
    request = Request(url, headers={"User-Agent": "SelfPlayGraphFlowSteer/0.1 retrieval-index"})
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=timeout_s) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def search_documents(question: str, limit: int) -> list[dict[str, str]]:
    query = urlencode({"search": question, "title": "Special:Search", "ns0": 1, "limit": limit})
    body = fetch(f"https://en.wikipedia.org/w/index.php?{query}")
    documents = []
    for href, title, snippet in _RESULT.findall(body)[:limit]:
        title = html.unescape(title)
        documents.append(
            {
                "id": hashlib.sha256(href.encode()).hexdigest()[:24],
                "title": title,
                "text": clean_markup(snippet),
                "url": f"https://en.wikipedia.org{href}",
            }
        )
    return documents


def enrich_document(document: dict[str, str]) -> dict[str, str]:
    try:
        body = fetch(document["url"], attempts=1, timeout_s=10.0)
        paragraphs = [clean_markup(item) for item in _PARAGRAPH.findall(body)]
        article = "\n".join(item for item in paragraphs if len(item) >= 40)[:24_000]
        if article:
            document = {**document, "text": article}
    except Exception:
        pass
    return document


def read_questions(path: Path) -> list[str]:
    questions = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            question = str(row.get("prompt", "")).strip()
            if question:
                questions.append(question)
    if not questions:
        raise ValueError("dataset contains no public questions")
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local Wikipedia index for NQ-open")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-results", type=int, default=10)
    parser.add_argument("--pages-per-query", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--minimum-query-coverage", type=float, default=0.9)
    args = parser.parse_args()
    if not 0 < args.minimum_query_coverage <= 1:
        parser.error("--minimum-query-coverage must be in (0, 1]")

    found: dict[str, dict[str, str]] = {}
    selected_urls: set[str] = set()
    questions = read_questions(args.dataset)
    successful_queries = 0
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(search_documents, question, args.search_results): question
            for question in questions
        }
        for future in as_completed(futures):
            question = futures[future]
            try:
                documents = future.result()
            except Exception as exc:
                failures.append(f"{question}: {type(exc).__name__}: {exc}")
                continue
            if not documents:
                failures.append(f"{question}: no search results")
                continue
            successful_queries += 1
            selected_urls.update(
                document["url"] for document in documents[: args.pages_per_query]
            )
            for document in documents:
                found.setdefault(document["url"], document)
            if (successful_queries + len(failures)) % 16 == 0:
                print(
                    f"searched {successful_queries + len(failures)}/{len(questions)} questions",
                    file=sys.stderr,
                    flush=True,
                )

    coverage = successful_queries / len(questions)
    if coverage < args.minimum_query_coverage:
        examples = "\n".join(failures[:5])
        raise RuntimeError(
            f"Wikipedia search coverage {coverage:.1%} is below "
            f"{args.minimum_query_coverage:.1%}; examples:\n{examples}"
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(enrich_document, found[url]): url
            for url in selected_urls
            if url in found
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            found[futures[future]] = future.result()
            if completed % 32 == 0 or completed == len(futures):
                print(
                    f"enriched {completed}/{len(futures)} Wikipedia pages",
                    file=sys.stderr,
                    flush=True,
                )

    create_index(args.output, list(found.values()))
    print(
        json.dumps(
            {
                "questions": len(questions),
                "successful_queries": successful_queries,
                "failed_queries": len(failures),
                "documents": len(found),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
