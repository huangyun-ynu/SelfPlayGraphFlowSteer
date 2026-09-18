"""On-demand English Wikipedia retrieval with an auditable response cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .retrieval_service import RetrievalHandler, RetrievalServer


class WikipediaRateLimited(RuntimeError):
    pass


class WikipediaIndex:
    schema = "spgfs-online-wikipedia-v1"
    document_count = 0  # There is no preloaded local corpus.
    upstream_concurrency = 3
    request_interval_s = 0.4

    def __init__(self, cache: Path, *, timeout: float = 15, attempts: int = 2,
                 contact: str | None = None, opener=None):
        self.opener = opener
        if contact is None:
            contact = os.environ.get("SPGFS_WIKIPEDIA_CONTACT", "").strip()
            contact_path = Path(__file__).resolve().parents[2] / "state/private/wikipedia-contact.txt"
            if not contact and contact_path.exists():
                contact = contact_path.read_text().strip()
        if any(char in contact for char in "\r\n"):
            raise ValueError("Wikipedia contact must be a single line")
        self.contact = contact
        self.user_agent = f"SelfPlayGraphFlowSteer/0.1 ({contact})"
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.attempts = attempts
        self.slots = threading.BoundedSemaphore(self.upstream_concurrency)
        self.rate_lock = threading.Lock()
        self.last_request = 0.0
        self.blocked_until = 0.0
        self.key_locks = [threading.Lock() for _ in range(64)]

    def _request(self, params: dict) -> dict:
        if not self.contact:
            raise RuntimeError("Set SPGFS_WIKIPEDIA_CONTACT to a real contact email or project URL")
        params = {"action": "query", "format": "json", "formatversion": 2,
                  "maxlag": 5, **params}
        request = Request(
            "https://en.wikipedia.org/w/api.php?" + urlencode(params),
            headers={"User-Agent": self.user_agent,
                     "Accept": "application/json"},
        )
        for attempt in range(self.attempts):
            try:
                with self.slots:
                    with self.rate_lock:
                        if time.monotonic() < self.blocked_until:
                            raise WikipediaRateLimited("Wikipedia rate-limited; retry after cooldown")
                        time.sleep(max(0, self.request_interval_s - (time.monotonic() - self.last_request)))
                        self.last_request = time.monotonic()
                    open_request = self.opener.open if self.opener is not None else urlopen
                    with open_request(request, timeout=self.timeout) as response:
                        payload = json.load(response)
                if not isinstance(payload, dict):
                    raise RuntimeError("Wikipedia returned invalid JSON")
                if "error" in payload:
                    raise RuntimeError("Wikipedia API error: " + str(payload["error"].get("code")))
                return payload
            except WikipediaRateLimited:
                raise
            except (HTTPError, URLError, TimeoutError, ValueError, RuntimeError) as error:
                if isinstance(error, HTTPError) and error.code == 429:
                    retry_after = error.headers.get("Retry-After", "60")
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        try:
                            delay = (parsedate_to_datetime(retry_after) - datetime.now(UTC)).total_seconds()
                        except (ValueError, TypeError):
                            delay = 60
                    with self.rate_lock:
                        self.blocked_until = max(self.blocked_until, time.monotonic() + max(1, delay))
                    raise WikipediaRateLimited("Wikipedia HTTP 429; upstream requests paused") from error
                logging.warning("Wikipedia %s request failed: %s (HTTP %s), attempt %s/%s",
                                "search" if "generator" in params else "article",
                                type(error).__name__, getattr(error, "code", "n/a"),
                                attempt + 1, self.attempts)
                if attempt + 1 == self.attempts:
                    raise RuntimeError("Wikipedia upstream request failed") from error
                time.sleep(1 + attempt)
        raise AssertionError("unreachable")

    def _fetch(self, query: str, top_k: int) -> dict:
        payload = self._request({"generator": "search", "gsrsearch": query,
                                 "gsrnamespace": 0, "gsrlimit": top_k, "prop": "info"})
        pages = payload.get("query", {}).get("pages", [])

        def enrich(page):
            article = self._request({"pageids": page["pageid"],
                                     "prop": "extracts|info|revisions", "explaintext": 1,
                                     "inprop": "url", "rvprop": "ids"})
            full = article.get("query", {}).get("pages", [])
            if full:
                page.update(full[0])
            return page

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(enrich, pages))
        return payload

    def search(self, query: str, top_k: int) -> list[dict]:
        query = query.strip()
        if not query or len(query) > 2000:
            raise ValueError("Wikipedia query must contain 1-2000 characters")
        key = hashlib.sha256(json.dumps([self.schema, query, top_k]).encode()).hexdigest()
        path = self.cache / (key + ".json")
        with self.key_locks[int(key[:2], 16) % len(self.key_locks)]:
            if path.exists():
                return json.loads(path.read_text())["result"]
            payload = self._fetch(query, top_k)
            pages = sorted(payload.get("query", {}).get("pages", []),
                           key=lambda page: page.get("index", 0))
            result = []
            for page in pages:
                text = page.get("extract", "").strip()
                if not text:
                    continue
                title = page["title"]
                # Keep the lead plus query-relevant paragraphs from the full article.
                paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
                terms = set(query.casefold().split()) - {"the", "a", "of", "in", "is", "who", "what", "and"}
                ranked = sorted(range(1, len(paragraphs)),
                                key=lambda i: (-sum(t in paragraphs[i].casefold() for t in terms), i))
                selected, remaining = {0}, max(0, 6000 - len(paragraphs[0]))
                for i in ranked:
                    if len(paragraphs[i]) <= remaining:
                        selected.add(i)
                        remaining -= len(paragraphs[i])
                excerpt = "\n".join(paragraphs[i] for i in sorted(selected))[:6000]
                result.append({"document": {
                    "id": str(page["pageid"]), "title": title, "text": excerpt,
                    "contents": title + "\n" + excerpt,
                    "url": page.get("fullurl"),
                    "revision_id": (page.get("revisions") or [{}])[0].get("revid"),
                }})
            if pages and not result:
                raise RuntimeError("Wikipedia returned pages without usable article text")
            record = {"schema": self.schema, "query": query, "top_k": top_k,
                      "retrieved_at": datetime.now(UTC).isoformat(),
                      "endpoint": "https://en.wikipedia.org/w/api.php",
                      "result": result, "raw_response": payload}
            with tempfile.NamedTemporaryFile(mode="w", dir=self.cache, delete=False) as temporary:
                json.dump(record, temporary, ensure_ascii=False)
            os.replace(temporary.name, path)
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    with RetrievalServer((args.host, args.port), RetrievalHandler) as server:
        server.index = WikipediaIndex(args.cache)
        server.serve_forever()


if __name__ == "__main__":
    main()
