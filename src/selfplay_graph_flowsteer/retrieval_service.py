from __future__ import annotations

import argparse
import json
import re
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "what",
    "when", "where", "which", "who", "why", "with",
}


def create_index(path: Path, documents: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.execute(
            "CREATE VIRTUAL TABLE documents USING fts5("
            "doc_id UNINDEXED, title, text, url UNINDEXED, tokenize='porter unicode61')"
        )
        connection.executemany(
            "INSERT INTO documents(doc_id, title, text, url) VALUES (?, ?, ?, ?)",
            [
                (item["id"], item["title"], item["text"], item.get("url", ""))
                for item in documents
            ],
        )
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [("schema", "spgfs-local-retrieval-v1"), ("document_count", str(len(documents)))],
        )
        connection.commit()
    finally:
        connection.close()
    temporary.replace(path)


class RetrievalIndex:
    schema = "spgfs-local-retrieval-v1"

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"retrieval index does not exist: {self.path}")
        with self._connect() as connection:
            schema = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema'"
            ).fetchone()
            if schema != ("spgfs-local-retrieval-v1",):
                raise ValueError("retrieval index has an incompatible schema")
            self.document_count = int(
                connection.execute(
                    "SELECT value FROM metadata WHERE key = 'document_count'"
                ).fetchone()[0]
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.path}?mode=ro&immutable=1", uri=True)

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        terms = [term for term in _TOKEN.findall(query.casefold()) if term not in _STOPWORDS]
        terms = list(dict.fromkeys(terms))[:32]
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT doc_id, title, text, url, bm25(documents, 6.0, 1.0) AS rank "
                "FROM documents WHERE documents MATCH ? ORDER BY rank LIMIT ?",
                (expression, top_k),
            ).fetchall()
        return [
            {
                "document": {"id": row[0], "title": row[1], "text": row[2]},
                "score": float(-row[4]),
            }
            for row in rows
        ]


class RetrievalServer(ThreadingHTTPServer):
    request_queue_size = 128
    index: RetrievalIndex
    batcher: RetrievalBatcher | None = None

    def server_close(self) -> None:
        if self.batcher is not None:
            self.batcher.close()
        super().server_close()


class RetrievalOverloaded(RuntimeError):
    """The bounded retrieval queue has no free slots."""


class RetrievalTimedOut(RuntimeError):
    """Queueing plus retrieval exceeded the service deadline."""


@dataclass
class _QueuedQuery:
    query: str
    top_k: int
    ready: threading.Event = field(default_factory=threading.Event)
    result: list[dict[str, Any]] | None = None
    error: Exception | None = None
    cancelled: bool = False


class RetrievalBatcher:
    """Collect queries across HTTP requests for one encoder and FAISS call."""

    def __init__(
        self, index: Any, *, batch_size: int = 8, max_wait_ms: int = 20,
        max_pending: int = 128, timeout_s: float = 240.0,
    ) -> None:
        if batch_size < 1 or max_wait_ms < 0 or max_pending < batch_size or timeout_s <= 0:
            raise ValueError("invalid retrieval batcher limits")
        self.index = index
        self.batch_size = batch_size
        self.max_wait_s = max_wait_ms / 1000
        self.max_pending = max_pending
        self.timeout_s = timeout_s
        self._condition = threading.Condition()
        self._queue: deque[_QueuedQuery] = deque()
        self._outstanding = 0
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="retrieval-batcher", daemon=True)
        self._worker.start()

    def search_batch(self, queries: list[str], top_k: int) -> list[list[dict[str, Any]]]:
        if not queries:
            return []
        pending = [_QueuedQuery(query, top_k) for query in queries]
        with self._condition:
            if self._closed:
                raise RuntimeError("retrieval batcher is closed")
            if self._outstanding + len(pending) > self.max_pending:
                raise RetrievalOverloaded("retrieval queue is full")
            self._queue.extend(pending)
            self._outstanding += len(pending)
            self._condition.notify_all()
        deadline = time.monotonic() + self.timeout_s
        try:
            results = []
            for item in pending:
                if not item.ready.wait(max(0.0, deadline - time.monotonic())):
                    raise RetrievalTimedOut("retrieval request timed out")
                if item.error is not None:
                    raise RuntimeError("retrieval batch failed") from item.error
                assert item.result is not None
                results.append(item.result)
            return results
        except (RetrievalTimedOut, RuntimeError):
            # The worker drops queued entries and eventually releases in-flight slots.
            with self._condition:
                for item in pending:
                    item.cancelled = True
                self._condition.notify_all()
            raise

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if self._closed and not self._queue:
                    return
                deadline = time.monotonic() + self.max_wait_s
                while len(self._queue) < self.batch_size and not self._closed:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                batch = [self._queue.popleft() for _ in range(min(self.batch_size, len(self._queue)))]
                active = [item for item in batch if not item.cancelled]
            if active:
                try:
                    results = self.index.search_batch(
                        [item.query for item in active], max(item.top_k for item in active)
                    )
                    if len(results) != len(active):
                        raise RuntimeError("retrieval result count does not match query count")
                    for item, result in zip(active, results, strict=True):
                        item.result = result[:item.top_k]
                except Exception as exc:  # propagate search failures to each waiting request
                    for item in active:
                        item.error = exc
            with self._condition:
                self._outstanding -= len(batch)
                for item in batch:
                    item.ready.set()
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._worker.join(timeout=5)


class RetrievalHandler(BaseHTTPRequestHandler):
    server: RetrievalServer

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        health = {
            "status": "ok",
            "schema": self.server.index.schema,
            "document_count": self.server.index.document_count,
        }
        if hasattr(self.server.index, "health_identity"):
            health.update(self.server.index.health_identity)
        if self.server.batcher is not None:
            health["batching"] = {
                "batch_size": self.server.batcher.batch_size,
                "max_wait_ms": int(self.server.batcher.max_wait_s * 1000),
                "max_pending": self.server.batcher.max_pending,
            }
        self._send(HTTPStatus.OK, health)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/retrieve":
            self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("invalid request length")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
            queries = payload.get("queries")
            top_k = int(payload.get("topk", 3))
            if (
                not isinstance(queries, list)
                or not queries
                or len(queries) > 32
                or not all(isinstance(query, str) and query.strip() for query in queries)
                or not 1 <= top_k <= 20
            ):
                raise ValueError("invalid retrieval request")
            if self.server.batcher is not None:
                result = self.server.batcher.search_batch(queries, top_k)
            elif hasattr(self.server.index, "search_batch"):
                result = self.server.index.search_batch(queries, top_k)
            else:
                result = [self.server.index.search(query, top_k) for query in queries]
            if payload.get("return_scores", False) is False:
                result = [[hit["document"] for hit in group] for group in result]
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except RetrievalOverloaded as exc:
            self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc), "retryable": True})
            return
        except RetrievalTimedOut as exc:
            self._send(HTTPStatus.GATEWAY_TIMEOUT, {"error": str(exc), "retryable": True})
            return
        except RuntimeError as exc:
            self._send(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
            return
        self._send(HTTPStatus.OK, {"result": result})

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # A rollout may already have exhausted its client-side deadline.
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local public evidence index")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    server = RetrievalServer((args.host, args.port), RetrievalHandler)
    server.index = RetrievalIndex(args.index)
    server.serve_forever()


if __name__ == "__main__":
    main()
