from __future__ import annotations

import argparse
import json
import re
import sqlite3
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


class RetrievalHandler(BaseHTTPRequestHandler):
    server: RetrievalServer

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        self._send(
            HTTPStatus.OK,
            {
                "status": "ok",
                "schema": self.server.index.schema,
                "document_count": self.server.index.document_count,
            },
        )

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
            result = [self.server.index.search(query, top_k) for query in queries]
            if payload.get("return_scores", False) is False:
                result = [[hit["document"] for hit in group] for group in result]
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
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
