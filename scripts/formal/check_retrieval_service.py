"""Check backend identity, not just whether an unrelated service has HTTP 200."""
import json
import os
from urllib.request import ProxyHandler, build_opener

schema = {
    "wikipedia": "spgfs-online-wikipedia-v1",
    "faiss": "spgfs-searchr1-e5-faiss-v1",
    "sqlite": "spgfs-local-retrieval-v1",
}[os.environ.get("SPGFS_RETRIEVAL_BACKEND", "wikipedia")]
try:
    port = int(os.environ.get("SPGFS_RETRIEVAL_PORT", "18010"))
    with build_opener(ProxyHandler({})).open(
        f"http://127.0.0.1:{port}/health", timeout=2
    ) as response:
        health = json.load(response)
    ready = health.get("status") == "ok" and health.get("schema") == schema
    ready = ready and (schema == "spgfs-online-wikipedia-v1" or health.get("document_count", 0) > 0)
except Exception:
    ready = False
raise SystemExit(0 if ready else 1)
