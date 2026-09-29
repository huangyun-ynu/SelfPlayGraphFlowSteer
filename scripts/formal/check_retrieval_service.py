"""Check backend identity, not just whether an unrelated service has HTTP 200."""
import json
import os
from urllib.request import ProxyHandler, build_opener

schema = {
    "wikipedia": "spgfs-online-wikipedia-v1",
    "faiss": "spgfs-searchr1-e5-faiss-v1",
    "sqlite": "spgfs-local-retrieval-v1",
}[os.environ.get("SPGFS_RETRIEVAL_BACKEND", "wikipedia")]
backend = os.environ.get("SPGFS_RETRIEVAL_BACKEND", "wikipedia")
try:
    port = int(os.environ.get("SPGFS_RETRIEVAL_PORT", "18010"))
    with build_opener(ProxyHandler({})).open(
        f"http://127.0.0.1:{port}/health", timeout=2
    ) as response:
        health = json.load(response)
    ready = health.get("status") == "ok" and health.get("schema") == schema
    ready = ready and (schema == "spgfs-online-wikipedia-v1" or health.get("document_count", 0) > 0)
    if backend == "faiss":
        assets = health.get("asset_identity")
        ready = ready and health.get("profile_id") == os.environ.get(
            "SPGFS_RETRIEVAL_PROFILE", "nq-dense8-v1"
        )
        ready = ready and isinstance(assets, dict)
        if ready:
            corpus = assets.get("corpus", {})
            index = assets.get("index", {})
            if corpus.get("repo") == "local/R2D2-pruned-NQ":
                ready = health.get("document_count") == 1_702_133
                ready = ready and corpus.get("sha256") == os.environ.get(
                    "SPGFS_RETRIEVAL_CORPUS_SHA256"
                )
                ready = ready and index.get("sha256") == os.environ.get(
                    "SPGFS_RETRIEVAL_INDEX_SHA256"
                )
            else:
                ready = health.get("document_count") == 21_015_324
                ready = ready and corpus.get("sha256") == (
                    "43d7d3f58d01d711d95b00b70584211eea639fa46802905a4b7e11cf0617752d"
                )
                ready = ready and index.get("parts_sha256") == [
                    "a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023",
                    "b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11",
                ]
            ready = ready and assets.get("encoder", {}).get("files_sha256", {}).get(
                "model.safetensors"
            ) == "d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693"
        identity = health.get("identity_sha256")
        ready = ready and isinstance(identity, str) and len(identity) == 64
        expected = os.environ.get("SPGFS_RETRIEVAL_IDENTITY_SHA256")
        if expected:
            ready = ready and identity == expected
except Exception:
    ready = False
raise SystemExit(0 if ready else 1)
