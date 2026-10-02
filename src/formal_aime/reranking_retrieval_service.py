"""CPU MiniLM/RRF gateway over an already-running, pinned dense retriever."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from .passage_reranker import PassageReranker
from .public_evidence import public_search_results
from .retrieval_service import RetrievalBatcher, RetrievalHandler, RetrievalServer


class RerankingRetrievalIndex:
    schema = "spgfs-searchr1-e5-faiss-v1"

    def __init__(self, upstream_url: str, expected_identity: str, reranker: PassageReranker,
                 *, candidate_k: int = 20, timeout_s: float = 240.0) -> None:
        parsed = urlsplit(upstream_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                or parsed.path != "/retrieve" or parsed.username or parsed.password
                or parsed.query or parsed.fragment):
            raise ValueError("reranking upstream must be a loopback /retrieve endpoint")
        if not 8 <= candidate_k <= 20 or timeout_s <= 0:
            raise ValueError("gateway candidate_k must be between 8 and 20")
        if reranker.fusion != "rrf":
            raise ValueError("MiniLM/RRF profile requires rrf fusion")
        self.upstream_url = upstream_url
        self.health_url = parsed._replace(path="/health").geturl()
        self.expected_identity = expected_identity
        self.timeout_s = timeout_s
        self.candidate_k = candidate_k
        self.reranker = reranker
        self.opener = build_opener(ProxyHandler({}))
        health = self._check_upstream()
        self.document_count = int(health["document_count"])
        profile = {
            "profile_id": "nq-minilm-rrf8-v1", "candidate_k": candidate_k,
            "reranker_sha256": reranker.fingerprint, "fusion": "rrf", "rrf_constant": 60,
            "dense_keep": reranker.dense_keep, "upstream_identity_sha256": expected_identity,
        }
        identity = {"assets": health["asset_identity"], "profile": profile}
        self.health_identity = {
            "profile_id": profile["profile_id"], "asset_identity": copy.deepcopy(health["asset_identity"]),
            "identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            "ranking": profile, "upstream_url": upstream_url,
        }

    def _fetch(self, request: Request | str) -> dict:
        try:
            with self.opener.open(request, timeout=self.timeout_s) as response:
                result = json.load(response)
        except Exception as exc:
            raise RuntimeError("pinned dense retrieval request failed") from exc
        if not isinstance(result, dict):
            raise RuntimeError("invalid dense retrieval response")
        return result

    def _check_upstream(self) -> dict:
        health = self._fetch(self.health_url)
        if (health.get("status") != "ok" or health.get("schema") != self.schema
                or health.get("profile_id") != "nq-dense8-v1"
                or health.get("identity_sha256") != self.expected_identity
                or not isinstance(health.get("asset_identity"), dict)):
            raise RuntimeError("dense retrieval identity/profile changed")
        return health

    def search_batch(self, queries: list[str], top_k: int) -> list[list[dict]]:
        if not queries:
            return []
        if not 1 <= top_k <= 8:
            raise ValueError("MiniLM/RRF profile returns at most 8 passages")
        health = self._check_upstream()
        if health["document_count"] != self.document_count:
            raise RuntimeError("dense corpus document count changed")
        response = self._fetch(Request(
            self.upstream_url,
            data=json.dumps({"queries": queries, "topk": self.candidate_k, "return_scores": True}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        ))
        if not isinstance(response.get("result"), list):
            raise RuntimeError("dense retrieval response has no result groups")
        groups = public_search_results(response["result"])
        if len(groups) != len(queries):
            raise RuntimeError("dense retrieval returned wrong query count")
        corpus_id = health["asset_identity"]["corpus"]["sha256"]
        if any("score" not in hit or hit["document"].get("corpus_id") != corpus_id
               for hits in groups for hit in hits):
            raise RuntimeError("dense retrieval returned invalid scores or foreign corpus passages")
        return [self.reranker.rerank(query, hits[:self.candidate_k], top_k)
                for query, hits in zip(queries, groups, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--expected-upstream-identity", required=True)
    parser.add_argument("--reranker", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost", "::1"])
    parser.add_argument("--port", type=int, default=19013)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO)
    reranker = PassageReranker(args.reranker, device="cpu", fusion="rrf")
    index = RerankingRetrievalIndex(args.upstream, args.expected_upstream_identity, reranker,
                                  candidate_k=args.candidate_k)
    server = RetrievalServer((args.host, args.port), RetrievalHandler)
    server.index = index
    server.batcher = RetrievalBatcher(index, batch_size=8, max_wait_ms=20, max_pending=256)
    logging.info("CPU reranking gateway ready: %s", json.dumps(index.health_identity, sort_keys=True))
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
