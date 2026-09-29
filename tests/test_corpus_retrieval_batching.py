import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
import pytest

from selfplay_graph_flowsteer import dense_retrieval_service as dense
from selfplay_graph_flowsteer.passage_reranker import select_reranked
from selfplay_graph_flowsteer.retrieval_service import (
    RetrievalBatcher,
    RetrievalHandler,
    RetrievalOverloaded,
    RetrievalServer,
)


def test_dense_search_batches_queries_once_and_preserves_document_rows():
    index = object.__new__(dense.DenseRetrievalIndex)
    index.lock = threading.Lock()
    index.threads = 1
    index.document_count = 4
    index.reranker = None
    index.asset_identity = {"corpus": {"sha256": "corpus-fingerprint"}}
    index.faiss = SimpleNamespace(omp_set_num_threads=lambda _: None)
    encoded = []

    def encode(queries, *, query):
        encoded.append((queries, query))
        return [[i, 1.0] for i in range(len(queries))]

    index.encoder = SimpleNamespace(encode=encode)
    searched = []

    def search(matrix, count):
        searched.append((matrix.shape, count))
        return np.array([[0.9, 0.8, 0.7, 0.6], [0.8, 0.7, 0.6, 0.5]]), np.array(
            [[2, 0, 1, 3], [3, 1, 0, 2]]
        )

    index.index = SimpleNamespace(search=search, d=2)
    index.corpus = [
        {"id": "a", "contents": "Alpha"},
        {"id": "b", "contents": "Beta"},
        {"id": "c", "contents": "Gamma"},
        {"id": "d", "contents": "Delta"},
    ]
    groups = index.search_batch(["first question", "second question"], 2)
    assert encoded == [(["first question", "second question"], True)]
    assert searched == [((2, 2), 4)]
    assert [[hit["document"]["id"] for hit in group] for group in groups] == [
        ["c", "a"], ["d", "b"]
    ]
    assert groups[0][0]["faiss_row"] == 2
    assert groups[0][0]["document"]["faiss_row"] == 2
    assert groups[0][0]["document"]["corpus_id"] == "corpus-fingerprint"


def test_http_requests_are_microbatched_without_mixing_responses():
    class Index:
        schema = "test"
        document_count = 100
        health_identity = {"profile_id": "nq-dense8-v1", "identity_sha256": "a" * 64}

        def __init__(self):
            self.batches = []
            self.lock = threading.Lock()

        def search_batch(self, queries, top_k):
            with self.lock:
                self.batches.append(tuple(queries))
            return [[{"document": {"id": query, "contents": query}, "score": 1.0}]
                    for query in queries]

    index = Index()
    opener = build_opener(ProxyHandler({}))
    barrier = threading.Barrier(12)
    with RetrievalServer(("127.0.0.1", 0), RetrievalHandler) as server:
        server.index = index
        server.batcher = RetrievalBatcher(index, batch_size=8, max_wait_ms=100, max_pending=16)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            with opener.open(url + "/health") as response:
                health = json.load(response)
            assert health["profile_id"] == "nq-dense8-v1"
            assert health["batching"]["batch_size"] == 8

            def request(number):
                query = f"question {number}"
                barrier.wait(timeout=5)
                payload = json.dumps({"queries": [query], "topk": 1,
                                      "return_scores": True}).encode()
                with opener.open(Request(url + "/retrieve", data=payload), timeout=5) as response:
                    return json.load(response)["result"][0][0]["document"]["id"]

            with ThreadPoolExecutor(max_workers=12) as executor:
                assert list(executor.map(request, range(12))) == [f"question {i}" for i in range(12)]
            assert sum(len(batch) for batch in index.batches) == 12
            assert max(map(len, index.batches)) > 1
        finally:
            server.shutdown()
            thread.join()


def test_bounded_queue_rejects_excess_during_active_search():
    started = threading.Event()
    release = threading.Event()

    class Index:
        def search_batch(self, queries, top_k):
            started.set()
            assert release.wait(5)
            return [[{"document": {"id": query}}] for query in queries]

    batcher = RetrievalBatcher(Index(), batch_size=1, max_wait_ms=0, max_pending=1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(batcher.search_batch, ["first"], 1)
        assert started.wait(5)
        with pytest.raises(RetrievalOverloaded):
            batcher.search_batch(["second"], 1)
        release.set()
        assert first.result(timeout=5)[0][0]["document"]["id"] == "first"
    batcher.close()


def test_asset_identity_requires_prepared_content_and_exact_encoder(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    corpus = source_dir / "wiki-18.jsonl"
    corpus.write_text('{"id":"0","contents":"passage"}\n')
    index = source_dir / "e5_Flat.index"
    index.write_bytes(b"tiny index")
    expected = {
        "part_aa": ("index-revision", 1, "a" * 64),
        "part_ab": ("index-revision", 1, "b" * 64),
        "wiki-18.jsonl.gz": ("corpus-revision", 1, "c" * 64),
    }
    monkeypatch.setattr(dense, "EXPECTED_SOURCES", expected)
    sources = [{"repo": "PeterJinGo/wiki-18-corpus" if name.endswith(".gz")
                else "PeterJinGo/wiki-18-e5-index", "revision": revision,
                "file": name, "bytes": size, "sha256": digest}
               for name, (revision, size, digest) in expected.items()]
    manifest = {"sources": sources, "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                "corpus_bytes": corpus.stat().st_size, "index_bytes": index.stat().st_size}
    (source_dir / "manifest.json").write_text(json.dumps(manifest))
    model = tmp_path / "model"
    model.mkdir()
    files = []
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        path = model / name
        path.write_bytes(name.encode())
        files.append({"path": name, "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (model / "download_manifest.json").write_text(json.dumps({
        "repo_id": dense.EXPECTED_ENCODER_REPO,
        "revision": dense.EXPECTED_ENCODER_REVISION, "files": files,
    }))
    identity = dense.asset_identity(index, corpus, model)
    assert identity["corpus"]["sha256"] == manifest["corpus_sha256"]
    assert identity["encoder"]["files_sha256"]["model.safetensors"] == files[1]["sha256"]
    (model / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        dense.asset_identity(index, corpus, model)


def test_rrf_reranking_uses_only_public_passage_scores():
    hits = [{"document": {"id": i, "contents": f"Public {i}"}, "score": 1 - i / 10}
            for i in range(10)]
    ranked = select_reranked(hits, [0] * 8 + [10, 10], 8, fusion="rrf")
    selected = next(hit for hit in ranked if hit["document"]["id"] == 8)
    assert selected["dense_rank"] == 9
    assert selected["reranker_score"] == 10.0
