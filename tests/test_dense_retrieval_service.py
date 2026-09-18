import json
import io
import tarfile
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from selfplay_graph_flowsteer.dense_retrieval_service import JsonlCorpus
from selfplay_graph_flowsteer.retrieval_service import RetrievalHandler, RetrievalServer


def test_corpus_preserves_row_order_and_refreshes_offsets(tmp_path: Path):
    path = tmp_path / "corpus.jsonl"
    docs = [{"id": 42, "contents": "first passage"}, {"id": 9, "text": "second passage"}]
    path.write_text("\n".join(json.dumps(doc) for doc in docs))
    corpus = JsonlCorpus(path)
    assert len(corpus) == 2
    assert corpus[0] == docs[0]
    assert corpus[1] == docs[1]
    with pytest.raises(IndexError):
        corpus[-1]
    corpus.mapping.close()
    path.write_text(json.dumps(docs[1]) + "\n")
    rebuilt = JsonlCorpus(path)
    assert len(rebuilt) == 1
    assert rebuilt[0] == docs[1]
    rebuilt.mapping.close()


def test_corpus_rejects_blank_rows(tmp_path: Path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"contents":"passage"}\n\n')
    with pytest.raises(ValueError, match="Empty corpus row"):
        JsonlCorpus(path)


def test_corpus_rejects_tar_mislabeled_as_jsonl(tmp_path: Path):
    path = tmp_path / "corpus.jsonl"
    with tarfile.open(path, mode="w") as bundle:
        data = b'{"contents":"passage"}\n'
        member = tarfile.TarInfo("wiki_dump.jsonl")
        member.size = len(data)
        bundle.addfile(member, io.BytesIO(data))
    with pytest.raises(ValueError, match="contains a TAR archive"):
        JsonlCorpus(path)


def test_http_retrieval_scores_validation_and_health():
    class Index:
        schema = "spgfs-searchr1-e5-faiss-v1"
        document_count = 1

        def search(self, query, top_k):
            return [{"document": {"contents": query}, "score": 0.9}]

    opener = build_opener(ProxyHandler({}))
    with RetrievalServer(("127.0.0.1", 0), RetrievalHandler) as server:
        server.index = Index()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            with opener.open(url + "/health") as response:
                assert json.load(response)["schema"] == Index.schema
            for scores in (True, False):
                request = Request(url + "/retrieve", data=json.dumps(
                    {"queries": ["first", "second"], "topk": 5, "return_scores": scores}
                ).encode(), headers={"Content-Type": "application/json"})
                with opener.open(request) as response:
                    groups = json.load(response)["result"]
                assert len(groups) == 2
                assert groups[0][0] == (
                    {"document": {"contents": "first"}, "score": 0.9}
                    if scores else {"contents": "first"}
                )
            for payload in ([], {"queries": []}, {"queries": ["test"], "topk": 0}):
                with pytest.raises(HTTPError) as error:
                    opener.open(Request(url + "/retrieve", data=json.dumps(payload).encode()))
                assert error.value.code == 400
        finally:
            server.shutdown()
            thread.join()
