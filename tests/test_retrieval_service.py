from pathlib import Path

from selfplay_graph_flowsteer.retrieval_service import RetrievalIndex, create_index


def test_local_retrieval_index_returns_public_documents(tmp_path: Path) -> None:
    path = tmp_path / "evidence.sqlite3"
    create_index(
        path,
        [
            {
                "id": "hamlet",
                "title": "Hamlet",
                "text": "Hamlet is a tragedy written by William Shakespeare.",
                "url": "https://en.wikipedia.org/wiki/Hamlet",
            },
            {
                "id": "other",
                "title": "Other",
                "text": "An unrelated document about chemistry.",
                "url": "https://example.invalid/other",
            },
        ],
    )

    index = RetrievalIndex(path)
    hits = index.search("Who wrote Hamlet?", 1)

    assert index.document_count == 2
    assert hits[0]["document"] == {
        "id": "hamlet",
        "title": "Hamlet",
        "text": "Hamlet is a tragedy written by William Shakespeare.",
    }
    assert isinstance(hits[0]["score"], float)
