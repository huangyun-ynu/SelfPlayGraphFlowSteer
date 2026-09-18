import json
from urllib.error import HTTPError

import pytest

from selfplay_graph_flowsteer.wikipedia_retrieval_service import WikipediaIndex


def test_online_search_fetches_articles_individually_and_caches(tmp_path, monkeypatch):
    index = WikipediaIndex(tmp_path)
    calls = []

    def request(params):
        calls.append(params)
        if "generator" in params:
            return {"query": {"pages": [{"pageid": 2, "index": 2}, {"pageid": 1, "index": 1}]}}
        pageid = params["pageids"]
        return {"query": {"pages": [{"pageid": pageid, "title": f"Page {pageid}",
                                      "extract": "Lead.\nHamlet was written by Shakespeare.",
                                      "revisions": [{"revid": 123}], "fullurl": "https://en.wikipedia.org/wiki/Hamlet"}]}}

    monkeypatch.setattr(index, "_request", request)
    result = index.search("Who wrote Hamlet?", 2)
    assert [hit["document"]["id"] for hit in result] == ["1", "2"]
    assert len(calls) == 3
    assert index.search("Who wrote Hamlet?", 2) == result
    assert len(calls) == 3
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["raw_response"]["query"]["pages"][0]["extract"]
    assert record["result"][0]["document"]["revision_id"] == 123
    assert record["retrieved_at"]


def test_upstream_failure_is_not_cached_as_empty(tmp_path, monkeypatch):
    index = WikipediaIndex(tmp_path)

    def fail(*args):
        raise RuntimeError("upstream failed")

    monkeypatch.setattr(index, "_fetch", fail)
    with pytest.raises(RuntimeError):
        index.search("Hamlet", 5)
    assert not list(tmp_path.glob("*.json"))


def test_no_hits_is_cached_but_missing_article_text_is_an_error(tmp_path, monkeypatch):
    index = WikipediaIndex(tmp_path)
    monkeypatch.setattr(index, "_fetch", lambda *args: {})
    assert index.search("nonexistent", 5) == []
    monkeypatch.setattr(index, "_fetch", lambda *args: {"query": {"pages": [{"pageid": 1}]}})
    with pytest.raises(RuntimeError, match="without usable"):
        index.search("missing article", 5)


def test_rate_limit_pauses_other_queries_without_retry_storm(tmp_path, monkeypatch):
    index = WikipediaIndex(tmp_path, contact="maintainer@example.org")
    calls = []

    def limited(*args, **kwargs):
        calls.append(1)
        raise HTTPError("https://en.wikipedia.org/w/api.php", 429, "limited",
                        {"Retry-After": "60"}, None)

    monkeypatch.setattr("selfplay_graph_flowsteer.wikipedia_retrieval_service.urlopen", limited)
    with pytest.raises(RuntimeError, match="429"):
        index.search("first", 5)
    with pytest.raises(RuntimeError, match="cooldown"):
        index.search("second", 5)
    assert len(calls) == 1
    assert not list(tmp_path.glob("*.json"))


def test_contact_is_sent_on_outgoing_request(tmp_path, monkeypatch):
    from io import BytesIO

    index = WikipediaIndex(tmp_path, contact="maintainer@example.org")
    headers = []

    def fetch(request, **kwargs):
        headers.append(request.get_header("User-agent"))
        return BytesIO(b'{}')

    monkeypatch.setattr("selfplay_graph_flowsteer.wikipedia_retrieval_service.urlopen", fetch)
    assert index._request({"generator": "search", "gsrsearch": "test"}) == {}
    assert headers == ["SelfPlayGraphFlowSteer/0.1 (maintainer@example.org)"]
    with pytest.raises(RuntimeError, match="CONTACT"):
        WikipediaIndex(tmp_path, contact="")._request({})
