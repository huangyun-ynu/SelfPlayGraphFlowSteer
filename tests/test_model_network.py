from __future__ import annotations

from unittest.mock import Mock, patch
from urllib.request import Request

from selfplay_graph_flowsteer import model_network


def test_local_model_request_uses_proxy_free_opener(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    response = Mock()

    with patch.object(model_network._DIRECT_OPENER, "open", return_value=response) as direct:
        actual = model_network.model_urlopen(
            Request("http://127.0.0.1:18603/tokenize"), timeout=30
        )

    assert actual is response
    direct.assert_called_once()
    assert direct.call_args.kwargs == {"timeout": 30}

