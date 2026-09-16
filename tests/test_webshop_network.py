from __future__ import annotations

from unittest.mock import patch

from selfplay_graph_flowsteer import webshop


def test_local_webshop_request_uses_proxy_free_opener(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    client = webshop.WebShopHTTPClient("http://127.0.0.1:8020")

    with patch.object(webshop._DIRECT_OPENER, "open") as direct:
        direct.return_value.__enter__.return_value.read.return_value = b'{"status":"ok"}'
        result = client.health()

    assert result == {"status": "ok"}
    direct.assert_called_once()
    request = direct.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:8020/health"
    assert direct.call_args.kwargs == {"timeout": 10.0}

