"""Optional model proxy; local endpoints and explicit direct routes bypass it."""

from __future__ import annotations

import io
import ipaddress
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener


_DIRECT_OPENER = build_opener(ProxyHandler({}))


def model_proxy(base_url: str, network_path: str = "configured_proxy") -> str | None:
    if network_path == "direct":
        return None
    if network_path != "configured_proxy":
        raise ValueError("unknown model network_path")
    host = (urlsplit(base_url).hostname or "").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return None
    try:
        address = ipaddress.ip_address(host)
        if address.is_private or address.is_loopback:
            return None
    except ValueError:
        pass
    proxy = os.environ.get("MODEL_API_PROXY", "").strip()
    if not proxy:
        return None
    parsed = urlsplit(proxy)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("MODEL_API_PROXY must be an HTTP(S) proxy when configured")
    return proxy


def model_urlopen(request, *, timeout):
    """urllib-shaped transport for Gemini; preserve existing HTTP error handling."""
    proxy = model_proxy(request.full_url)
    if proxy is None:
        # urllib's module-level urlopen still honors HTTP(S)_PROXY.  Local and
        # explicitly direct model routes must bypass inherited proxy settings.
        return _DIRECT_OPENER.open(request, timeout=timeout)
    import httpx

    with httpx.Client(proxy=proxy, trust_env=False, timeout=timeout) as client:
        try:
            response = client.request(
                request.get_method(),
                request.full_url,
                content=request.data,
                headers=dict(request.header_items()),
            )
        except httpx.RequestError as exc:
            raise URLError(exc) from exc
        if response.status_code >= 300:
            raise HTTPError(
                request.full_url,
                response.status_code,
                response.reason_phrase,
                response.headers,
                io.BytesIO(response.content),
            )
        return io.BytesIO(response.content)
