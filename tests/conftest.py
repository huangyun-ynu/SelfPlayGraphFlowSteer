"""Unit tests use dummy provider credentials, never a developer's API keys."""

import pytest


@pytest.fixture(autouse=True)
def dummy_provider_credentials(monkeypatch):
    for name in (
        "MINIMAX_OFFICIAL_API_KEY",
        "DEEPSEEK_OFFICIAL_API_KEY",
        "NEXUS_API_KEY",
        "NEXUS_PRO_API_KEY",
        "FLOWSTEER_API_KEY",
        "UUAPI_API_KEY",
        "UUAPI_API_KEY_2",
        "DEEPSEEK_API_KEY",
        "MINIMAX_API_KEY",
        "TOOLCODE_API_KEY",
        "ZOOAPI_API_KEY",
        "HILXYO_API_KEY",
    ):
        monkeypatch.setenv(name, "unit-test-key")
