from __future__ import annotations

from unittest.mock import patch

import httpx

from core import client


def setup_function():
    client._server_online = None
    client._server_checked_at = 0.0
    client._server_base_url = None


def test_api_url_candidates_prefer_ipv4_for_localhost():
    with patch("core.config.load_config", return_value={"api": {"host": "localhost", "port": 8000}}):
        assert client._api_url_candidates() == ["http://127.0.0.1:8000", "http://localhost:8000"]


def test_api_url_candidates_normalize_explicit_localhost_url():
    with patch("core.config.load_config", return_value={"api": {"url": "http://localhost:8000"}}):
        assert client._api_url_candidates() == ["http://127.0.0.1:8000", "http://localhost:8000"]


def test_is_server_running_falls_back_to_second_candidate():
    calls: list[str] = []

    def fake_get(url: str, timeout: int):
        calls.append(url)
        if url.startswith("http://127.0.0.1:8000"):
            raise httpx.ConnectError("refused")
        return httpx.Response(200, request=httpx.Request("GET", url))

    with patch("core.config.load_config", return_value={"api": {"host": "localhost", "port": 8000}}):
        with patch("httpx.get", side_effect=fake_get):
            assert client.is_server_running() is True

    assert calls == [
        "http://127.0.0.1:8000/api/status",
        "http://localhost:8000/api/status",
    ]
    assert client._api_url() == "http://localhost:8000"
