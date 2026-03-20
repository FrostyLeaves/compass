"""Tests for the Compass FastAPI backend."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from api import app
from server import mcp
from core.config import load_config

FAKE_PAPERS = [
    {
        "paper_id": "abc123",
        "title": "Attention Is All You Need",
        "path": "/data/papers/abc123/paper.pdf",
        "chunks_count": 12,
        "markdown_path": "/data/papers/abc123/paper.md",
        "pdf_path": "/data/papers/abc123/paper.pdf",
        "source_url": "https://arxiv.org/pdf/2106.00001",
        "keywords": ["transformer", "attention"],
        "ingested_at": "2025-01-15T10:00:00Z",
    },
    {
        "paper_id": "def456",
        "title": "BERT: Pre-training",
        "path": "/data/papers/def456/paper.pdf",
        "chunks_count": 8,
        "markdown_path": "/data/papers/def456/paper.md",
        "pdf_path": "/data/papers/def456/paper.pdf",
        "source_url": "",
        "keywords": ["bert", "pretraining"],
        "ingested_at": "2025-02-20T12:00:00Z",
    },
]

FAKE_SEARCH_RESULTS = [
    {
        "paper_id": "abc123",
        "title": "Attention Is All You Need",
        "display_title": "",
        "text": "The dominant sequence transduction models...",
        "score": 0.92,
        "source_path": "/data/papers/abc123/paper.pdf",
        "chunk_index": 3,
        "markdown_path": "/data/papers/abc123/paper.md",
        "pdf_path": "/data/papers/abc123/paper.pdf",
        "keywords": ["transformer", "attention"],
        "source_url": "https://arxiv.org/pdf/2106.00001",
        "ingested_at": "2025-01-15T10:00:00Z",
    },
]


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client


# ── GET /api/papers ──────────────────────────────────────────────

@pytest.mark.anyio
@patch("api._list_papers_impl", return_value=FAKE_PAPERS)
async def test_list_papers(mock_list, client):
    resp = await client.get("/api/papers")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    # default sort: ingested_at desc → BERT first
    assert data[0]["title"] == "BERT: Pre-training"


@pytest.mark.anyio
@patch("api._list_papers_impl", return_value=FAKE_PAPERS)
async def test_list_papers_sort_title_asc(mock_list, client):
    resp = await client.get("/api/papers?sort_by=title&order=asc")
    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["title"] == "Attention Is All You Need"


# ── GET /api/papers/{paper_id}/content ──────────────────────────

@pytest.mark.anyio
@patch("api._get_paper_impl", return_value=None)
async def test_paper_content_not_found(mock_get, client):
    resp = await client.get("/api/papers/nonexistent/content")
    assert resp.status_code == 404


@pytest.mark.anyio
@patch("api.Path.read_text", return_value="# Attention\n\nSome content")
@patch("api.Path.exists", return_value=True)
@patch("api._get_paper_impl", return_value=FAKE_PAPERS[0])
async def test_paper_content_ok(mock_get, mock_exists, mock_read, client):
    resp = await client.get("/api/papers/abc123/content")
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Attention Is All You Need"
    assert "Attention" in data["content"]


# ── POST /api/search ─────────────────────────────────────────────

@pytest.mark.anyio
@patch("api._search_impl", return_value=FAKE_SEARCH_RESULTS)
async def test_search(mock_search, client):
    resp = await client.post("/api/search", json={"query": "transformer"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["score"] == pytest.approx(0.92)
    mock_search.assert_called_once_with(
        query="transformer", top_k=None, deduplicate=True,
        filter_title=None, filter_keywords=None,
    )


@pytest.mark.anyio
@patch("api._search_impl", return_value=FAKE_SEARCH_RESULTS)
async def test_search_with_top_k(mock_search, client):
    resp = await client.post("/api/search", json={"query": "attention", "top_k": 3})
    assert resp.status_code == 200
    mock_search.assert_called_once_with(
        query="attention", top_k=3, deduplicate=True,
        filter_title=None, filter_keywords=None,
    )


@pytest.mark.anyio
async def test_search_missing_query(client):
    resp = await client.post("/api/search", json={})
    assert resp.status_code == 422


# ── POST /api/ask ─────────────────────────────────────────────────

@pytest.mark.anyio
@patch("api._generate_impl", return_value="Transformers use self-attention.")
@patch("api._search_impl", return_value=FAKE_SEARCH_RESULTS)
async def test_ask(mock_search, mock_gen, client):
    resp = await client.post("/api/ask", json={"question": "What is a transformer?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "self-attention" in data["answer"]
    assert len(data["sources"]) == 1


@pytest.mark.anyio
@patch("api._search_impl", return_value=[])
async def test_ask_no_results(mock_search, client):
    resp = await client.post("/api/ask", json={"question": "unknown topic"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "No relevant papers found."
    assert resp.json()["sources"] == []


# ── GET /api/status ──────────────────────────────────────────────

@pytest.mark.anyio
@patch("core.status.check_qdrant", return_value=(True, "42 chunks indexed"))
@patch("core.status.check_llm", return_value=(True, "claude/claude-sonnet-4-6"))
@patch("core.status.check_embedding", return_value=(True, "ollama/bge-m3"))
async def test_status(mock_emb, mock_llm, mock_qdr, client):
    resp = await client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["embedding"]["ok"] is True
    assert data["llm"]["ok"] is True
    assert data["qdrant"]["ok"] is True
    assert "42" in data["qdrant"]["detail"]


@pytest.mark.anyio
@patch("api._localize_title", return_value="localized-title")
@patch("api._is_within_directory", side_effect=[False, True])
async def test_localize_titles_filters_outside_paths(mock_within, mock_localize, client):
    outside = str(Path("/tmp/papers-evil/paper.md").resolve())
    inside = str(Path("/tmp/papers/abc123/paper.md").resolve())
    resp = await client.post(
        "/api/localize-titles",
        json={"markdown_paths": [outside, inside], "lang": "zh"},
    )
    assert resp.status_code == 200
    assert resp.json() == {inside: "localized-title"}
    mock_localize.assert_called_once_with(inside, "zh")


@pytest.mark.anyio
async def test_mcp_config_excludes_write_tools(client):
    resp = await client.get("/api/mcp-config")
    assert resp.status_code == 200
    tool_names = {tool["name"] for tool in resp.json()["tools"]}
    assert tool_names == {"search_papers", "read_paper", "list_papers"}


def test_server_mcp_tools_exclude_write_tools():
    tool_names = set(mcp._tool_manager._tools.keys())
    assert tool_names == {"search_papers", "read_paper", "list_papers"}


@pytest.mark.anyio
@patch("api._ingest_url_impl", return_value={
    "paper_id": "paper123",
    "title": "Sample Paper",
    "chunks_count": 7,
    "path": "/data/papers/paper123/paper.pdf",
    "markdown_path": "/data/papers/paper123/paper.md",
    "pdf_path": "/data/papers/paper123/paper.pdf",
    "source_url": "https://example.com/paper.pdf",
    "error": "",
})
async def test_ingest_endpoint_ok(mock_ingest, client):
    secret = load_config().get("api", {}).get("secret_key", "")
    headers = {"X-API-Key": secret} if secret else {}
    resp = await client.post("/api/ingest", json={"path": "https://example.com/paper.pdf"}, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["title"] == "Sample Paper"
    mock_ingest.assert_called_once_with("https://example.com/paper.pdf")


@pytest.mark.anyio
@patch("api._remove_paper_impl", return_value=True)
async def test_remove_endpoint_ok(mock_remove, client):
    secret = load_config().get("api", {}).get("secret_key", "")
    headers = {"X-API-Key": secret} if secret else {}
    resp = await client.post("/api/remove", json={"paper_id": "abc123"}, headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    mock_remove.assert_called_once_with("abc123")
