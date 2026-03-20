"""Compass — FastAPI REST API backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.routing import Route

from core.config import load_config, get_papers_dir


def _list_papers_impl() -> list[dict]:
    from core import ingest
    return ingest.list_papers()


def _get_paper_impl(paper_id: str) -> dict | None:
    from core import ingest
    return ingest.get_paper(paper_id)


def _search_impl(
    *,
    query: str,
    top_k: int | None = None,
    deduplicate: bool = False,
    filter_title: str | None = None,
    filter_keywords: list[str] | None = None,
) -> list[dict]:
    from core import retriever
    return retriever.search(
        query=query,
        top_k=top_k,
        deduplicate=deduplicate,
        filter_title=filter_title,
        filter_keywords=filter_keywords,
    )


def _generate_impl(*, question: str, context: list[dict]) -> str:
    from core import generator
    return generator.generate(question=question, context=context)


def _ingest_url_impl(url: str) -> dict:
    from core import ingest
    return ingest.ingest_url(url)


def _ingest_pdf_impl(path: Path) -> dict:
    from core import ingest
    return ingest.ingest_pdf(path)


def _ingest_directory_impl(path: Path) -> list[dict]:
    from core import ingest
    return ingest.ingest_directory(path, recursive=True)


def _reindex_paper_impl(paper_id: str) -> dict:
    from core import ingest
    return ingest.reindex_paper(paper_id)


def _reindex_all_impl() -> list[dict]:
    from core import ingest
    return ingest.reindex_all()


def _remove_paper_impl(paper_id: str) -> bool:
    from core import ingest
    return ingest.remove_paper(paper_id)


def _audit_translations_impl() -> dict:
    from core import ingest
    return ingest.audit_translations()


def _is_within_directory(path: Path, root: Path) -> bool:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
        return True
    except ValueError:
        return False


def _require_secret(x_api_key: str | None = Header(None)):
    """Dependency: reject write endpoints if secret_key is set and doesn't match."""
    cfg = load_config()
    secret = cfg.get("api", {}).get("secret_key", "")
    if secret and x_api_key != secret:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


# ── Mount MCP server (Streamable HTTP transport) ─────────────────
from server import mcp as mcp_server


mcp_http_app = mcp_server.streamable_http_app()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # FastMCP's Streamable HTTP app needs its session manager to be started
    # by the parent app lifecycle.
    async with mcp_server.session_manager.run():
        yield


app = FastAPI(title="Compass API", version="0.1.0", lifespan=lifespan)
app.router.routes.append(Route("/mcp", endpoint=mcp_http_app))

_cors_origins = load_config().get("api", {}).get(
    "cors_origins", ["http://localhost:5173", "http://localhost:3000"],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve paper assets (images, PDFs) as static files
papers_dir = get_papers_dir()
if papers_dir.exists():
    app.mount("/static/papers", StaticFiles(directory=str(papers_dir)), name="papers")

# ── Pydantic models ──────────────────────────────────────────────

class PaperItem(BaseModel):
    paper_id: str
    title: str
    display_title: str = ""
    path: str
    chunks_count: int
    markdown_path: str
    pdf_path: str
    source_url: str
    keywords: list[str] = []
    ingested_at: str


def _localize_title(markdown_path: str, lang: str | None) -> str | None:
    """Read H1 from a translated markdown file. Returns None if not found."""
    if not lang or not markdown_path:
        return None
    md = Path(markdown_path)
    lang_md = md.parent / f"{md.stem}.{lang}.md"
    if not lang_md.exists():
        return None
    for line in lang_md.read_text(encoding="utf-8").split('\n')[:10]:
        line = line.strip()
        if line.startswith('# '):
            return line[2:].strip()
    return None


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None
    lang: str | None = None
    filter_title: str | None = None
    filter_keywords: list[str] | None = None


class SearchResultItem(BaseModel):
    paper_id: str
    title: str
    display_title: str = ""
    text: str
    score: float
    source_path: str
    chunk_index: int
    markdown_path: str
    pdf_path: str = ""
    keywords: list[str] = []
    source_url: str = ""
    ingested_at: str = ""


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None
    lang: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SearchResultItem]


class ServiceStatus(BaseModel):
    ok: bool
    detail: str


class StatusResponse(BaseModel):
    embedding: ServiceStatus
    llm: ServiceStatus
    qdrant: ServiceStatus


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/api/papers", response_model=list[PaperItem])
def list_papers(
    sort_by: Literal["title", "ingested_at", "chunks_count"] = "ingested_at",
    order: Literal["asc", "desc"] = "desc",
    lang: str | None = None,
):
    papers = _list_papers_impl()
    for p in papers:
        if lang:
            t = _localize_title(p.get("markdown_path", ""), lang)
            p["display_title"] = t or p["title"]
        else:
            p["display_title"] = p["title"]
    reverse = order == "desc"
    papers.sort(key=lambda p: p.get(sort_by, ""), reverse=reverse)
    return papers


@app.get("/api/papers/{paper_id}/content")
def get_paper_content(paper_id: str, lang: str | None = None):
    paper = _get_paper_impl(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    md_path = paper.get("markdown_path", "")
    if not md_path or not Path(md_path).exists():
        raise HTTPException(status_code=404, detail="Markdown content not available")
    md = Path(md_path)
    # If a language is requested, try to find the translated version
    if lang:
        lang_md = md.parent / f"{md.stem}.{lang}.md"
        if lang_md.exists():
            md = lang_md
    content = md.read_text(encoding="utf-8")
    folder = md.parent.name
    return {
        "paper_id": paper_id,
        "title": paper["title"],
        "content": content,
        "folder": folder,
        "source_url": paper.get("source_url", ""),
    }


@app.get("/api/i18n")
def get_i18n():
    cfg = load_config()
    i18n = cfg.get("i18n", {"enabled": False, "languages": []})
    return i18n


def _localize_results(results: list[dict], lang: str | None) -> list[dict]:
    for r in results:
        if lang:
            t = _localize_title(r.get("markdown_path", ""), lang)
            r["display_title"] = t or r["title"]
        else:
            r["display_title"] = r["title"]
    return results


class LocalizeTitlesRequest(BaseModel):
    markdown_paths: list[str]
    lang: str


@app.post("/api/localize-titles")
def localize_titles(req: LocalizeTitlesRequest):
    result: dict[str, str] = {}
    papers_root = get_papers_dir()
    for mp in req.markdown_paths:
        resolved_mp = Path(mp).resolve()
        if not _is_within_directory(resolved_mp, papers_root):
            continue
        t = _localize_title(str(resolved_mp), req.lang)
        if t:
            result[mp] = t
    return result


@app.post("/api/search", response_model=list[SearchResultItem])
def search_papers(req: SearchRequest):
    results = _search_impl(
        query=req.query,
        top_k=req.top_k,
        deduplicate=True,
        filter_title=req.filter_title,
        filter_keywords=req.filter_keywords,
    )
    return _localize_results(results, req.lang)


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    sources = _search_impl(query=req.question, top_k=req.top_k, deduplicate=False)
    if not sources:
        return AskResponse(answer="No relevant papers found.", sources=[])
    answer = _generate_impl(question=req.question, context=sources)
    # Deduplicate for display — keep best chunk per paper
    seen: dict[str, dict] = {}
    for s in sources:
        key = s.get("paper_id") or s.get("source_path", "")
        if key not in seen or s["score"] > seen[key]["score"]:
            seen[key] = s
    display_sources = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return AskResponse(answer=answer, sources=_localize_results(display_sources, req.lang))


class IngestRequest(BaseModel):
    path: str  # file path, directory, or URL


class IngestResultItem(BaseModel):
    paper_id: str = ""
    title: str
    chunks_count: int
    path: str
    markdown_path: str = ""
    pdf_path: str = ""
    source_url: str = ""
    error: str = ""


@app.post("/api/ingest", response_model=list[IngestResultItem], dependencies=[Depends(_require_secret)])
def ingest_papers(req: IngestRequest):
    target = req.path
    try:
        if target.startswith("http://") or target.startswith("https://"):
            result = _ingest_url_impl(target)
            return [result]
        path = Path(target)
        if path.is_file():
            result = _ingest_pdf_impl(path)
            return [result]
        elif path.is_dir():
            return _ingest_directory_impl(path)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, OSError, KeyError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=404, detail=f"Path not found: {target}")


class ReindexRequest(BaseModel):
    paper_id: str | None = None


class ReindexResultItem(BaseModel):
    paper_id: str = ""
    title: str
    chunks_count: int
    keywords: list[str] = []
    error: str = ""


@app.post("/api/reindex", response_model=list[ReindexResultItem], dependencies=[Depends(_require_secret)])
def reindex_papers(req: ReindexRequest):
    if req.paper_id:
        try:
            return [_reindex_paper_impl(req.paper_id)]
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
    return _reindex_all_impl()


class RemoveRequest(BaseModel):
    paper_id: str


@app.post("/api/remove", dependencies=[Depends(_require_secret)])
def remove_paper(req: RemoveRequest):
    if _remove_paper_impl(req.paper_id):
        return {"ok": True}
    raise HTTPException(status_code=404, detail="Paper not found")


class AuditResponse(BaseModel):
    papers_scanned: int
    titles_fixed: int
    translations_added: int
    translations_removed: int


def _get_base_url() -> str:
    cfg = load_config()
    api_cfg = cfg.get("api", {})
    host = api_cfg.get("host", "localhost")
    port = api_cfg.get("port", 8000)
    return api_cfg.get("url", f"http://{host}:{port}")


def _get_mcp_url() -> str:
    return f"{_get_base_url()}/mcp"


def _get_skill_prompt() -> str:
    return """You have access to the Compass MCP server, an academic paper knowledge base.

Recommended workflow:
1. Use `search_papers` first to find relevant passages for a topic or question.
2. Use `read_paper` when you need the full paper or a named section.
3. Use `list_papers` to browse available papers or recover an exact title/paper_id.
4. Ground paper-related answers in retrieved content instead of guessing.

Behavior rules:
- Always search before answering paper-related questions.
- Prefer `read_paper(section=...)` for long papers instead of dumping the full text.
- Cite paper titles and source URLs when responding.
- If search returns nothing relevant, say so plainly instead of fabricating an answer.
"""


def _build_mcp_guide_text() -> str:
    import json

    mcp_url = _get_mcp_url()
    skill_prompt = _get_skill_prompt().rstrip()

    claude_desktop_cfg = json.dumps({"mcpServers": {"compass": {"type": "streamable-http", "url": mcp_url}}}, indent=2)
    claude_code_cfg = json.dumps({"mcpServers": {"compass": {"type": "streamable-http", "url": mcp_url}}}, indent=2)
    cursor_cfg = json.dumps({"mcpServers": {"compass": {"type": "streamable-http", "url": mcp_url}}}, indent=2)
    codex_cfg = json.dumps({"mcpServers": {"compass": {"type": "streamable-http", "url": mcp_url}}}, indent=2)

    return f"""\
# Compass — MCP Auto-Setup Guide For AI Agents

This document is intended to be read by an AI agent.

Goal:
1. Configure the Compass MCP server in your MCP-capable client.
2. Store the Compass skill prompt below in your memory/skills/system instructions.
3. Verify the connection by listing tools or calling `list_papers`.

If you are able to modify your own client configuration, do it now.
If you are not able to modify your own client configuration, show the user the
exact snippet for their client from this document.

## MCP Endpoint

    {mcp_url}

Transport: Streamable HTTP

## Available Tools

1. **search_papers** — Search the knowledge base for passages relevant to a query.
   Args: query (str), top_k (int, default 5), filter_title (str), filter_keywords (str, comma-separated)

2. **read_paper** — Read the full content or a specific section of a paper.
   Args: title (str — paper title or paper_id), section (str, optional — heading name)

3. **list_papers** — List all papers in the knowledge base with titles and chunk counts.

## Install Instructions

### Claude Desktop — claude_desktop_config.json

```json
{claude_desktop_cfg}
```

### Claude Code — .claude/settings.json

```json
{claude_code_cfg}
```

CLI alternative:

    claude mcp add compass --transport streamable-http {mcp_url}

### Cursor — .cursor/mcp.json

```json
{cursor_cfg}
```

### OpenAI Codex — codex.json

```json
{codex_cfg}
```

CLI alternative:

    codex --mcp-server-url {mcp_url}

## Skill Prompt To Store

```text
{skill_prompt}
```

## Verification

After configuration:
1. Connect to the MCP server at `{mcp_url}`.
2. Confirm the server exposes the expected tools.
3. Run `list_papers`.
4. Optionally run `search_papers` with a simple query to confirm retrieval works.

## Notes

- This MCP server is for academic paper retrieval and reading.
- Prefer `search_papers` before `read_paper`.
- Do not invent paper content if retrieval does not support the answer.
"""


@app.post("/api/audit", response_model=AuditResponse, dependencies=[Depends(_require_secret)])
def audit_translations():
    result = _audit_translations_impl()
    return result


@app.get("/api/mcp-config")
def mcp_config(request: Request):
    """Return MCP endpoint URL and client configuration snippets."""
    mcp_url = _get_mcp_url()

    return {
        "mcp_url": mcp_url,
        "skill_prompt": _get_skill_prompt(),
        "tools": [
            {"name": "search_papers", "description": "Search the paper knowledge base for passages relevant to a query"},
            {"name": "read_paper", "description": "Read the full content or a specific section of a paper"},
            {"name": "list_papers", "description": "List all papers in the knowledge base"},
        ],
        "clients": {
            "claude_desktop": {
                "file": "claude_desktop_config.json",
                "config": {
                    "mcpServers": {
                        "compass": {
                            "type": "streamable-http",
                            "url": mcp_url,
                        }
                    }
                },
            },
            "claude_code": {
                "file": ".claude/settings.json",
                "config": {
                    "mcpServers": {
                        "compass": {
                            "type": "streamable-http",
                            "url": mcp_url,
                        }
                    }
                },
            },
            "cursor": {
                "file": ".cursor/mcp.json",
                "config": {
                    "mcpServers": {
                        "compass": {
                            "type": "streamable-http",
                            "url": mcp_url,
                        }
                    }
                },
            },
            "codex": {
                "file": "codex.json or CLI flag",
                "config": {
                    "mcpServers": {
                        "compass": {
                            "type": "streamable-http",
                            "url": mcp_url,
                        }
                    }
                },
                "cli": f"codex --mcp-server-url {mcp_url}",
            },
        },
    }


@app.get("/api/mcp-guide", response_class=PlainTextResponse)
def mcp_guide():
    """LLM-readable guide for configuring and using the Compass MCP server."""
    return _build_mcp_guide_text()

    import json

    cfg = load_config()
    api_cfg = cfg.get("api", {})
    host = api_cfg.get("host", "localhost")
    port = api_cfg.get("port", 8000)
    base_url = api_cfg.get("url", f"http://{host}:{port}")
    mcp_url = f"{base_url}/mcp"

    claude_desktop_cfg = json.dumps({"mcpServers": {"compass": {"type": "streamable-http", "url": mcp_url}}}, indent=2)
    claude_code_cfg = json.dumps({"mcpServers": {"compass": {"type": "streamable-http", "url": mcp_url}}}, indent=2)
    cursor_cfg = json.dumps({"mcpServers": {"compass": {"type": "streamable-http", "url": mcp_url}}}, indent=2)
    codex_cfg = json.dumps({"mcpServers": {"compass": {"type": "streamable-http", "url": mcp_url}}}, indent=2)

    return f"""\
# Compass — MCP Server Setup Guide

Compass provides an MCP (Model Context Protocol) server that gives AI clients
direct access to an academic paper knowledge base. Transport: Streamable HTTP.

## MCP Endpoint

    {mcp_url}

## Available Tools

1. **search_papers** — Search the knowledge base for passages relevant to a query.
   Args: query (str), top_k (int, default 5), filter_title (str), filter_keywords (str, comma-separated)

2. **read_paper** — Read the full content or a specific section of a paper.
   Args: title (str — paper title or paper_id), section (str, optional — heading name)

3. **list_papers** — List all papers in the knowledge base with titles and chunk counts.

## Client Configuration

### Claude Desktop — claude_desktop_config.json

```json
{claude_desktop_cfg}
```

### Claude Code — .claude/settings.json

```json
{claude_code_cfg}
```

Or via CLI:

    claude mcp add compass --transport streamable-http {mcp_url}

### Cursor — .cursor/mcp.json

```json
{cursor_cfg}
```

### OpenAI Codex — codex.json

```json
{codex_cfg}
```

Or via CLI:

    codex --mcp-server-url {mcp_url}

## Recommended Workflow

1. Use `search_papers` to find relevant passages for a topic or question.
2. Use `read_paper` to get full text or a specific section of a paper.
3. Use `list_papers` to browse all available papers.
4. Synthesize information from retrieved content to answer the user's question.

## Tips

- Always search before answering paper-related questions — do not guess content.
- Use `read_paper` with a `section` argument for long papers to avoid truncation.
- Cite paper titles and source URLs in your responses.
- `search_papers` returns deduplicated results (one best passage per paper).
"""


@app.get("/api/status", response_model=StatusResponse)
def get_status():
    from core.status import check_embedding, check_llm, check_qdrant
    cfg = load_config()

    emb_ok, emb_detail = check_embedding(cfg)
    llm_ok, llm_detail = check_llm(cfg)
    qdr_ok, qdr_detail = check_qdrant()

    return StatusResponse(
        embedding=ServiceStatus(ok=emb_ok, detail=emb_detail),
        llm=ServiceStatus(ok=llm_ok, detail=llm_detail),
        qdrant=ServiceStatus(ok=qdr_ok, detail=qdr_detail),
    )
