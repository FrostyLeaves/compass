"""Unified access layer: auto-detects API server, falls back to direct access.

When the web server (uvicorn api:app) is running, CLI operations go through
HTTP so they don't conflict with the server's Qdrant file lock.
When the server is not running, operations use direct module imports.
"""

from __future__ import annotations

import time
from pathlib import Path

_SERVER_DETECT_TIMEOUT = 2
_SERVER_CACHE_TTL = 30
_SEARCH_TIMEOUT = 60
_ASK_TIMEOUT = 120
_LIST_TIMEOUT = 30
_REMOVE_TIMEOUT = 30
_AUDIT_TIMEOUT = 300
_server_online: bool | None = None
_server_checked_at: float = 0.0


def _api_url() -> str:
    from .config import load_config, get_api_port
    cfg = load_config()
    return cfg.get("api", {}).get("url", f"http://localhost:{get_api_port()}")


def _auth_headers() -> dict[str, str]:
    """Return auth headers for write operations (ingest/remove/reindex/audit)."""
    from .config import load_config
    cfg = load_config()
    secret = cfg.get("api", {}).get("secret_key", "")
    if secret:
        return {"X-API-Key": secret}
    return {}


def is_server_running() -> bool:
    """Check if the API server is reachable (cached with short TTL)."""
    global _server_online, _server_checked_at
    now = time.monotonic()
    if _server_online is not None and (now - _server_checked_at) < _SERVER_CACHE_TTL:
        return _server_online
    import httpx
    try:
        resp = httpx.get(f"{_api_url()}/api/status", timeout=_SERVER_DETECT_TIMEOUT)
        _server_online = resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        _server_online = False
    _server_checked_at = now
    return _server_online


def search(
    query: str,
    top_k: int | None = None,
    filter_title: str | None = None,
    filter_keywords: list[str] | None = None,
    **kwargs,
) -> list[dict]:
    if is_server_running():
        import httpx
        body: dict = {"query": query, "top_k": top_k}
        if filter_title:
            body["filter_title"] = filter_title
        if filter_keywords:
            body["filter_keywords"] = filter_keywords
        with httpx.Client(timeout=_SEARCH_TIMEOUT) as client:
            resp = client.post(f"{_api_url()}/api/search", json=body)
            resp.raise_for_status()
            return resp.json()
    from . import retriever
    return retriever.search(
        query, top_k=top_k,
        filter_title=filter_title, filter_keywords=filter_keywords,
        **kwargs,
    )


def ask(question: str, top_k: int | None = None, **kwargs) -> tuple[str, list[dict]]:
    """Returns (answer, sources)."""
    if is_server_running():
        import httpx
        with httpx.Client(timeout=_ASK_TIMEOUT) as client:
            resp = client.post(
                f"{_api_url()}/api/ask",
                json={"question": question, "top_k": top_k},
            )
            resp.raise_for_status()
            data = resp.json()
        return data["answer"], data["sources"]
    from . import retriever, generator
    results = retriever.search(question, top_k=top_k, deduplicate=False, **kwargs)
    if not results:
        return "No relevant papers found in Compass.", []
    answer = generator.generate(question=question, context=results, **kwargs)
    # Deduplicate for display — keep best chunk per paper
    seen: dict[str, dict] = {}
    for r in results:
        key = r.get("paper_id") or r.get("source_path", "")
        if key not in seen or r["score"] > seen[key]["score"]:
            seen[key] = r
    display = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return answer, display


def list_papers() -> list[dict]:
    if is_server_running():
        import httpx
        with httpx.Client(timeout=_LIST_TIMEOUT) as client:
            resp = client.get(f"{_api_url()}/api/papers")
            resp.raise_for_status()
            return resp.json()
    from . import ingest
    return ingest.list_papers()


def ingest_paper(
    path: str,
    progress_callback=None,
    **kwargs,
) -> list[dict]:
    """Ingest a PDF file, directory, or URL. Returns list of result dicts."""
    if is_server_running():
        import httpx
        timeout = httpx.Timeout(connect=30, read=None, write=30, pool=30)
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{_api_url()}/api/ingest",
                json={"path": path},
                headers=_auth_headers(),
            )
            if resp.status_code >= 400:
                detail = (
                    resp.json().get("detail", resp.text)
                    if resp.headers.get("content-type", "").startswith("application/json")
                    else resp.text
                )
                raise RuntimeError(f"Server error ({resp.status_code}): {detail}")
            return resp.json()
    from . import ingest as _ingest
    if path.startswith("http://") or path.startswith("https://"):
        return [_ingest.ingest_url(path, progress_callback=progress_callback, **kwargs)]
    p = Path(path)
    if p.is_file():
        return [_ingest.ingest_pdf(p, progress_callback=progress_callback, **kwargs)]
    elif p.is_dir():
        return _ingest.ingest_directory(p, recursive=True, **kwargs)
    raise FileNotFoundError(f"Not found: {path}")


def remove(paper_id: str) -> bool:
    if is_server_running():
        import httpx
        with httpx.Client(timeout=_REMOVE_TIMEOUT) as client:
            resp = client.post(
                f"{_api_url()}/api/remove",
                json={"paper_id": paper_id},
                headers=_auth_headers(),
            )
            return resp.status_code == 200
    from . import ingest
    return ingest.remove_paper(paper_id)


def reindex(
    paper_id: str | None = None,
    progress_callback=None,
    **kwargs,
) -> list[dict]:
    """Re-embed papers from saved markdown. If paper_id is given, reindex that paper only."""
    if is_server_running():
        import httpx
        body: dict = {}
        if paper_id:
            body["paper_id"] = paper_id
        timeout = httpx.Timeout(connect=30, read=None, write=30, pool=30)
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{_api_url()}/api/reindex",
                json=body,
                headers=_auth_headers(),
            )
            resp.raise_for_status()
            return resp.json()
    from . import ingest as _ingest
    if paper_id:
        return [_ingest.reindex_paper(paper_id, progress_callback=progress_callback, **kwargs)]
    return _ingest.reindex_all(progress_callback=progress_callback, **kwargs)


def audit(progress_callback=None) -> dict:
    if is_server_running():
        import httpx
        with httpx.Client(timeout=_AUDIT_TIMEOUT) as client:
            resp = client.post(
                f"{_api_url()}/api/audit",
                headers=_auth_headers(),
            )
            resp.raise_for_status()
            return resp.json()
    from . import ingest
    return ingest.audit_translations(progress_callback=progress_callback)
