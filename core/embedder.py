"""Unified embedding interface, supporting ollama / openai backends."""

from __future__ import annotations

from collections.abc import Callable

from .config import load_config

_OLLAMA_TIMEOUT = 120
_FALLBACK_TEXT_LEN = 400

_sparse_model = None


def sparse_embed(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    """Generate sparse (BM25) vectors using fastembed.

    Returns list of (indices, values) tuples.
    """
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding
        _sparse_model = SparseTextEmbedding("Qdrant/bm25")
    results = []
    for sparse in _sparse_model.embed(texts):
        results.append((sparse.indices.tolist(), sparse.values.tolist()))
    return results


def embed(texts: list[str], provider: str | None = None, model: str | None = None, progress_callback: Callable[[int, int], None] | None = None) -> list[list[float]]:
    """Convert a list of texts to a list of vectors.

    progress_callback: optional callable(current: int, total: int)
    """
    cfg = load_config()["embedding"]
    provider = provider or cfg["provider"]
    model = model or cfg["model"]

    if provider == "ollama":
        return _embed_ollama(texts, model, cfg.get("ollama_base_url", "http://localhost:11434"), progress_callback)
    elif provider == "openai":
        return _embed_openai(texts, model, cfg, progress_callback)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def _sanitize_text(text: str) -> str:
    """Remove characters that can cause Ollama embedding NaN/500 errors."""
    import re
    # Replace null bytes and other control characters (keep newlines/tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    # Collapse excessive whitespace
    text = re.sub(r' {10,}', '  ', text)
    return text.strip()


def _embed_ollama(texts: list[str], model: str, base_url: str, progress_callback=None) -> list[list[float]]:
    import httpx
    base_url = base_url.replace("localhost", "127.0.0.1")
    results = []
    total = len(texts)
    with httpx.Client(timeout=_OLLAMA_TIMEOUT) as client:
        for i, text in enumerate(texts):
            sanitized = _sanitize_text(text)
            try:
                resp = client.post(
                    f"{base_url}/api/embed",
                    json={"model": model, "input": sanitized},
                )
                resp.raise_for_status()
                results.append(resp.json()["embeddings"][0])
            except httpx.HTTPStatusError:
                fallback = sanitized[:_FALLBACK_TEXT_LEN] if len(sanitized) > _FALLBACK_TEXT_LEN else "empty chunk"
                resp = client.post(
                    f"{base_url}/api/embed",
                    json={"model": model, "input": fallback},
                )
                resp.raise_for_status()
                results.append(resp.json()["embeddings"][0])
            if progress_callback:
                progress_callback(i + 1, total)
    return results


def _embed_openai(texts: list[str], model: str, cfg: dict, progress_callback=None) -> list[list[float]]:
    import openai as _openai
    from .auth import resolve_openai_api_key
    llm_cfg = load_config().get("llm", {})
    client = _openai.OpenAI(
        api_key=resolve_openai_api_key({**llm_cfg, **cfg}),
        base_url=cfg.get("base_url") or None,
    )
    resp = client.embeddings.create(input=texts, model=model)
    if progress_callback:
        progress_callback(len(texts), len(texts))
    return [item.embedding for item in resp.data]
