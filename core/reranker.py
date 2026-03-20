"""Optional cross-encoder reranker using fastembed."""

from __future__ import annotations

_DEFAULT_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
_cross_encoder = None
_loaded_model: str | None = None


def rerank(
    query: str,
    items: list[dict],
    top_k: int = 5,
    model: str | None = None,
) -> list[dict]:
    """Re-score items with a cross-encoder and return them sorted by relevance.

    Each item must have a "text" key. Items are returned with updated "score".
    """
    if not items:
        return items

    global _cross_encoder, _loaded_model
    model = model or _DEFAULT_MODEL
    if _cross_encoder is None or _loaded_model != model:
        from fastembed import TextCrossEncoder
        _cross_encoder = TextCrossEncoder(model)
        _loaded_model = model

    scores = list(_cross_encoder.rerank(query, [item["text"] for item in items]))

    for entry in scores:
        items[entry["index"]]["score"] = float(entry["score"])

    items.sort(key=lambda x: x["score"], reverse=True)
    return items[:top_k]
