"""Hybrid vector search with configurable lexical fallback."""

from __future__ import annotations

import re

from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchText,
    MatchValue,
    Prefetch,
    SparseVector,
)

from .config import load_config, to_absolute
from .embedder import embed, sparse_embed
from .vectorstore import _COLLECTION, _get_client

_DEFAULT_QUERY_STRATEGY_MODE = "semantic_with_lexical_fallback"
_DEFAULT_KEYWORD_PHRASE_BOOST = 0.2
_DEFAULT_KEYWORD_TOKEN_OVERLAP_WEIGHT = 0.15
_DEFAULT_LEXICAL_MAX_CANDIDATES = 3
_DEFAULT_LEXICAL_MIN_CANDIDATE_SCORE = 0.35
_DEFAULT_LEXICAL_MIN_STRONG_TITLE_OVERLAP = 0.6
_DEFAULT_MIN_TOKEN_LENGTH = 2
_DEFAULT_TITLE_EXACT_BOOST = 1.0
_DEFAULT_TITLE_SUBSTRING_BOOST = 0.35
_DEFAULT_TITLE_TOKEN_OVERLAP_WEIGHT = 0.35
_FETCH_K_MULTIPLIER = 3
_MARKDOWN_RE = re.compile(r"[*_`#]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")
_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
    "of", "on", "or", "the", "to", "with",
}


def _has_sparse_vectors(client) -> bool:
    """Check if the collection was created with sparse vector support."""
    try:
        info = client.get_collection(_COLLECTION)
        sparse_cfg = info.config.params.sparse_vectors
        return sparse_cfg is not None and "sparse" in sparse_cfg
    except (ValueError, AttributeError):
        return False


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_query_strategy_config(cfg: dict) -> dict:
    strategy_cfg = cfg.get("retrieval", {}).get("query_strategy", {})
    lexical_cfg = strategy_cfg.get("lexical", {})
    return {
        "mode": strategy_cfg.get("mode", _DEFAULT_QUERY_STRATEGY_MODE),
        "keyword_phrase_boost": _as_float(
            lexical_cfg.get("keyword_phrase_boost"),
            _DEFAULT_KEYWORD_PHRASE_BOOST,
        ),
        "keyword_token_overlap_weight": _as_float(
            lexical_cfg.get("keyword_token_overlap_weight"),
            _DEFAULT_KEYWORD_TOKEN_OVERLAP_WEIGHT,
        ),
        "max_candidates": max(0, _as_int(
            lexical_cfg.get("max_candidates"),
            _DEFAULT_LEXICAL_MAX_CANDIDATES,
        )),
        "min_candidate_score": _as_float(
            lexical_cfg.get("min_candidate_score"),
            _DEFAULT_LEXICAL_MIN_CANDIDATE_SCORE,
        ),
        "min_strong_title_overlap": _as_float(
            lexical_cfg.get("min_strong_title_overlap"),
            _DEFAULT_LEXICAL_MIN_STRONG_TITLE_OVERLAP,
        ),
        "min_token_length": max(1, _as_int(
            lexical_cfg.get("min_token_length"),
            _DEFAULT_MIN_TOKEN_LENGTH,
        )),
        "title_exact_boost": _as_float(
            lexical_cfg.get("title_exact_boost"),
            _DEFAULT_TITLE_EXACT_BOOST,
        ),
        "title_substring_boost": _as_float(
            lexical_cfg.get("title_substring_boost"),
            _DEFAULT_TITLE_SUBSTRING_BOOST,
        ),
        "title_token_overlap_weight": _as_float(
            lexical_cfg.get("title_token_overlap_weight"),
            _DEFAULT_TITLE_TOKEN_OVERLAP_WEIGHT,
        ),
    }


def _normalize_text(text: str) -> str:
    clean = _MARKDOWN_RE.sub("", (text or "").lower())
    clean = _NON_ALNUM_RE.sub(" ", clean)
    return _WHITESPACE_RE.sub(" ", clean).strip()


def _tokenize(text: str, min_token_length: int) -> set[str]:
    normalized = _normalize_text(text)
    return {
        token for token in normalized.split()
        if len(token) >= min_token_length and token not in _STOPWORDS
    }


def _overlap_ratio(query_tokens: set[str], target_tokens: set[str]) -> float:
    if not query_tokens or not target_tokens:
        return 0.0
    return len(query_tokens & target_tokens) / len(query_tokens)


def _paper_lexical_score(
    query: str,
    *,
    title: str,
    keywords: list[str] | None,
    strategy_cfg: dict,
) -> tuple[float, bool]:
    query_norm = _normalize_text(query)
    if not query_norm:
        return 0.0, False

    min_token_length = strategy_cfg["min_token_length"]
    query_tokens = _tokenize(query, min_token_length)
    title_norm = _normalize_text(title)
    title_tokens = _tokenize(title, min_token_length)
    keyword_norms = [_normalize_text(keyword) for keyword in (keywords or []) if keyword]
    keyword_tokens = set()
    for keyword in keyword_norms:
        keyword_tokens |= _tokenize(keyword, min_token_length)

    score = 0.0
    strong_match = False
    title_overlap = _overlap_ratio(query_tokens, title_tokens)

    if query_norm and title_norm and query_norm == title_norm:
        score += strategy_cfg["title_exact_boost"]
        strong_match = True
    elif query_norm and title_norm and (query_norm in title_norm or title_norm in query_norm):
        score += strategy_cfg["title_substring_boost"]
        strong_match = True

    score += title_overlap * strategy_cfg["title_token_overlap_weight"]
    if title_overlap >= strategy_cfg["min_strong_title_overlap"]:
        strong_match = True

    phrase_matches = sum(
        1 for keyword in keyword_norms
        if keyword and (keyword in query_norm or query_norm in keyword)
    )
    if phrase_matches:
        score += phrase_matches * strategy_cfg["keyword_phrase_boost"]
        strong_match = True

    score += _overlap_ratio(query_tokens, keyword_tokens) * strategy_cfg["keyword_token_overlap_weight"]
    return score, strong_match


def _build_filter(
    filter_title: str | None = None,
    filter_keywords: list[str] | None = None,
) -> Filter | None:
    """Build a Qdrant Filter from optional title/keyword constraints."""
    conditions = []
    if filter_title:
        conditions.append(FieldCondition(key="title", match=MatchText(text=filter_title)))
    if filter_keywords:
        conditions.append(FieldCondition(key="keywords", match=MatchAny(any=filter_keywords)))
    return Filter(must=conditions) if conditions else None


def _merge_filters(*filters: Filter | None) -> Filter | None:
    must_conditions = []
    for filt in filters:
        if filt is None:
            continue
        must_conditions.extend(filt.must or [])
    return Filter(must=must_conditions) if must_conditions else None


def _paper_id_filter(paper_id: str) -> Filter:
    return Filter(must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))])


def _query_points(
    client,
    *,
    query_vec,
    sparse_vec: SparseVector | None,
    limit: int,
    use_hybrid: bool,
    qfilter: Filter | None,
):
    if use_hybrid and sparse_vec is not None:
        results = client.query_points(
            _COLLECTION,
            prefetch=[
                Prefetch(query=query_vec, using="dense", limit=limit, filter=qfilter),
                Prefetch(query=sparse_vec, using="sparse", limit=limit, filter=qfilter),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
            query_filter=qfilter,
        )
    else:
        results = client.query_points(
            _COLLECTION,
            query=query_vec,
            limit=limit,
            with_payload=True,
            query_filter=qfilter,
        )
    return results.points


def _point_to_item(point) -> dict:
    payload = point.payload
    return {
        "paper_id": payload.get("paper_id") or payload.get("content_hash") or "",
        "title": payload["title"],
        "text": payload["document"],
        "parent_text": payload.get("parent_text", payload["document"]),
        "score": point.score,
        "source_path": to_absolute(payload["source_path"]),
        "chunk_index": payload["chunk_index"],
        "markdown_path": to_absolute(payload.get("markdown_path", "")) if payload.get("markdown_path") else "",
        "pdf_path": to_absolute(payload.get("pdf_path", "")) if payload.get("pdf_path") else "",
        "keywords": payload.get("keywords", []),
        "source_url": payload.get("source_url", ""),
        "ingested_at": payload.get("ingested_at", ""),
    }


def _paper_matches_filters(
    paper: dict,
    *,
    filter_title: str | None,
    filter_keywords: list[str] | None,
) -> bool:
    if filter_title and _normalize_text(filter_title) not in _normalize_text(paper.get("title", "")):
        return False
    if filter_keywords:
        normalized_keywords = {_normalize_text(keyword) for keyword in (paper.get("keywords") or []) if keyword}
        wanted = {_normalize_text(keyword) for keyword in filter_keywords if keyword}
        if not (normalized_keywords & wanted):
            return False
    return True


def _collect_lexical_candidates(
    *,
    query: str,
    strategy_cfg: dict,
    existing_items: list[dict],
    filter_title: str | None,
    filter_keywords: list[str] | None,
) -> list[dict]:
    if strategy_cfg["mode"] != "semantic_with_lexical_fallback" or strategy_cfg["max_candidates"] <= 0:
        return []

    from .ingest import list_papers as _list_papers

    existing_keys = {item.get("paper_id") or item.get("source_path", "") for item in existing_items}
    candidates: list[tuple[float, dict]] = []
    for paper in _list_papers():
        paper_key = paper.get("paper_id") or paper.get("source_path", "")
        if not paper_key or paper_key in existing_keys or paper.get("chunks_count", 0) <= 0:
            continue
        if not _paper_matches_filters(paper, filter_title=filter_title, filter_keywords=filter_keywords):
            continue
        lexical_score, strong_match = _paper_lexical_score(
            query,
            title=paper.get("title", ""),
            keywords=paper.get("keywords", []),
            strategy_cfg=strategy_cfg,
        )
        if strong_match and lexical_score >= strategy_cfg["min_candidate_score"]:
            candidates.append((lexical_score, paper))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [paper for _, paper in candidates[:strategy_cfg["max_candidates"]]]


def _apply_lexical_boost(items: list[dict], *, query: str, strategy_cfg: dict) -> list[dict]:
    if strategy_cfg["mode"] == "semantic_only":
        return items
    for item in items:
        lexical_score, _ = _paper_lexical_score(
            query,
            title=item.get("title", ""),
            keywords=item.get("keywords", []),
            strategy_cfg=strategy_cfg,
        )
        item["score"] += lexical_score
    return sorted(items, key=lambda item: item["score"], reverse=True)


def _deduplicate_items(items: list[dict]) -> list[dict]:
    seen = {}
    for item in items:
        key = item["paper_id"] or item["source_path"]
        if key not in seen or item["score"] > seen[key]["score"]:
            seen[key] = item
    return sorted(seen.values(), key=lambda item: item["score"], reverse=True)


def search(
    query: str,
    top_k: int | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    deduplicate: bool = False,
    filter_title: str | None = None,
    filter_keywords: list[str] | None = None,
) -> list[dict]:
    """Hybrid search: semantic retrieval with configurable lexical boosting/fallback."""
    cfg = load_config()
    top_k = top_k or cfg["retrieval"]["top_k"]
    strategy_cfg = _get_query_strategy_config(cfg)

    client = _get_client()
    if not client.collection_exists(_COLLECTION):
        return []

    query_vec = embed([query], provider=embedding_provider, model=embedding_model)[0]
    fetch_k = top_k * _FETCH_K_MULTIPLIER
    use_hybrid = _has_sparse_vectors(client)
    sparse_vec = None
    if use_hybrid:
        sparse_result = sparse_embed([query])[0]
        sparse_vec = SparseVector(indices=sparse_result[0], values=sparse_result[1])

    qfilter = _build_filter(filter_title, filter_keywords)
    points = _query_points(
        client,
        query_vec=query_vec,
        sparse_vec=sparse_vec,
        limit=fetch_k,
        use_hybrid=use_hybrid,
        qfilter=qfilter,
    )
    items = [_point_to_item(point) for point in points]

    for paper in _collect_lexical_candidates(
        query=query,
        strategy_cfg=strategy_cfg,
        existing_items=items,
        filter_title=filter_title,
        filter_keywords=filter_keywords,
    ):
        candidate_points = _query_points(
            client,
            query_vec=query_vec,
            sparse_vec=sparse_vec,
            limit=1,
            use_hybrid=use_hybrid,
            qfilter=_merge_filters(qfilter, _paper_id_filter(paper["paper_id"])),
        )
        if candidate_points:
            items.append(_point_to_item(candidate_points[0]))

    reranker_cfg = cfg["retrieval"].get("reranker", {})
    if reranker_cfg.get("enabled", False):
        from .reranker import rerank
        items = rerank(query, items, top_k=len(items), model=reranker_cfg.get("model"))

    items = _apply_lexical_boost(items, query=query, strategy_cfg=strategy_cfg)

    if deduplicate:
        items = _deduplicate_items(items)

    min_score = cfg["retrieval"].get("min_score", 0)
    if min_score > 0:
        items = [item for item in items if item["score"] >= min_score]

    return items[:top_k]
