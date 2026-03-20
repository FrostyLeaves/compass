"""Qdrant client management and low-level collection operations."""

from __future__ import annotations

import atexit
import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    SparseVectorParams,
    TextIndexParams,
    TokenizerType,
    VectorParams,
)

from .config import get_qdrant_config

logger = logging.getLogger("compass")

_qdrant_client: QdrantClient | None = None
_COLLECTION = "papers"
_SCROLL_LIMIT = 256


def _get_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    qcfg = get_qdrant_config()
    try:
        _qdrant_client = QdrantClient(**qcfg)
    except RuntimeError as e:
        if "already accessed" in str(e):
            path = qcfg.get("path", "data/qdrant")
            raise SystemExit(
                f"Error: Qdrant storage '{path}' is locked by another process "
                f"(e.g. a running web server or another CLI instance).\n"
                f"Stop the other process or use a different storage path."
            ) from e
        raise
    atexit.register(_close_client)
    return _qdrant_client


def _try_get_client() -> QdrantClient | None:
    try:
        return _get_client()
    except SystemExit:
        return None


def _close_client():
    global _qdrant_client
    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
        except Exception:
            pass
        _qdrant_client = None


def _ensure_payload_indexes(client: QdrantClient) -> None:
    text_index = TextIndexParams(type="text", tokenizer=TokenizerType.WORD, lowercase=True)
    index_specs: list[tuple[str, Any]] = [
        ("document", text_index),
        ("title", text_index),
        ("keywords", PayloadSchemaType.KEYWORD),
        ("paper_id", PayloadSchemaType.KEYWORD),
        ("source_urls", PayloadSchemaType.KEYWORD),
    ]
    for field_name, field_schema in index_specs:
        try:
            client.create_payload_index(_COLLECTION, field_name=field_name, field_schema=field_schema)
        except (RuntimeError, ValueError):
            pass  # index already exists


def _ensure_collection(dim: int) -> None:
    """Create the papers collection if needed and ensure indexes exist."""
    client = _get_client()
    if not client.collection_exists(_COLLECTION):
        client.create_collection(
            _COLLECTION,
            vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
    _ensure_payload_indexes(client)


def _scroll_by_field(
    client: QdrantClient,
    key: str,
    value: str,
    *,
    with_vectors: bool = False,
    max_points: int | None = None,
) -> list:
    points = []
    offset = None
    filt = Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))])
    while True:
        limit = _SCROLL_LIMIT if max_points is None else min(_SCROLL_LIMIT, max_points - len(points))
        if limit <= 0:
            break
        batch, offset = client.scroll(
            _COLLECTION,
            scroll_filter=filt,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        points.extend(batch)
        if offset is None or (max_points is not None and len(points) >= max_points):
            break
    return points


def _paper_points(client: QdrantClient, paper_id: str, *, with_vectors: bool = False) -> list:
    if not paper_id or not client.collection_exists(_COLLECTION):
        return []
    return _scroll_by_field(client, "paper_id", paper_id, with_vectors=with_vectors)


def _first_point_by_field(
    client: QdrantClient,
    key: str,
    value: str,
    *,
    with_vectors: bool = False,
):
    points = _scroll_by_field(client, key, value, with_vectors=with_vectors, max_points=1)
    return points[0] if points else None
