from __future__ import annotations

from types import SimpleNamespace

from core import ingest, retriever


def _make_config(mode: str) -> dict:
    return {
        "retrieval": {
            "top_k": 5,
            "min_score": 0,
            "reranker": {"enabled": False},
            "query_strategy": {
                "mode": mode,
                "lexical": {
                    "max_candidates": 3,
                    "min_candidate_score": 0.35,
                    "min_strong_title_overlap": 0.6,
                    "min_token_length": 2,
                    "title_exact_boost": 1.0,
                    "title_substring_boost": 0.35,
                    "title_token_overlap_weight": 0.35,
                    "keyword_phrase_boost": 0.2,
                    "keyword_token_overlap_weight": 0.15,
                },
            },
        },
    }


def _make_point(*, paper_id: str, title: str, score: float, keywords: list[str] | None = None):
    payload = {
        "paper_id": paper_id,
        "title": title,
        "document": f"Excerpt for {title}",
        "parent_text": f"Parent excerpt for {title}",
        "source_path": f"data/papers/{paper_id}/paper.pdf",
        "chunk_index": 0,
        "markdown_path": f"data/papers/{paper_id}/paper.md",
        "pdf_path": f"data/papers/{paper_id}/paper.pdf",
        "keywords": keywords or [],
        "source_url": "",
        "ingested_at": "",
    }
    return SimpleNamespace(payload=payload, score=score)


class _FakeClient:
    def __init__(self, *, main_points: list, fallback_points: dict[str, object] | None = None):
        self.calls: list[dict] = []
        self.main_points = main_points
        self.fallback_points = fallback_points or {}

    def collection_exists(self, _: str) -> bool:
        return True

    def query_points(self, *args, **kwargs):
        self.calls.append(kwargs)
        query_filter = kwargs.get("query_filter")
        paper_id = None
        if query_filter is not None:
            for condition in query_filter.must or []:
                if getattr(condition, "key", None) != "paper_id":
                    continue
                paper_id = getattr(getattr(condition, "match", None), "value", None)
                break
        if paper_id and paper_id in self.fallback_points:
            return SimpleNamespace(points=[self.fallback_points[paper_id]])
        return SimpleNamespace(points=self.main_points)


def test_search_lexical_boost_promotes_exact_title(monkeypatch):
    query = "NeRFactor: Neural Factorization of Shape and Reflectance Under an Unknown Illumination"
    nerf = _make_point(
        paper_id="nerf",
        title="NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
        score=0.5,
        keywords=["neural radiance fields", "view synthesis"],
    )
    nerfactor = _make_point(
        paper_id="nerfactor",
        title=query,
        score=0.5,
        keywords=["brdf estimation", "unknown illumination"],
    )
    client = _FakeClient(main_points=[nerf, nerfactor])

    monkeypatch.setattr(retriever, "load_config", lambda: _make_config("semantic_with_lexical_boost"))
    monkeypatch.setattr(retriever, "_get_client", lambda: client)
    monkeypatch.setattr(retriever, "_has_sparse_vectors", lambda _: False)
    monkeypatch.setattr(retriever, "embed", lambda *args, **kwargs: [[0.1, 0.2]])

    results = retriever.search(query, top_k=2, deduplicate=True)

    assert [result["title"] for result in results] == [
        query,
        "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
    ]


def test_search_lexical_fallback_adds_missing_title_match(monkeypatch):
    query = "NeRFactor: Neural Factorization of Shape and Reflectance Under an Unknown Illumination"
    nerf = _make_point(
        paper_id="nerf",
        title="NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
        score=0.7,
        keywords=["neural radiance fields", "view synthesis"],
    )
    nerfactor = _make_point(
        paper_id="nerfactor",
        title=query,
        score=0.4,
        keywords=["brdf estimation", "unknown illumination"],
    )
    client = _FakeClient(main_points=[nerf], fallback_points={"nerfactor": nerfactor})

    monkeypatch.setattr(retriever, "load_config", lambda: _make_config("semantic_with_lexical_fallback"))
    monkeypatch.setattr(retriever, "_get_client", lambda: client)
    monkeypatch.setattr(retriever, "_has_sparse_vectors", lambda _: False)
    monkeypatch.setattr(retriever, "embed", lambda *args, **kwargs: [[0.1, 0.2]])
    monkeypatch.setattr(
        ingest,
        "list_papers",
        lambda: [
            {
                "paper_id": "nerfactor",
                "title": query,
                "chunks_count": 10,
                "source_path": "data/papers/nerfactor/paper.pdf",
                "keywords": ["brdf estimation", "unknown illumination"],
            },
        ],
    )

    results = retriever.search(query, top_k=2, deduplicate=True)

    assert [result["title"] for result in results] == [
        query,
        "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
    ]
    assert len(client.calls) == 2


def test_search_semantic_only_keeps_original_order(monkeypatch):
    query = "NeRFactor: Neural Factorization of Shape and Reflectance Under an Unknown Illumination"
    nerf = _make_point(
        paper_id="nerf",
        title="NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
        score=0.5,
        keywords=["neural radiance fields", "view synthesis"],
    )
    nerfactor = _make_point(
        paper_id="nerfactor",
        title=query,
        score=0.5,
        keywords=["brdf estimation", "unknown illumination"],
    )
    client = _FakeClient(main_points=[nerf, nerfactor])

    monkeypatch.setattr(retriever, "load_config", lambda: _make_config("semantic_only"))
    monkeypatch.setattr(retriever, "_get_client", lambda: client)
    monkeypatch.setattr(retriever, "_has_sparse_vectors", lambda _: False)
    monkeypatch.setattr(retriever, "embed", lambda *args, **kwargs: [[0.1, 0.2]])

    results = retriever.search(query, top_k=2, deduplicate=True)

    assert [result["title"] for result in results] == [
        "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
        query,
    ]
