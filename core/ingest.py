"""PDF parsing, chunking, vectorization, and ingestion."""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client.models import (
    PointStruct,
    SparseVector,
)

from .config import get_papers_dir, load_config, to_absolute, to_relative
from .converter import convert_pdf
from .embedder import embed, sparse_embed
from .generator import detect_language, extract_keywords, translate
from .vectorstore import (
    _get_client, _try_get_client, _COLLECTION,
    _ensure_collection, _paper_points, _first_point_by_field,
)
from .paper import (
    _PAPER_MD_NAME, _PAPER_PDF_NAME, _PAPER_META_NAME,
    _paper_dir, _paper_markdown_path, _paper_pdf_path, _paper_meta_path,
    _find_main_markdown_path, _find_main_pdf_path, _paper_files_exist,
    _normalize_url, _normalize_source_urls, _payload_source_urls, _payload_paper_id,
    _relative_path, _absolute_if_present,
    _paper_metadata_payload, _write_paper_metadata, _write_paper_metadata_from_payload,
    _read_paper_metadata, _build_paper_result, _paper_result_from_metadata,
    _save_markdown, _translation_code_from_file,
)
from .text import _chunk_text, _assign_parent_text, _extract_title, _cleanup_markdown_with_llm

logger = logging.getLogger("compass")

_DOWNLOAD_CHUNK_SIZE = 65536
_LEGACY_CHARS_PER_TOKEN = 4
_DEFAULT_CHUNK_TOKEN_SIZE = 220
_DEFAULT_CHUNK_TOKEN_OVERLAP = 40
_DEFAULT_PARENT_CHUNK_TOKEN_SIZE = 600
_MIN_TOKEN_BUDGET = 48


def _hash_file(path: str | Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _doc_id(paper_id: str, chunk_index: int) -> str:
    return f"{paper_id}__chunk_{chunk_index}"


def _point_id(paper_id: str, chunk_index: int) -> str:
    return hashlib.md5(_doc_id(paper_id, chunk_index).encode("utf-8")).hexdigest()


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_token_budget(
    cfg: dict,
    *,
    token_key: str,
    legacy_key: str,
    default: int,
    allow_zero: bool = False,
    min_value: int = _MIN_TOKEN_BUDGET,
) -> int:
    token_value = cfg.get(token_key)
    if token_value is not None:
        parsed = _as_int(token_value, default)
        if allow_zero and parsed <= 0:
            return 0
        return max(min_value, parsed)

    legacy_value = cfg.get(legacy_key)
    if legacy_value is not None:
        parsed = _as_int(legacy_value, 0)
        if allow_zero and parsed <= 0:
            return 0
        if parsed > 0:
            return max(min_value, round(parsed / _LEGACY_CHARS_PER_TOKEN))

    if allow_zero and default <= 0:
        return 0
    return max(min_value, default)


def _resolve_chunking_config(cfg: dict) -> tuple[int, int, int]:
    return (
        _resolve_token_budget(
            cfg,
            token_key="chunk_token_size",
            legacy_key="chunk_size",
            default=_DEFAULT_CHUNK_TOKEN_SIZE,
        ),
        _resolve_token_budget(
            cfg,
            token_key="chunk_token_overlap",
            legacy_key="chunk_overlap",
            default=_DEFAULT_CHUNK_TOKEN_OVERLAP,
            allow_zero=True,
            min_value=0,
        ),
        _resolve_token_budget(
            cfg,
            token_key="parent_chunk_token_size",
            legacy_key="parent_chunk_size",
            default=_DEFAULT_PARENT_CHUNK_TOKEN_SIZE,
            allow_zero=True,
        ),
    )


def _refresh_paper_metadata(paper_id: str) -> dict | None:
    paper_dir = _paper_dir(paper_id)
    md_path = _find_main_markdown_path(paper_dir)
    pdf_path = _find_main_pdf_path(paper_dir)
    if not md_path or not pdf_path:
        return None

    text = md_path.read_text(encoding="utf-8")
    title = _extract_title(text, str(pdf_path))
    chunks_count = 0
    keywords: list[str] = []
    source_urls: list[str] = []
    ingested_at = ""

    metadata_payload = _paper_metadata_payload(
        paper_id=paper_id,
        title=title,
        chunks_count=chunks_count,
        markdown_path=md_path,
        pdf_path=pdf_path,
        keywords=keywords,
        source_urls=source_urls,
        ingested_at=ingested_at,
    )

    client = _try_get_client()
    if client is None:
        return _paper_result_from_metadata(metadata_payload)

    if client.collection_exists(_COLLECTION):
        points = _paper_points(client, paper_id)
        if points:
            payload = points[0].payload or {}
            chunks_count = len(points)
            keywords = list(payload.get("keywords") or [])
            source_urls = _payload_source_urls(payload)
            ingested_at = str(payload.get("ingested_at", ""))

    metadata = _write_paper_metadata(
        paper_id=paper_id,
        title=title,
        chunks_count=chunks_count,
        markdown_path=md_path,
        pdf_path=pdf_path,
        keywords=keywords,
        source_urls=source_urls,
        ingested_at=ingested_at,
    )
    return _paper_result_from_metadata(metadata)


def get_paper(paper_id: str) -> dict | None:
    """Load a paper summary by paper_id without scanning the chunk collection."""
    if not paper_id:
        return None
    metadata = _read_paper_metadata(paper_id)
    if metadata and _paper_files_exist(paper_id):
        return _paper_result_from_metadata(metadata)
    return _refresh_paper_metadata(paper_id)


def _update_existing_paper_sources(client, paper_id: str, source_url: str | None) -> dict | None:
    points = _paper_points(client, paper_id, with_vectors=True)
    if not points:
        return None
    merged_urls = _normalize_source_urls(source_url, *(_payload_source_urls(point.payload) for point in points))
    updated = []
    for point in points:
        payload = dict(point.payload)
        payload["paper_id"] = paper_id
        payload["source_urls"] = merged_urls
        payload["source_url"] = merged_urls[0] if merged_urls else ""
        updated.append(PointStruct(id=point.id, vector=point.vector, payload=payload))
    client.upsert(_COLLECTION, updated)
    metadata = _write_paper_metadata_from_payload(updated[0].payload, paper_id=paper_id, chunks_count=len(updated))
    return _paper_result_from_metadata(metadata)


def _translate_markdown(md_path: str, text: str, progress_callback=None) -> None:
    """If i18n is enabled, detect language and translate markdown to configured languages."""
    cfg = load_config()
    i18n = cfg.get("i18n", {})
    if not i18n.get("enabled", False):
        return

    languages = i18n.get("languages", [])
    if not languages:
        return

    def _progress(stage: str, current: int, total: int) -> None:
        if progress_callback:
            progress_callback(stage, current, total)

    md = Path(to_absolute(md_path))
    stem = md.stem
    paper_dir = md.parent

    _progress("detecting_lang", 0, 1)
    src_lang = detect_language(text)
    _progress("detecting_lang", 1, 1)
    logger.info("Detected language: %s", src_lang)

    src_lang_file = paper_dir / f"{stem}.{src_lang}.md"
    if not src_lang_file.exists():
        src_lang_file.write_text(text, encoding="utf-8")

    targets = [lang for lang in languages if lang["code"] != src_lang and not (paper_dir / f"{stem}.{lang['code']}.md").exists()]
    for i, lang in enumerate(targets):
        code = lang["code"]
        name = lang["name"]
        _progress("translating", i, len(targets))
        logger.info("Translating to %s (%s)...", name, code)
        translated = translate(text, name)
        target_file = paper_dir / f"{stem}.{code}.md"
        target_file.write_text(translated, encoding="utf-8")
        logger.info("Saved %s", target_file.name)
    if targets:
        _progress("translating", len(targets), len(targets))


def _index_paper(
    *,
    paper_id: str,
    title: str,
    text: str,
    markdown_path: str,
    pdf_path: str,
    keywords: list[str],
    source_urls: list[str],
    ingested_at: str,
    embedding_provider: str | None,
    embedding_model: str | None,
    progress_callback=None,
) -> int:
    cfg = load_config()["ingest"]
    chunk_size, chunk_overlap, parent_chunk_size = _resolve_chunking_config(cfg)

    def _progress(stage: str, current: int, total: int) -> None:
        if progress_callback:
            progress_callback(stage, current, total)

    chunks = _chunk_text(text, chunk_size, chunk_overlap)
    _assign_parent_text(chunks, parent_chunk_size)

    client = _get_client()
    existing = _paper_points(client, paper_id)
    if existing:
        client.delete(_COLLECTION, points_selector=[point.id for point in existing])

    if not chunks:
        _write_paper_metadata(
            paper_id=paper_id,
            title=title,
            chunks_count=0,
            markdown_path=markdown_path,
            pdf_path=pdf_path,
            keywords=keywords,
            source_urls=source_urls,
            ingested_at=ingested_at,
        )
        return 0

    def _contextual_text(chunk: dict) -> str:
        heading_path = list(chunk.get("heading_path") or [])
        if heading_path and heading_path[0].strip() == title.strip():
            heading_path = heading_path[1:]
        prefix = " > ".join([title, *heading_path]) if heading_path else title
        return f"{prefix}: {chunk['text']}"

    embed_texts = [_contextual_text(chunk) for chunk in chunks]
    raw_texts = [chunk["text"] for chunk in chunks]
    vectors = embed(
        embed_texts,
        provider=embedding_provider,
        model=embedding_model,
        progress_callback=lambda cur, tot: _progress("embedding", cur, tot),
    )
    sparse_vectors = sparse_embed(embed_texts)

    _progress("storing", 0, 1)
    _ensure_collection(dim=len(vectors[0]))

    source_urls = _normalize_source_urls(source_urls)
    primary_url = source_urls[0] if source_urls else ""
    rel_md = _relative_path(markdown_path)
    rel_pdf = _relative_path(pdf_path)

    points = [
        PointStruct(
            id=_point_id(paper_id, chunk["chunk_index"]),
            vector={
                "dense": vec,
                "sparse": SparseVector(indices=sp[0], values=sp[1]),
            },
            payload={
                "doc_id": _doc_id(paper_id, chunk["chunk_index"]),
                "paper_id": paper_id,
                "document": doc,
                "parent_text": chunk.get("parent_text", doc),
                "title": title,
                "source_path": rel_pdf,
                "chunk_index": chunk["chunk_index"],
                "section_heading": chunk.get("section_heading", ""),
                "heading_path": chunk.get("heading_path", []),
                "heading_level": chunk.get("heading_level", 0),
                "heading_path_text": chunk.get("heading_path_text", ""),
                "token_count": chunk.get("token_count", 0),
                "parent_start_chunk_index": chunk.get("parent_start_chunk_index", chunk["chunk_index"]),
                "parent_end_chunk_index": chunk.get("parent_end_chunk_index", chunk["chunk_index"]),
                "parent_token_count": chunk.get("parent_token_count", chunk.get("token_count", 0)),
                "content_types": chunk.get("content_types", []),
                "dominant_content_type": chunk.get("dominant_content_type", ""),
                "markdown_path": rel_md,
                "pdf_path": rel_pdf,
                "source_url": primary_url,
                "source_urls": source_urls,
                "keywords": keywords,
                "ingested_at": ingested_at,
            },
        )
        for vec, doc, sp, chunk in zip(vectors, raw_texts, sparse_vectors, chunks)
    ]

    client.upsert(_COLLECTION, points)
    _write_paper_metadata(
        paper_id=paper_id,
        title=title,
        chunks_count=len(points),
        markdown_path=rel_md,
        pdf_path=rel_pdf,
        keywords=keywords,
        source_urls=source_urls,
        ingested_at=ingested_at,
    )
    _progress("storing", 1, 1)
    return len(points)


def ingest_pdf(
    pdf_path: str | Path,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    progress_callback=None,
    source_url: str | None = None,
) -> dict:
    """Import a single PDF into Compass using content hash as the paper id."""
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    def _progress(stage: str, current: int, total: int) -> None:
        if progress_callback:
            progress_callback(stage, current, total)

    _progress("hashing", 0, 1)
    full_cfg = load_config()
    conv_cfg = full_cfg.get("converter", full_cfg.get("marker", {}))
    cleanup_markdown = conv_cfg.get("marker", conv_cfg).get("cleanup_markdown", False)
    client = _get_client()

    paper_id = _hash_file(pdf_path)
    _progress("hashing", 1, 1)

    canonical_pdf = _paper_pdf_path(paper_id)
    canonical_md = _paper_markdown_path(paper_id)
    normalized_url = _normalize_url(source_url)

    _progress("dedup", 0, 1)
    existing_points = _paper_points(client, paper_id, with_vectors=True)
    _progress("dedup", 1, 1)
    if existing_points and _paper_files_exist(paper_id):
        _progress("skipped", 0, 1)
        updated = _update_existing_paper_sources(client, paper_id, normalized_url)
        _progress("skipped", 1, 1)
        if updated is not None:
            return updated

    if canonical_pdf.exists() and canonical_md.exists():
        text = canonical_md.read_text(encoding="utf-8")
        title = _extract_title(text, str(canonical_pdf))
        source_urls = _normalize_source_urls(normalized_url, *(_payload_source_urls(point.payload) for point in existing_points))
        keywords = list(existing_points[0].payload.get("keywords", [])) if existing_points else []
        ingested_at = str(existing_points[0].payload.get("ingested_at", "")) if existing_points else ""

        if not existing_points:
            _progress("keywords", 0, 1)
            keywords = extract_keywords(text)
            _progress("keywords", 1, 1)
            logger.info("Extracted keywords: %s", keywords)
            _translate_markdown(to_relative(canonical_md), text, progress_callback=progress_callback)
            ingested_at = datetime.now(timezone.utc).isoformat()
            chunks_count = _index_paper(
                paper_id=paper_id,
                title=title,
                text=text,
                markdown_path=to_relative(canonical_md),
                pdf_path=to_relative(canonical_pdf),
                keywords=keywords,
                source_urls=source_urls,
                ingested_at=ingested_at,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                progress_callback=progress_callback,
            )
        else:
            chunks_count = len(existing_points)

        return {
            "paper_id": paper_id,
            "title": title,
            "chunks_count": chunks_count,
            "path": str(canonical_pdf),
            "markdown_path": str(canonical_md),
            "pdf_path": str(canonical_pdf),
            "source_url": source_urls[0] if source_urls else "",
            "keywords": keywords,
            "ingested_at": ingested_at,
        }

    _progress("parsing", 0, 1)
    text, images = convert_pdf(pdf_path)
    _progress("parsing", 1, 1)

    if cleanup_markdown:
        _progress("cleaning", 0, 1)
        text = _cleanup_markdown_with_llm(text)
        _progress("cleaning", 1, 1)

    markdown_path, pdf_copy_path = _save_markdown(pdf_path, paper_id, text, images)
    title = _extract_title(text, str(pdf_path))

    _progress("keywords", 0, 1)
    keywords = extract_keywords(text)
    _progress("keywords", 1, 1)
    logger.info("Extracted keywords: %s", keywords)

    _translate_markdown(markdown_path, text, progress_callback=progress_callback)

    source_urls = _normalize_source_urls(normalized_url, *(_payload_source_urls(point.payload) for point in existing_points))
    ingested_at = datetime.now(timezone.utc).isoformat()
    chunks_count = _index_paper(
        paper_id=paper_id,
        title=title,
        text=text,
        markdown_path=markdown_path,
        pdf_path=pdf_copy_path,
        keywords=keywords,
        source_urls=source_urls,
        ingested_at=ingested_at,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        progress_callback=progress_callback,
    )

    return {
        "paper_id": paper_id,
        "title": title,
        "chunks_count": chunks_count,
        "path": str(_paper_pdf_path(paper_id)),
        "markdown_path": str(_paper_markdown_path(paper_id)),
        "pdf_path": str(_paper_pdf_path(paper_id)),
        "source_url": source_urls[0] if source_urls else "",
        "keywords": keywords,
        "ingested_at": ingested_at,
    }


def ingest_directory(dir_path: str | Path, recursive: bool = True, **kwargs) -> list[dict]:
    """Batch import all PDFs in a directory."""
    dir_path = Path(dir_path)
    pattern = "**/*.pdf" if recursive else "*.pdf"
    results = []
    for pdf in sorted(dir_path.glob(pattern)):
        try:
            result = ingest_pdf(pdf, **kwargs)
            results.append(result)
            logger.info("OK: %s (%d chunks)", result["title"], result["chunks_count"])
        except (FileNotFoundError, RuntimeError, ValueError, OSError, KeyError) as e:
            results.append({"title": pdf.name, "chunks_count": 0, "path": str(pdf), "error": str(e)})
            logger.error("FAIL: %s: %s", pdf.name, e)
    return results


def reindex_paper(
    paper_id: str,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    progress_callback=None,
    source_urls: list[str] | None = None,
    ingested_at: str | None = None,
) -> dict:
    """Re-embed a single paper from its saved markdown."""
    paper_dir = _paper_dir(paper_id)
    md_file = _find_main_markdown_path(paper_dir)
    pdf_file = _find_main_pdf_path(paper_dir)
    if not md_file:
        raise FileNotFoundError(f"Markdown not found for paper_id: {paper_id}")
    if not pdf_file:
        raise FileNotFoundError(f"PDF not found for paper_id: {paper_id}")

    text = md_file.read_text(encoding="utf-8")
    title = _extract_title(text, str(pdf_file or md_file))
    rel_md_path = to_relative(md_file)
    rel_pdf_path = to_relative(pdf_file) if pdf_file else ""

    client = _get_client()
    existing_points = _paper_points(client, paper_id, with_vectors=True)
    source_urls = _normalize_source_urls(source_urls, *(_payload_source_urls(point.payload) for point in existing_points))

    def _progress(stage: str, current: int, total: int) -> None:
        if progress_callback:
            progress_callback(stage, current, total)

    _progress("keywords", 0, 1)
    keywords = extract_keywords(text)
    _progress("keywords", 1, 1)
    logger.info("Extracted keywords for %s: %s", title, keywords)

    if ingested_at is None:
        ingested_at = datetime.now(timezone.utc).isoformat()

    chunks_count = _index_paper(
        paper_id=paper_id,
        title=title,
        text=text,
        markdown_path=rel_md_path,
        pdf_path=rel_pdf_path,
        keywords=keywords,
        source_urls=source_urls,
        ingested_at=ingested_at,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        progress_callback=progress_callback,
    )

    return {
        "paper_id": paper_id,
        "title": title,
        "chunks_count": chunks_count,
        "keywords": keywords,
        "path": _absolute_if_present(rel_pdf_path),
        "markdown_path": _absolute_if_present(rel_md_path),
        "pdf_path": _absolute_if_present(rel_pdf_path),
        "source_url": source_urls[0] if source_urls else "",
        "ingested_at": ingested_at,
    }


def reindex_all(
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    progress_callback=None,
) -> list[dict]:
    """Re-embed and update all papers from their saved markdown."""
    papers_dir = get_papers_dir()
    if not papers_dir.exists():
        return []

    results = []
    for paper_dir in sorted(papers_dir.iterdir()):
        if not paper_dir.is_dir() or not _find_main_markdown_path(paper_dir):
            continue
        try:
            result = reindex_paper(
                paper_dir.name,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                progress_callback=progress_callback,
            )
            results.append(result)
            logger.info("Reindexed: %s (%d chunks)", result["title"], result["chunks_count"])
        except (FileNotFoundError, RuntimeError, ValueError, OSError, KeyError) as e:
            results.append({"paper_id": paper_dir.name, "title": paper_dir.name, "chunks_count": 0, "error": str(e)})
            logger.error("Reindex failed: %s: %s", paper_dir.name, e)
    return results


def list_papers() -> list[dict]:
    """List all ingested papers. Paths are returned as absolute."""
    papers_dir = get_papers_dir()
    if not papers_dir.exists():
        return []

    papers: list[dict] = []
    for paper_dir in sorted(papers_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        paper = get_paper(paper_dir.name)
        if paper is not None:
            papers.append(paper)
    return papers


def _find_paper_by_source_url(url: str) -> dict | None:
    normalized = _normalize_url(url)
    if not normalized:
        return None
    client = _try_get_client()
    if client is not None and client.collection_exists(_COLLECTION):
        point = _first_point_by_field(client, "source_urls", normalized, with_vectors=False)
        if point is not None:
            paper_id = _payload_paper_id(point.payload or {})
            paper = get_paper(paper_id)
            if paper is not None:
                return paper

    papers_dir = get_papers_dir()
    if not papers_dir.exists():
        return None

    for paper_dir in sorted(papers_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        metadata = _read_paper_metadata(paper_dir.name)
        if metadata is None:
            paper = get_paper(paper_dir.name)
            if paper is None:
                continue
            metadata = _read_paper_metadata(paper_dir.name)
        if metadata is not None and normalized in _payload_source_urls(metadata):
            return _paper_result_from_metadata(metadata)
    return None


def ingest_url(
    url: str,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    progress_callback=None,
) -> dict:
    """Download a PDF from a URL and ingest it using content-hash deduplication."""
    import tempfile
    import urllib.parse
    import time

    import httpx

    def _progress(stage: str, current: int, total: int) -> None:
        if progress_callback:
            progress_callback(stage, current, total)

    normalized = _normalize_url(url)
    _progress("dedup", 0, 1)
    existing = _find_paper_by_source_url(normalized)
    _progress("dedup", 1, 1)
    if existing:
        existing_pdf = Path(existing.get("pdf_path", ""))
        if existing.get("chunks_count", 0) > 0 and existing_pdf.exists():
            _progress("skipped", 0, 1)
            _progress("skipped", 1, 1)
            logger.info("URL already ingested: %s, skipping download", normalized)
            return existing
        if existing_pdf.exists():
            logger.info("URL already downloaded: %s, skipping download", normalized)
            return ingest_pdf(
                existing_pdf,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                progress_callback=progress_callback,
                source_url=normalized,
            )

    parsed = urllib.parse.urlparse(normalized)
    url_stem = Path(parsed.path).stem or "paper"
    url_stem = re.sub(r"[^\w\-]", "_", url_stem)[:80] or "paper"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_pdf = Path(tmpdir) / f"{url_stem}.pdf"
        timeout = httpx.Timeout(connect=30, read=300, write=30, pool=30)
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            _progress("downloading", 0, 1)
            try:
                with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                    with client.stream("GET", normalized) as resp:
                        resp.raise_for_status()
                        total_bytes = int(resp.headers.get("content-length", 0))
                        downloaded = 0
                        with tmp_pdf.open("wb") as fh:
                            for chunk in resp.iter_bytes(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                                fh.write(chunk)
                                downloaded += len(chunk)
                                if total_bytes:
                                    _progress("downloading", downloaded, total_bytes)
                break
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Download failed ({e.response.status_code}): {normalized}") from e
            except (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError) as e:
                if attempt == max_attempts:
                    raise RuntimeError(f"Download failed: {normalized} - {e}") from e
                logger.warning(
                    "Retrying PDF download after transient error (%s/%s): %s",
                    attempt,
                    max_attempts,
                    e,
                )
                time.sleep(min(attempt, 3))
        _progress("downloading", 1, 1)

        return ingest_pdf(
            tmp_pdf,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            progress_callback=progress_callback,
            source_url=normalized,
        )


def audit_translations(progress_callback=None) -> dict:
    """Audit translation files: add missing translations, remove stale ones."""
    cfg = load_config()
    i18n = cfg.get("i18n", {})
    enabled = i18n.get("enabled", False)
    languages = i18n.get("languages", []) if enabled else []
    lang_codes = {lang["code"] for lang in languages}

    papers_dir = get_papers_dir()
    stats = {"papers_scanned": 0, "translations_added": 0, "translations_removed": 0, "titles_fixed": 0}

    if not papers_dir.exists():
        return stats

    client = _get_client()
    collection_exists = client.collection_exists(_COLLECTION)

    for paper_dir in sorted(papers_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        main_md = _find_main_markdown_path(paper_dir)
        if not main_md:
            continue

        stats["papers_scanned"] += 1
        if progress_callback:
            progress_callback(paper_dir.name)

        main_pdf = _find_main_pdf_path(paper_dir)
        text = main_md.read_text(encoding="utf-8")
        correct_title = _extract_title(text, str(main_pdf or main_md))
        chunks = _paper_points(client, paper_dir.name, with_vectors=True) if collection_exists else []
        if chunks and chunks[0].payload.get("title") != correct_title:
            updated_points = [
                PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload={**point.payload, "title": correct_title},
                )
                for point in chunks
            ]
            client.upsert(_COLLECTION, updated_points)
            _write_paper_metadata_from_payload(
                updated_points[0].payload,
                paper_id=paper_dir.name,
                chunks_count=len(updated_points),
                title=correct_title,
            )
            stats["titles_fixed"] += 1
            if progress_callback:
                progress_callback(paper_dir.name, title_fixed=correct_title)

        existing_translations = {}
        for file_path in paper_dir.glob(f"{main_md.stem}.*.md"):
            code = _translation_code_from_file(file_path)
            if code:
                existing_translations[code] = file_path

        if not enabled:
            for _, file_path in existing_translations.items():
                file_path.unlink()
                stats["translations_removed"] += 1
                if progress_callback:
                    progress_callback(paper_dir.name, removed=file_path.name)
            continue

        src_lang = detect_language(text)
        src_lang_file = paper_dir / f"{main_md.stem}.{src_lang}.md"
        if not src_lang_file.exists():
            src_lang_file.write_text(text, encoding="utf-8")
            stats["translations_added"] += 1

        for lang in languages:
            code = lang["code"]
            name = lang["name"]
            if code == src_lang:
                continue
            target_file = paper_dir / f"{main_md.stem}.{code}.md"
            if target_file.exists():
                continue
            if progress_callback:
                progress_callback(paper_dir.name, translating=name)
            translated = translate(text, name)
            target_file.write_text(translated, encoding="utf-8")
            stats["translations_added"] += 1

        allowed = lang_codes | {src_lang}
        for code, file_path in existing_translations.items():
            if code not in allowed:
                file_path.unlink()
                stats["translations_removed"] += 1

    return stats

def remove_paper(paper_id: str) -> bool:
    """Remove a paper by paper_id, including local files."""
    if not paper_id:
        return False

    client = _get_client()
    if client.collection_exists(_COLLECTION):
        existing = _paper_points(client, paper_id)
        if existing:
            client.delete(_COLLECTION, points_selector=[point.id for point in existing])

    paper_dir = _paper_dir(paper_id)
    if not paper_dir.exists():
        return False
    shutil.rmtree(paper_dir)
    return True
