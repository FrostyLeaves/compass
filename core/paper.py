"""Paper filesystem layout, metadata, and result building."""

from __future__ import annotations

import json
import re
import shutil
import logging
from pathlib import Path
from typing import Any

from .config import get_papers_dir, to_relative, to_absolute

logger = logging.getLogger("compass")

_PAPER_MD_NAME = "paper.md"
_PAPER_PDF_NAME = "paper.pdf"
_PAPER_META_NAME = "metadata.json"


def _paper_dir(paper_id: str) -> Path:
    return get_papers_dir() / paper_id


def _paper_markdown_path(paper_id: str) -> Path:
    return _paper_dir(paper_id) / _PAPER_MD_NAME


def _paper_pdf_path(paper_id: str) -> Path:
    return _paper_dir(paper_id) / _PAPER_PDF_NAME


def _paper_meta_path(paper_id: str) -> Path:
    return _paper_dir(paper_id) / _PAPER_META_NAME


def _find_main_markdown_path(paper_dir: Path) -> Path | None:
    canonical = paper_dir / _PAPER_MD_NAME
    return canonical if canonical.exists() else None


def _find_main_pdf_path(paper_dir: Path) -> Path | None:
    canonical = paper_dir / _PAPER_PDF_NAME
    return canonical if canonical.exists() else None


def _paper_files_exist(paper_id: str) -> bool:
    return _paper_markdown_path(paper_id).exists() and _paper_pdf_path(paper_id).exists()


def _normalize_url(url: str | None) -> str:
    return (url or "").strip()


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _coerce_chunks_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _normalize_source_urls(*values: Any) -> list[str]:
    urls: list[str] = []
    for value in values:
        if not value:
            continue
        if isinstance(value, str):
            normalized = _normalize_url(value)
            if normalized:
                urls.append(normalized)
            continue
        if isinstance(value, (list, tuple, set)):
            for item in value:
                normalized = _normalize_url(str(item))
                if normalized:
                    urls.append(normalized)
    return _unique_strings(urls)


def _payload_source_urls(payload: dict[str, Any]) -> list[str]:
    return _normalize_source_urls(payload.get("source_urls", []), payload.get("source_url", ""))


def _payload_paper_id(payload: dict[str, Any]) -> str:
    return str(payload.get("paper_id", "")).strip()


def _relative_path(path: str | Path) -> str:
    return to_relative(path) if path else ""


def _absolute_if_present(path: str) -> str:
    return to_absolute(path) if path else ""


def _paper_metadata_payload(
    *,
    paper_id: str,
    title: str,
    chunks_count: int,
    markdown_path: str | Path,
    pdf_path: str | Path,
    keywords: list[str],
    source_urls: list[str],
    ingested_at: str,
) -> dict[str, Any]:
    normalized_urls = _normalize_source_urls(source_urls)
    rel_md = _relative_path(markdown_path)
    rel_pdf = _relative_path(pdf_path)
    return {
        "paper_id": paper_id,
        "title": title,
        "chunks_count": _coerce_chunks_count(chunks_count),
        "markdown_path": rel_md,
        "pdf_path": rel_pdf,
        "source_path": rel_pdf,
        "source_url": normalized_urls[0] if normalized_urls else "",
        "source_urls": normalized_urls,
        "keywords": list(keywords or []),
        "ingested_at": ingested_at,
    }


def _write_paper_metadata(
    *,
    paper_id: str,
    title: str,
    chunks_count: int,
    markdown_path: str | Path,
    pdf_path: str | Path,
    keywords: list[str],
    source_urls: list[str],
    ingested_at: str,
) -> dict[str, Any]:
    metadata = _paper_metadata_payload(
        paper_id=paper_id,
        title=title,
        chunks_count=chunks_count,
        markdown_path=markdown_path,
        pdf_path=pdf_path,
        keywords=keywords,
        source_urls=source_urls,
        ingested_at=ingested_at,
    )
    meta_path = _paper_meta_path(paper_id)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = meta_path.with_suffix(f"{meta_path.suffix}.tmp")
    tmp_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(meta_path)
    return metadata


def _write_paper_metadata_from_payload(
    payload: dict[str, Any],
    *,
    paper_id: str | None = None,
    chunks_count: int = 0,
    title: str | None = None,
) -> dict[str, Any]:
    resolved_paper_id = (paper_id or _payload_paper_id(payload)).strip()
    if not resolved_paper_id:
        raise ValueError("paper_id is required to write paper metadata")
    return _write_paper_metadata(
        paper_id=resolved_paper_id,
        title=title if title is not None else str(payload.get("title", "")),
        chunks_count=chunks_count,
        markdown_path=str(payload.get("markdown_path") or _paper_markdown_path(resolved_paper_id)),
        pdf_path=str(payload.get("pdf_path") or payload.get("source_path") or _paper_pdf_path(resolved_paper_id)),
        keywords=list(payload.get("keywords") or []),
        source_urls=_payload_source_urls(payload),
        ingested_at=str(payload.get("ingested_at", "")),
    )


def _read_paper_metadata(paper_id: str) -> dict[str, Any] | None:
    meta_path = _paper_meta_path(paper_id)
    if not meta_path.exists():
        return None
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    stored_paper_id = str(raw.get("paper_id", "")).strip()
    if stored_paper_id and stored_paper_id != paper_id:
        return None
    return _paper_metadata_payload(
        paper_id=paper_id,
        title=str(raw.get("title", "")),
        chunks_count=raw.get("chunks_count", 0),
        markdown_path=str(raw.get("markdown_path") or _paper_markdown_path(paper_id)),
        pdf_path=str(raw.get("pdf_path") or raw.get("source_path") or _paper_pdf_path(paper_id)),
        keywords=list(raw.get("keywords") or []),
        source_urls=_payload_source_urls(raw),
        ingested_at=str(raw.get("ingested_at", "")),
    )


def _build_paper_result(payload: dict[str, Any], chunks_count: int = 0) -> dict:
    paper_id = _payload_paper_id(payload)
    pdf_rel = str(payload.get("pdf_path") or payload.get("source_path") or "")
    md_rel = str(payload.get("markdown_path") or "")
    source_urls = _payload_source_urls(payload)
    pdf_abs = _absolute_if_present(pdf_rel)
    return {
        "paper_id": paper_id,
        "title": str(payload.get("title", "")),
        "path": pdf_abs,
        "chunks_count": _coerce_chunks_count(chunks_count),
        "markdown_path": _absolute_if_present(md_rel),
        "pdf_path": pdf_abs,
        "source_url": source_urls[0] if source_urls else "",
        "keywords": list(payload.get("keywords") or []),
        "ingested_at": str(payload.get("ingested_at", "")),
    }


def _paper_result_from_metadata(metadata: dict[str, Any]) -> dict:
    return _build_paper_result(metadata, chunks_count=metadata.get("chunks_count", 0))


def _save_markdown(pdf_path: str | Path, paper_id: str, markdown: str, images: dict) -> tuple[str, str]:
    """Save markdown, images, and a PDF copy under data/papers/{paper_id}/."""
    src_pdf = Path(pdf_path).resolve()
    paper_dir = _paper_dir(paper_id)
    paper_dir.mkdir(parents=True, exist_ok=True)

    md_path = _paper_markdown_path(paper_id)
    pdf_copy = _paper_pdf_path(paper_id)

    markdown = re.sub(r"<span[^>]*>\s*</span>", "", markdown)
    md_path.write_text(markdown, encoding="utf-8")

    if images:
        for filename, img_data in images.items():
            img_path = paper_dir / filename
            if isinstance(img_data, bytes):
                img_path.write_bytes(img_data)
            elif hasattr(img_data, "save"):
                img_data.save(str(img_path))

    if pdf_copy.resolve() != src_pdf:
        shutil.copy2(str(src_pdf), str(pdf_copy))

    return to_relative(md_path), to_relative(pdf_copy)


def _translation_code_from_file(path: Path) -> str | None:
    match = re.match(r"^.+\.([A-Za-z-]{2,10})\.md$", path.name)
    return match.group(1) if match else None
