"""Text chunking, title extraction, and LLM cleanup."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("compass")

_CLEANUP_SYSTEM = (
    "You are a document formatting specialist. Your task is to improve the layout of markdown that was "
    "automatically converted from a PDF (which may be a research paper, technical report, or presentation slides). "
    "Fix text fragmentation and structural issues while preserving all content including image references."
)

_CLEANUP_PROMPT = """\
Below is raw markdown converted from a PDF. It may contain these issues:
- Broken image syntax like `![](_page_X.jpeg>)` (note the stray `>` inside the parenthesis)
- Fragmented text: related sentences split across multiple lines with no paragraph logic
- Redundant page header/footer lines repeating the paper/course title and page number
- Inconsistent or missing heading structure

Please clean up this markdown by:
1. Fixing any broken image syntax (e.g. `![](_page_X.jpeg>)` -> `![](_page_X.jpeg)`) but keeping all image references intact
2. Merging fragmented lines into coherent paragraphs based on their meaning
3. Removing repetitive header/footer lines (e.g. course name + page number appearing every few paragraphs)
4. Fixing heading levels so the document has a logical structure
5. Preserving ALL technical content: formulas, code, tables, citations, and every piece of information
6. Keeping LaTeX math notation unchanged

Output ONLY the cleaned markdown. No preamble, no explanation.

---

{text}"""

_CLEANUP_MAX_CHARS = 60_000
_MIN_CHUNK_TOKENS = 48
_SENTENCE_JOIN_KINDS = {"paragraph"}
_LINE_JOIN_KINDS = {"list", "quote"}
_NON_SPLITTABLE_KINDS = {"code", "table", "math"}
_ADAPTIVE_LIMIT_FACTORS = {
    "paragraph": 1.0,
    "quote": 0.85,
    "list": 0.85,
    "math": 0.75,
    "table": 0.65,
    "code": 0.65,
}

_TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]|[^\w\s]",
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？；;])\s+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_CODE_FENCE_RE = re.compile(r"^\s*([`~]{3,})(.*)$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_DIVIDER_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*(?::?-{3,}:?)?\s*\|?\s*$")
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?")
_DISPLAY_MATH_BOUNDARY_RE = re.compile(r"^\s*(\$\$|\\\[|\\\])\s*$")


@dataclass(frozen=True)
class _Block:
    text: str
    kind: str
    token_count: int


@dataclass
class _Section:
    heading_path: tuple[str, ...]
    heading_line: str
    heading_level: int
    lines: list[str]


def _cleanup_markdown_with_llm(text: str) -> str:
    """Post-process marker-pdf output with an LLM to fix layout issues."""
    from .generator import _llm_call

    page_pattern = re.compile(r"(?=!\[\]\(_page_\d+_)")
    pages = page_pattern.split(text)

    batches: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for page in pages:
        if current_len + len(page) > _CLEANUP_MAX_CHARS and current:
            batches.append(current)
            current = [page]
            current_len = len(page)
        else:
            current.append(page)
            current_len += len(page)
    if current:
        batches.append(current)

    cleaned_parts = []
    for batch in batches:
        chunk = "".join(batch)
        prompt = _CLEANUP_PROMPT.format(text=chunk)
        cleaned = _llm_call(_CLEANUP_SYSTEM, prompt, max_tokens=16000)
        cleaned_parts.append(cleaned.strip())

    return "\n\n".join(cleaned_parts)


def _estimate_tokens(text: str) -> int:
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return len(_TOKEN_RE.findall(stripped))


def _make_block(text: str, kind: str) -> _Block | None:
    normalized = text.strip()
    if not normalized:
        return None
    return _Block(text=normalized, kind=kind, token_count=_estimate_tokens(normalized))


def _code_fence_token(line: str) -> str | None:
    match = _CODE_FENCE_RE.match(line)
    return match.group(1) if match else None


def _matches_code_fence(line: str, fence: str) -> bool:
    candidate = _code_fence_token(line)
    return candidate is not None and candidate[0] == fence[0] and len(candidate) >= len(fence)


def _matches_heading(line: str) -> re.Match[str] | None:
    return _HEADING_RE.match(line.lstrip())


def _is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    current = lines[index].strip()
    nxt = lines[index + 1].strip()
    return bool(_TABLE_ROW_RE.match(current) and _TABLE_DIVIDER_RE.match(nxt))


def _parse_sections(text: str) -> list[_Section]:
    sections: list[_Section] = []
    heading_stack: list[str] = []
    current = _Section(heading_path=(), heading_line="", heading_level=0, lines=[])
    active_fence: str | None = None

    for line in text.splitlines():
        fence = _code_fence_token(line)
        if active_fence:
            current.lines.append(line)
            if fence and _matches_code_fence(line, active_fence):
                active_fence = None
            continue
        if fence:
            current.lines.append(line)
            active_fence = fence
            continue

        heading_match = _matches_heading(line)
        if heading_match:
            if current.heading_line or any(part.strip() for part in current.lines):
                sections.append(current)
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            heading_stack = heading_stack[:level - 1] + [heading_text]
            current = _Section(
                heading_path=tuple(heading_stack),
                heading_line=line.strip(),
                heading_level=level,
                lines=[],
            )
            continue

        current.lines.append(line)

    if current.heading_line or any(part.strip() for part in current.lines):
        sections.append(current)
    return sections


def _split_list_units(text: str) -> list[str]:
    units: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if _LIST_ITEM_RE.match(line):
            if current:
                units.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        units.append("\n".join(current).strip())
    return [unit for unit in units if unit]


def _split_quote_units(text: str) -> list[str]:
    units = [line.strip() for line in text.splitlines() if line.strip()]
    return units or [text.strip()]


def _split_prose_units(text: str) -> list[str]:
    flattened = re.sub(r"\s*\n\s*", " ", text).strip()
    if not flattened:
        return []
    units = [part.strip() for part in _SENTENCE_SPLIT_RE.split(flattened) if part.strip()]
    return units or [flattened]


def _join_units(units: list[str], kind: str) -> str:
    if kind in _LINE_JOIN_KINDS:
        return "\n".join(units)
    if kind in _SENTENCE_JOIN_KINDS:
        return " ".join(units)
    return "\n".join(units)


def _split_dense_text(text: str, kind: str, max_tokens: int) -> list[_Block]:
    approx_chars = max(80, max_tokens * 4)
    parts: list[_Block] = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= approx_chars:
            block = _make_block(remaining, kind)
            if block:
                parts.append(block)
            break
        split_point = remaining.rfind(" ", 0, approx_chars)
        if split_point < approx_chars // 2:
            split_point = approx_chars
        block = _make_block(remaining[:split_point], kind)
        if block:
            parts.append(block)
        remaining = remaining[split_point:].strip()
    return parts


def _pack_units_as_blocks(units: list[str], kind: str, max_tokens: int) -> list[_Block]:
    blocks: list[_Block] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        normalized = unit.strip()
        if not normalized:
            continue
        unit_tokens = _estimate_tokens(normalized)
        if unit_tokens > max_tokens:
            if current:
                block = _make_block(_join_units(current, kind), kind)
                if block:
                    blocks.append(block)
                current = []
                current_tokens = 0
            blocks.extend(_split_dense_text(normalized, kind, max_tokens))
            continue
        if current and current_tokens + unit_tokens > max_tokens:
            block = _make_block(_join_units(current, kind), kind)
            if block:
                blocks.append(block)
            current = [normalized]
            current_tokens = unit_tokens
        else:
            current.append(normalized)
            current_tokens += unit_tokens
    if current:
        block = _make_block(_join_units(current, kind), kind)
        if block:
            blocks.append(block)
    return blocks


def _split_oversized_block(block: _Block, max_tokens: int) -> list[_Block]:
    if block.token_count <= max_tokens or block.kind in _NON_SPLITTABLE_KINDS:
        return [block]
    if block.kind == "list":
        units = _split_list_units(block.text)
    elif block.kind == "quote":
        units = _split_quote_units(block.text)
    else:
        units = _split_prose_units(block.text)
    return _pack_units_as_blocks(units, block.kind, max_tokens)


def _split_section_blocks(lines: list[str]) -> list[_Block]:
    blocks: list[_Block] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue

        fence = _code_fence_token(line)
        if fence:
            block_lines = [line]
            index += 1
            while index < len(lines):
                block_lines.append(lines[index])
                if _matches_code_fence(lines[index], fence):
                    index += 1
                    break
                index += 1
            block = _make_block("\n".join(block_lines), "code")
            if block:
                blocks.append(block)
            continue

        if _is_table_start(lines, index):
            block_lines = [lines[index]]
            index += 1
            while index < len(lines) and lines[index].strip() and _TABLE_ROW_RE.match(lines[index].strip()):
                block_lines.append(lines[index])
                index += 1
            block = _make_block("\n".join(block_lines), "table")
            if block:
                blocks.append(block)
            continue

        if _DISPLAY_MATH_BOUNDARY_RE.match(line.strip()):
            boundary = line.strip()
            block_lines = [line]
            index += 1
            closing = "\\]" if boundary == "\\[" else boundary
            while index < len(lines):
                block_lines.append(lines[index])
                if lines[index].strip() == closing:
                    index += 1
                    break
                index += 1
            block = _make_block("\n".join(block_lines), "math")
            if block:
                blocks.append(block)
            continue

        if _LIST_ITEM_RE.match(line):
            block_lines = [line]
            index += 1
            while index < len(lines) and lines[index].strip():
                if _LIST_ITEM_RE.match(lines[index]) or lines[index].startswith((" ", "\t")):
                    block_lines.append(lines[index])
                    index += 1
                else:
                    break
            block = _make_block("\n".join(block_lines), "list")
            if block:
                blocks.append(block)
            continue

        if _BLOCKQUOTE_RE.match(line):
            block_lines = [line]
            index += 1
            while index < len(lines) and lines[index].strip() and _BLOCKQUOTE_RE.match(lines[index]):
                block_lines.append(lines[index])
                index += 1
            block = _make_block("\n".join(block_lines), "quote")
            if block:
                blocks.append(block)
            continue

        block_lines = [line]
        index += 1
        while index < len(lines) and lines[index].strip():
            if (
                _code_fence_token(lines[index])
                or _is_table_start(lines, index)
                or _DISPLAY_MATH_BOUNDARY_RE.match(lines[index].strip())
                or _LIST_ITEM_RE.match(lines[index])
                or _BLOCKQUOTE_RE.match(lines[index])
            ):
                break
            block_lines.append(lines[index])
            index += 1
        block = _make_block("\n".join(block_lines), "paragraph")
        if block:
            blocks.append(block)

    return blocks


def _adaptive_chunk_limit(kinds: set[str], base_tokens: int) -> int:
    factor = min((_ADAPTIVE_LIMIT_FACTORS.get(kind, 1.0) for kind in kinds), default=1.0)
    return max(_MIN_CHUNK_TOKENS, int(base_tokens * factor))


def _collect_overlap_count(blocks: list[_Block], overlap_tokens: int) -> int:
    if overlap_tokens <= 0 or len(blocks) <= 1:
        return 0
    total = 0
    count = 0
    for block in reversed(blocks):
        if count >= len(blocks) - 1:
            break
        if total and total + block.token_count > overlap_tokens:
            break
        if not total and block.token_count > overlap_tokens:
            break
        total += block.token_count
        count += 1
        if total >= overlap_tokens:
            break
    return count


def _build_chunk(section: _Section, blocks: list[_Block]) -> dict:
    text_parts = [section.heading_line] if section.heading_line else []
    text_parts.extend(block.text for block in blocks)
    text = "\n\n".join(part for part in text_parts if part.strip())
    kind_tokens: dict[str, int] = {}
    for block in blocks:
        kind_tokens[block.kind] = kind_tokens.get(block.kind, 0) + block.token_count
    dominant_kind = max(kind_tokens.items(), key=lambda item: item[1])[0]
    heading_path = list(section.heading_path)
    return {
        "text": text,
        "chunk_index": -1,
        "section_heading": heading_path[-1] if heading_path else "",
        "heading_path": heading_path,
        "heading_level": section.heading_level,
        "heading_path_text": " > ".join(heading_path),
        "token_count": _estimate_tokens(text),
        "content_types": sorted(kind_tokens),
        "dominant_content_type": dominant_kind,
    }


def _chunk_section(section: _Section, chunk_size: int, chunk_overlap: int) -> list[dict]:
    base_blocks = _split_section_blocks(section.lines)
    if not base_blocks:
        return []

    heading_tokens = _estimate_tokens(section.heading_line)
    normalized_blocks: list[_Block] = []
    for block in base_blocks:
        block_limit = _adaptive_chunk_limit({block.kind}, chunk_size)
        max_body_tokens = max(_MIN_CHUNK_TOKENS, block_limit - heading_tokens)
        normalized_blocks.extend(_split_oversized_block(block, max_body_tokens))

    chunks: list[dict] = []
    start = 0
    while start < len(normalized_blocks):
        current: list[_Block] = []
        current_kinds: set[str] = set()
        current_tokens = heading_tokens
        index = start

        while index < len(normalized_blocks):
            block = normalized_blocks[index]
            proposed_kinds = current_kinds | {block.kind}
            limit = _adaptive_chunk_limit(proposed_kinds, chunk_size)
            if current and current_tokens + block.token_count > limit:
                break
            current.append(block)
            current_kinds = proposed_kinds
            current_tokens += block.token_count
            index += 1

        if not current:
            current = [normalized_blocks[start]]
            index = start + 1

        chunks.append(_build_chunk(section, current))
        if index >= len(normalized_blocks):
            break

        overlap_count = _collect_overlap_count(current, chunk_overlap)
        next_start = index - overlap_count
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def _chunk_text(text: str, chunk_size: int = 220, chunk_overlap: int = 40) -> list[dict]:
    """Split markdown by heading-aware blocks using token-estimated chunk budgets."""
    chunk_size = max(_MIN_CHUNK_TOKENS, int(chunk_size or 0))
    chunk_overlap = max(0, int(chunk_overlap or 0))

    chunks: list[dict] = []
    for section in _parse_sections(text):
        chunks.extend(_chunk_section(section, chunk_size, chunk_overlap))

    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index
    return chunks


def _chunk_token_count(chunk: dict) -> int:
    return int(chunk.get("token_count") or _estimate_tokens(chunk.get("text", "")))


def _heading_prefix_len(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    size = 0
    for l_item, r_item in zip(left, right):
        if l_item != r_item:
            break
        size += 1
    return size


def _parent_candidate_priority(anchor_path: tuple[str, ...], anchor_index: int, candidate: dict, candidate_index: int) -> tuple[int, int, int, int]:
    candidate_path = tuple(candidate.get("heading_path") or ())
    common_prefix = _heading_prefix_len(anchor_path, candidate_path)
    exact_match = 1 if anchor_path and candidate_path == anchor_path else 0
    distance = abs(candidate_index - anchor_index)
    return (exact_match, common_prefix, -distance, -_chunk_token_count(candidate))


def _assign_parent_text(chunks: list[dict], parent_chunk_size: int) -> None:
    """Assign a larger, heading-aware context window to each child chunk."""
    if not chunks:
        return
    if parent_chunk_size <= 0:
        for chunk in chunks:
            chunk["parent_text"] = chunk["text"]
            chunk["parent_start_chunk_index"] = chunk["chunk_index"]
            chunk["parent_end_chunk_index"] = chunk["chunk_index"]
            chunk["parent_token_count"] = _chunk_token_count(chunk)
        return

    texts = [chunk["text"] for chunk in chunks]
    total_chunks = len(texts)

    for index, chunk in enumerate(chunks):
        anchor_path = tuple(chunk.get("heading_path") or ())
        lo = hi = index
        total = _chunk_token_count(chunk)

        while total < parent_chunk_size:
            candidates: list[tuple[str, int]] = []
            if lo > 0:
                candidates.append(("left", lo - 1))
            if hi < total_chunks - 1:
                candidates.append(("right", hi + 1))
            if not candidates:
                break

            side, candidate_index = max(
                candidates,
                key=lambda item: _parent_candidate_priority(anchor_path, index, chunks[item[1]], item[1]),
            )
            total += _chunk_token_count(chunks[candidate_index]) + 1
            if side == "left":
                lo = candidate_index
            else:
                hi = candidate_index

        chunk["parent_text"] = "\n".join(texts[lo:hi + 1])
        chunk["parent_start_chunk_index"] = chunks[lo]["chunk_index"]
        chunk["parent_end_chunk_index"] = chunks[hi]["chunk_index"]
        chunk["parent_token_count"] = total


def _extract_title(text: str, pdf_path: str) -> str:
    """Extract paper title from the first H1 heading, falling back to filename."""
    for line in text.split("\n")[:10]:
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return Path(pdf_path).stem
