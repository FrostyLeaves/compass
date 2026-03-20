"""Text chunking, title extraction, and LLM cleanup."""

from __future__ import annotations

import re
import logging
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


def _chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[dict]:
    """Split by headings, then chunk by size. Returns [{text, chunk_index, section_heading}]."""
    sections = re.split(r"\n(?=#{1,3} )", text)
    chunks = []
    current = ""
    current_heading = ""

    def _extract_heading(section_text: str) -> str:
        for line in section_text.split("\n"):
            line = line.strip()
            if re.match(r"^#{1,3} ", line):
                return re.sub(r"^#{1,3} ", "", line).strip()
        return ""

    chunk_headings = []

    for section in sections:
        heading = _extract_heading(section) or current_heading
        if len(current) + len(section) <= chunk_size:
            current += ("\n" if current else "") + section
            if heading:
                current_heading = heading
        else:
            if current:
                chunks.append(current.strip())
                chunk_headings.append(current_heading)
            if len(section) > chunk_size:
                if heading:
                    current_heading = heading
                words = section
                while len(words) > chunk_size:
                    split_point = words.rfind(" ", 0, chunk_size)
                    if split_point == -1:
                        split_point = chunk_size
                    chunks.append(words[:split_point].strip())
                    chunk_headings.append(current_heading)
                    words = words[max(0, split_point - chunk_overlap):]
                if words.strip():
                    chunks.append(words.strip())
                    chunk_headings.append(current_heading)
                current = ""
            else:
                current = section
                if heading:
                    current_heading = heading

    if current.strip():
        chunks.append(current.strip())
        chunk_headings.append(current_heading)

    return [{"text": chunk, "chunk_index": i, "section_heading": chunk_headings[i]} for i, chunk in enumerate(chunks)]


def _assign_parent_text(chunks: list[dict], parent_chunk_size: int) -> None:
    """Assign parent_text to each child chunk by expanding to neighboring chunks."""
    if not chunks:
        return
    if parent_chunk_size <= 0:
        for chunk in chunks:
            chunk["parent_text"] = chunk["text"]
        return

    texts = [chunk["text"] for chunk in chunks]
    n = len(texts)

    for i, chunk in enumerate(chunks):
        lo, hi = i, i
        total = len(texts[i])

        while total < parent_chunk_size:
            expanded = False
            if lo > 0:
                lo -= 1
                total += len(texts[lo]) + 1
                expanded = True
            if hi < n - 1 and total < parent_chunk_size:
                hi += 1
                total += len(texts[hi]) + 1
                expanded = True
            if not expanded:
                break

        chunk["parent_text"] = "\n".join(texts[lo:hi + 1])


def _extract_title(text: str, pdf_path: str) -> str:
    """Extract paper title from the first H1 heading, falling back to filename."""
    for line in text.split("\n")[:10]:
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return Path(pdf_path).stem
