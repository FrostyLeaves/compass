"""Compass — Paper Search Engine MCP Server.

Provides semantic search over an academic paper knowledge base.
Designed for LLM RAG: search relevant passages, then read full paper content.
"""

import sys
from pathlib import Path

# Ensure core is importable
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP

_READ_PAPER_MAX_CHARS = 50_000

mcp = FastMCP(
    "compass",
    instructions=(
        "Compass is an academic paper knowledge base with semantic search. "
        "Use it to find and read relevant papers before answering research questions.\n\n"
        "Recommended workflow:\n"
        "1. Use `search_papers` to find relevant passages for a question\n"
        "2. If you need more context, use `read_paper` to read the full paper or specific sections\n"
        "3. Synthesize the information to answer the user's question\n\n"
        "Available tools: search_papers, read_paper, list_papers"
    ),
    # The MCP surface is read-only, so stateless HTTP avoids session drift when
    # the dev server reloads or a client reconnects with a stale session id.
    stateless_http=True,
)


@mcp.tool()
def search_papers(query: str, top_k: int = 5, filter_title: str = "", filter_keywords: str = "") -> str:
    """Search the paper knowledge base for passages relevant to a query.

    Returns matching passages with paper titles, relevance scores, and chunk text.
    Use this as the first step to find relevant information, then use `read_paper`
    if you need more context from a specific paper.

    Args:
        query: Natural language search query
        top_k: Number of results to return (default 5)
        filter_title: Optional substring to match against paper titles
        filter_keywords: Optional comma-separated keywords for exact match filtering
    """
    from core.client import search

    kw_list = [k.strip().lower() for k in filter_keywords.split(",") if k.strip()] if filter_keywords else None
    results = search(
        query, top_k=top_k, deduplicate=True,
        filter_title=filter_title or None,
        filter_keywords=kw_list,
    )
    if not results:
        return "No relevant papers found."

    output = []
    for i, item in enumerate(results, 1):
        output.append(
            f"[{i}] \"{item['title']}\" (relevance: {item['score']:.3f})\n"
            f"{item['text']}"
        )
    return "\n\n---\n\n".join(output)


@mcp.tool()
def read_paper(title: str, section: str = "") -> str:
    """Read the full content or a specific section of a paper.

    Use this after `search_papers` to get more context from a paper.
    Returns the paper content in markdown format.

    Args:
        title: Paper title or paper id (as returned by search_papers or list_papers)
        section: Optional section heading to read (e.g. "Introduction", "Methods").
                 If empty, returns the full paper. If the paper is very long,
                 it will be truncated with a note.
    """
    from core.client import list_papers as _list
    import os

    papers = _list()
    # Find matching paper by paper_id or title (case-insensitive substring match)
    match = None
    title_lower = title.lower()
    for p in papers:
        if p.get("paper_id", "").lower() == title_lower or p["title"].lower() == title_lower:
            match = p
            break
    if not match:
        for p in papers:
            if title_lower in p.get("paper_id", "").lower() or title_lower in p["title"].lower():
                match = p
                break
    if not match:
        return f"Paper not found: {title}\nUse `list_papers` to see available papers."

    md_path = match.get("markdown_path", "")
    if not md_path or not os.path.isfile(md_path):
        return f"Markdown file not found for: {match['title']}"

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    if section:
        # Extract the requested section
        extracted = _extract_section(content, section)
        if extracted:
            return f"# {match['title']}\n## Section: {section}\n\n{extracted}"
        return (
            f"Section \"{section}\" not found in \"{match['title']}\".\n"
            f"Available sections: {', '.join(_get_sections(content))}"
        )

    # Full paper - truncate if very long to avoid overwhelming context
    if len(content) > _READ_PAPER_MAX_CHARS:
        sections = _get_sections(content)
        truncated = content[:_READ_PAPER_MAX_CHARS]
        return (
            f"{truncated}\n\n"
            f"[Content truncated at {_READ_PAPER_MAX_CHARS} characters. "
            f"Use `read_paper` with a `section` argument to read specific sections.]\n"
            f"Available sections: {', '.join(sections)}"
        )
    return content


def _extract_section(content: str, section: str) -> str:
    """Extract a section from markdown content by heading match."""
    import re
    lines = content.split("\n")
    section_lower = section.lower().strip()
    start = None
    start_level = None

    for i, line in enumerate(lines):
        heading_match = re.match(r"^(#{1,6})\s+(.+)", line)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip().lower()
            # Remove any trailing anchors or formatting
            heading_text = re.sub(r"\s*\{#[^}]*\}", "", heading_text)
            heading_text = re.sub(r"[*_`]", "", heading_text)

            if start is not None:
                # Found the next section at same or higher level, stop
                if level <= start_level:
                    return "\n".join(lines[start:i]).strip()
            elif section_lower in heading_text or heading_text in section_lower:
                start = i
                start_level = level

    if start is not None:
        return "\n".join(lines[start:]).strip()
    return ""


def _get_sections(content: str) -> list[str]:
    """Extract top-level section headings from markdown."""
    import re
    sections = []
    for line in content.split("\n"):
        m = re.match(r"^#{1,3}\s+(.+)", line)
        if m:
            heading = m.group(1).strip()
            heading = re.sub(r"\s*\{#[^}]*\}", "", heading)
            if heading:
                sections.append(heading)
    return sections


@mcp.tool()
def list_papers() -> str:
    """List all papers in the knowledge base with their titles and chunk counts."""
    from core.client import list_papers as _list

    papers = _list()
    if not papers:
        return "Knowledge base is empty."

    lines = [f"Total: {len(papers)} papers\n"]
    for p in papers:
        lines.append(f"  [{p['chunks_count']:3d} chunks] {p['title']} ({p.get('paper_id', '')})")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
