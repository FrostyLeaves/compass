"""Unified LLM generation interface, supporting ollama / claude / openai / codex / cli."""

from __future__ import annotations

from .config import load_config

_DETECT_LANG_MAX_CHARS = 2000
_TRANSLATE_MAX_TOKENS = 16_000
_KEYWORDS_MAX_TOKENS = 200
_OLLAMA_DEFAULT_URL = "http://localhost:11434"
_CLI_TIMEOUT = 300

SYSTEM_PROMPT = (
    "You are Compass, an academic paper search assistant. Answer user questions based on retrieved paper passages. "
    "Cite specific paper titles in your answers. Be accurate and concise. "
    "If the retrieved content is insufficient to answer the question, clearly state so."
)


def _context_key(item: dict) -> str:
    return item.get("paper_id") or item.get("source_path") or item.get("title", "")


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _context_range(item: dict) -> tuple[int, int]:
    start = _safe_int(item.get("parent_start_chunk_index", item.get("chunk_index", 0)))
    end = _safe_int(item.get("parent_end_chunk_index", item.get("chunk_index", start)), start)
    return (min(start, end), max(start, end))


def _merge_window_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if left in right:
        return right
    if right in left:
        return left

    max_overlap = min(len(left), len(right))
    for size in range(max_overlap, 0, -1):
        if left[-size:] == right[:size]:
            return left + right[size:]
    return left.rstrip() + "\n\n" + right.lstrip()


def _merge_context_windows(context: list[dict]) -> list[dict]:
    merged_groups: dict[str, list[dict]] = {}
    for item in sorted(context, key=lambda entry: (_context_key(entry), *_context_range(entry))):
        key = _context_key(item)
        window = dict(item)
        window["parent_text"] = item.get("parent_text", item.get("text", ""))
        start, end = _context_range(item)
        window["parent_start_chunk_index"] = start
        window["parent_end_chunk_index"] = end
        merged_groups.setdefault(key, [])
        group = merged_groups[key]
        if group and start <= group[-1]["parent_end_chunk_index"] + 1:
            existing = group[-1]
            existing_score = float(existing.get("score", 0.0))
            window_score = float(window.get("score", 0.0))
            existing["parent_text"] = _merge_window_text(existing.get("parent_text", ""), window["parent_text"])
            existing["parent_start_chunk_index"] = min(existing["parent_start_chunk_index"], start)
            existing["parent_end_chunk_index"] = max(existing["parent_end_chunk_index"], end)
            existing["score"] = max(existing_score, window_score)
            existing["chunk_index"] = min(existing.get("chunk_index", start), window.get("chunk_index", start))
            existing["parent_token_count"] = max(
                _safe_int(existing.get("parent_token_count", 0)),
                _safe_int(window.get("parent_token_count", 0)),
            )
            existing["content_types"] = sorted(set(existing.get("content_types", [])) | set(window.get("content_types", [])))
            if window_score > existing_score:
                existing["section_heading"] = window.get("section_heading", existing.get("section_heading", ""))
                existing["heading_path"] = window.get("heading_path", existing.get("heading_path", []))
                existing["heading_path_text"] = window.get("heading_path_text", existing.get("heading_path_text", ""))
                existing["dominant_content_type"] = window.get(
                    "dominant_content_type",
                    existing.get("dominant_content_type", ""),
                )
        else:
            group.append(window)

    merged = [item for group in merged_groups.values() for item in group]
    merged.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return merged


def _context_heading(item: dict) -> str:
    heading_path = [part for part in (item.get("heading_path") or []) if part]
    title = (item.get("title") or "").strip()
    if heading_path and heading_path[0].strip() == title:
        heading_path = heading_path[1:]
    if heading_path:
        return " > ".join(heading_path)
    section_heading = item.get("section_heading", "").strip()
    return "" if section_heading == title else section_heading


def _llm_call(
    system_prompt: str,
    user_prompt: str,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int = 4096,
) -> str:
    """Internal unified LLM call used by generate(), detect_language(), and translate()."""
    cfg = load_config()["llm"]
    provider = provider or cfg["provider"]
    model = model or cfg.get("model")

    if provider == "ollama":
        return _generate_ollama(user_prompt, model, cfg.get("ollama_base_url", _OLLAMA_DEFAULT_URL), system_prompt)
    elif provider == "claude":
        return _generate_claude(user_prompt, model, cfg, system_prompt, max_tokens=max_tokens)
    elif provider == "openai":
        return _generate_openai(user_prompt, model, cfg, system_prompt, max_tokens=max_tokens)
    elif provider == "codex":
        return _generate_codex(user_prompt, model, cfg, system_prompt)
    elif provider == "cli":
        return _generate_cli(user_prompt, cfg, system_prompt)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def generate(
    question: str,
    context: list[dict],
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """Generate an answer using an LLM based on retrieved context and the user question.

    context: list of results returned by retriever.search()
    """
    merged_context = _merge_context_windows(context)

    # Build context text
    ctx_parts = []
    for i, item in enumerate(merged_context, 1):
        heading = _context_heading(item)
        label = f"[{i}] \"{item['title']}\""
        if heading:
            label += f" [{heading}]"
        label += f" (relevance: {item['score']:.2f})"
        ctx_parts.append(f"{label}\n{item.get('parent_text', item['text'])}")
    context_text = "\n\n---\n\n".join(ctx_parts)

    user_prompt = f"The following are relevant passages retrieved from Compass:\n\n{context_text}\n\nUser question: {question}"
    return _llm_call(SYSTEM_PROMPT, user_prompt, provider, model)


def detect_language(text: str) -> str:
    """Detect the language of a text, returning an ISO 639-1 code (e.g. 'en', 'zh')."""
    snippet = text[:_DETECT_LANG_MAX_CHARS]
    system = "You are a language detection tool. Respond with ONLY the ISO 639-1 language code (e.g. en, zh, ja, de, fr). No explanation."
    user = f"Detect the language of the following text:\n\n{snippet}"
    result = _llm_call(system, user).strip().lower()
    # Extract just the code in case LLM returns extra text
    for token in result.split():
        if len(token) == 2 and token.isalpha():
            return token
    return result[:2]


_TRANSLATE_MAX_CHARS = 30_000


def translate(text: str, target_lang_name: str) -> str:
    """Translate markdown text to target language, preserving markdown formatting.

    Long documents are split at heading boundaries and translated in batches
    to avoid truncation from LLM output token limits.
    """
    import re as _re

    system = (
        f"You are a professional translator. Translate the following markdown document to {target_lang_name}. "
        "Preserve all markdown formatting, headings, links, images, code blocks, and LaTeX formulas exactly. "
        "Only translate the natural language text. Output ONLY the translated markdown, no preamble."
    )

    if len(text) <= _TRANSLATE_MAX_CHARS:
        return _llm_call(system, text, max_tokens=_TRANSLATE_MAX_TOKENS)

    # Split at markdown headings to get logical sections
    sections = _re.split(r'(?=\n#{1,3} )', text)

    # Group sections into batches under the char limit
    batches: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for section in sections:
        if current_len + len(section) > _TRANSLATE_MAX_CHARS and current:
            batches.append(current)
            current = [section]
            current_len = len(section)
        else:
            current.append(section)
            current_len += len(section)
    if current:
        batches.append(current)

    translated_parts = []
    for batch in batches:
        chunk = "".join(batch)
        translated = _llm_call(system, chunk, max_tokens=_TRANSLATE_MAX_TOKENS)
        translated_parts.append(translated.strip())

    return "\n\n".join(translated_parts)


_KEYWORDS_MAX_CHARS = 5000


def extract_keywords(text: str) -> list[str]:
    """Extract 5-10 keywords from paper text using the configured LLM.

    Returns lowercase keyword list; returns [] on failure (non-blocking).
    """
    prompt = (
        "Extract 5-10 keywords/key phrases from this paper. "
        "Return ONLY a comma-separated list, nothing else.\n\n"
        + text[:_KEYWORDS_MAX_CHARS]
    )
    try:
        result = _llm_call("You are a keyword extractor.", prompt, max_tokens=_KEYWORDS_MAX_TOKENS)
        return [k.strip().lower() for k in result.split(",") if k.strip()]
    except (ValueError, RuntimeError, FileNotFoundError):
        return []


def _generate_ollama(prompt: str, model: str, base_url: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    import ollama as _ollama
    client = _ollama.Client(host=base_url.replace("localhost", "127.0.0.1"))
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return resp["message"]["content"]


def _generate_claude(prompt: str, model: str, cfg: dict, system_prompt: str = SYSTEM_PROMPT, max_tokens: int = 4096) -> str:
    import os
    import anthropic
    api_key = cfg.get("api_key") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    base_url = cfg.get("api_base_url") or os.environ.get("ANTHROPIC_BASE_URL") or None
    if not api_key:
        raise ValueError("Claude API key not set. Set api_key in config.yaml or ANTHROPIC_API_KEY env var.")
    client = anthropic.Anthropic(
        api_key=api_key,
        base_url=base_url,
        default_headers={"User-Agent": "claude_code"},
    )
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def _generate_openai(prompt: str, model: str, cfg: dict, system_prompt: str = SYSTEM_PROMPT, max_tokens: int = 4096) -> str:
    import openai as _openai
    from .auth import resolve_openai_api_key
    client = _openai.OpenAI(
        api_key=resolve_openai_api_key(cfg),
        base_url=cfg.get("api_base_url") or None,
    )
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def _generate_codex(prompt: str, model: str, cfg: dict, system_prompt: str = SYSTEM_PROMPT) -> str:
    import openai as _openai
    from .auth import resolve_openai_api_key

    client = _openai.OpenAI(
        api_key=resolve_openai_api_key(cfg),
        base_url=cfg.get("api_base_url") or None,
    )
    resp = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=prompt,
    )

    output_text = getattr(resp, "output_text", "")
    if output_text:
        return output_text

    parts = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", "") != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", "") not in {"output_text", "text"}:
                continue
            text = getattr(content, "text", "")
            if isinstance(text, str) and text:
                parts.append(text)
            elif hasattr(text, "value") and text.value:
                parts.append(text.value)

    if parts:
        return "\n".join(parts)

    raise ValueError("Codex response did not contain text output.")


def _generate_cli(prompt: str, cfg: dict, system_prompt: str = SYSTEM_PROMPT) -> str:
    import shutil
    import subprocess
    cmd = cfg.get("cli_command", "claude")
    args = cfg.get("cli_args", ["--print"])
    model = cfg.get("model")

    resolved = shutil.which(cmd)
    if not resolved:
        raise FileNotFoundError(f"CLI command not found: {cmd}")

    full_prompt = f"{system_prompt}\n\n{prompt}"
    run_args = [resolved] + list(args)
    if model:
        run_args += ["--model", model]

    result = subprocess.run(run_args, input=full_prompt, capture_output=True, text=True, encoding="utf-8", timeout=_CLI_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError(f"CLI command failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip()
