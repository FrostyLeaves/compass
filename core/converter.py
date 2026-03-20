"""PDF-to-Markdown conversion using marker-pdf."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_config

_LLM_SERVICE_MAP = {
    "claude": "marker.services.claude.ClaudeService",
    "openai": "marker.services.openai.OpenAIService",
    "codex": "marker.services.openai.OpenAIService",
    "ollama": "marker.services.ollama.OllamaService",
    "gemini": "marker.services.gemini.GoogleGeminiService",
}


def _get_marker_config() -> tuple[dict, dict]:
    """Return (marker_cfg, llm_cfg), supporting both new and legacy config layout."""
    full_cfg = load_config()
    llm_cfg = full_cfg.get("llm", {})

    # New layout: converter.marker
    if "converter" in full_cfg:
        return full_cfg["converter"].get("marker", {}), llm_cfg

    # Legacy layout: top-level marker:
    return full_cfg.get("marker", {}), llm_cfg


def convert_pdf(pdf_path: Path) -> tuple[str, dict[str, Any]]:
    """Convert a PDF to (markdown_text, images_dict).

    images_dict maps filename to ``bytes`` or ``PIL.Image``.
    """
    marker_cfg, llm_cfg = _get_marker_config()
    return _convert_marker(pdf_path, marker_cfg, llm_cfg)


def _resolve_marker_llm_config(marker_cfg: dict, llm_cfg: dict) -> tuple[str | None, dict]:
    """Resolve marker's LLM configuration. Returns (llm_service_class_path, extra_config).

    When llm_service is "default", derives marker's required parameters
    from the Compass llm config section.
    """
    service_name = marker_cfg.get("llm_service", "default")

    if service_name == "default":
        provider = llm_cfg.get("provider", "ollama")
        provider_map = {"claude": "claude", "openai": "openai", "codex": "codex", "ollama": "ollama"}
        service_name = provider_map.get(provider)
        if not service_name:
            return None, {}

    service_cls = _LLM_SERVICE_MAP.get(service_name)
    if not service_cls:
        return None, {}

    extra: dict[str, Any] = {}
    if service_name == "default" or marker_cfg.get("llm_service", "default") == "default":
        provider = llm_cfg.get("provider", "ollama")
        if provider == "claude":
            extra["claude_model_name"] = llm_cfg.get("model", "claude-sonnet-4-20250514")
            if llm_cfg.get("api_key"):
                extra["claude_api_key"] = llm_cfg["api_key"]
        elif provider in {"openai", "codex"}:
            extra["openai_model"] = llm_cfg.get("model", "gpt-4o-mini")
            from .auth import resolve_openai_api_key
            try:
                extra["openai_api_key"] = resolve_openai_api_key(llm_cfg)
            except (ValueError, KeyError):
                pass
            if llm_cfg.get("base_url"):
                extra["openai_base_url"] = llm_cfg["base_url"]
        elif provider == "ollama":
            extra["ollama_model"] = llm_cfg.get("model", "llama3.2-vision")
            extra["ollama_base_url"] = (
                llm_cfg.get("ollama_base_url", "http://localhost:11434")
                .replace("localhost", "127.0.0.1")
            )
    else:
        for key in ("claude_model_name", "claude_api_key",
                     "openai_model", "openai_api_key", "openai_base_url",
                     "ollama_model", "ollama_base_url"):
            if key in marker_cfg:
                extra[key] = marker_cfg[key]

    return service_cls, extra


def _convert_marker(pdf_path: Path, marker_cfg: dict, llm_cfg: dict) -> tuple[str, dict]:
    """Convert PDF using marker-pdf."""
    from marker.converters.pdf import PdfConverter
    from marker.config.parser import ConfigParser
    from marker.models import create_model_dict

    use_llm = marker_cfg.get("use_llm", False)
    parser_config: dict[str, Any] = {
        "output_format": "markdown",
        "use_llm": use_llm,
        "disable_links": True,
    }

    llm_service = None
    if use_llm:
        service_cls, extra = _resolve_marker_llm_config(marker_cfg, llm_cfg)
        llm_service = service_cls
        parser_config.update(extra)

    config_parser = ConfigParser(parser_config)
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config=config_parser.generate_config_dict(),
        llm_service=llm_service,
    )
    rendered = converter(str(pdf_path))
    images = getattr(rendered, "images", {}) or {}
    return rendered.markdown, images
