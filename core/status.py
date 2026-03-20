"""Service status checks for Compass (embedding, LLM, Qdrant)."""

from __future__ import annotations

import logging
import os

import httpx

from .config import load_config

logger = logging.getLogger("compass")

_OLLAMA_STATUS_TIMEOUT = 15


def _check_ollama(base_url: str) -> tuple[bool, str]:
    """Ping Ollama and return (ok, model_list_or_error)."""
    try:
        url = base_url.replace("localhost", "127.0.0.1")
        with httpx.Client(timeout=_OLLAMA_STATUS_TIMEOUT) as client:
            resp = client.get(f"{url}/api/tags")
        models = [m["name"] for m in resp.json().get("models", [])]
        return True, ", ".join(models) if models else "no models"
    except (httpx.HTTPError, httpx.TimeoutException, ValueError) as e:
        return False, str(e)


def check_embedding(cfg: dict) -> tuple[bool, str]:
    """Check embedding service availability. Returns (ok, detail)."""
    provider = cfg["embedding"]["provider"]
    model = cfg["embedding"]["model"]
    if provider == "ollama":
        ok, info = _check_ollama(cfg["embedding"].get("ollama_base_url", "http://localhost:11434"))
        if not ok:
            return False, f"Ollama unreachable: {info}"
        if model.split(":")[0] not in [m.split(":")[0] for m in info.split(", ")]:
            return False, f"Model '{model}' not pulled. Available: {info}"
        return True, f"ollama/{model}"
    elif provider == "openai":
        try:
            from .auth import resolve_openai_api_key
            resolve_openai_api_key({**cfg.get("llm", {}), **cfg["embedding"]})
            return True, f"openai/{model}"
        except Exception as e:
            return False, str(e)
    return False, f"Unknown provider: {provider}"


def check_llm(cfg: dict) -> tuple[bool, str]:
    """Check LLM service availability. Returns (ok, detail)."""
    provider = cfg["llm"].get("provider", "ollama")
    model = cfg["llm"].get("model", "")
    if provider == "cli":
        cmd = cfg["llm"].get("cli_command", "")
        return True, f"cli/{cmd}"
    if provider == "ollama":
        ok, info = _check_ollama(cfg["llm"].get("ollama_base_url", "http://localhost:11434"))
        if not ok:
            return False, f"Ollama unreachable: {info}"
        if model.split(":")[0] not in [m.split(":")[0] for m in info.split(", ")]:
            return False, f"Model '{model}' not pulled. Available: {info}"
        return True, f"ollama/{model}"
    elif provider == "claude":
        if not (os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY") or cfg["llm"].get("api_key")):
            return False, "ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN not set"
        return True, f"claude/{model}"
    elif provider in {"openai", "codex"}:
        try:
            from .auth import resolve_openai_api_key
            resolve_openai_api_key(cfg["llm"])
            auth_mode = cfg["llm"].get("auth_mode", "api_key")
            return True, f"{provider}/{model} (auth: {auth_mode})"
        except Exception as e:
            return False, str(e)
    return False, f"Unknown provider: {provider}"


def check_qdrant() -> tuple[bool, str]:
    """Check Qdrant availability. Returns (ok, detail)."""
    try:
        from .vectorstore import _get_client, _COLLECTION
        client = _get_client()
        if client.collection_exists(_COLLECTION):
            info = client.get_collection(_COLLECTION)
            return True, f"{info.points_count} chunks indexed"
        return True, "0 chunks indexed"
    except Exception as e:
        return False, str(e)
