"""Auth helpers for reading OAuth tokens from Codex CLI config."""

from __future__ import annotations
import json
from pathlib import Path


def get_codex_oauth_token() -> str:
    """Read the OpenAI access_token from ~/.codex/auth.json (Codex CLI OAuth)."""
    auth_path = Path.home() / ".codex" / "auth.json"
    if not auth_path.exists():
        raise FileNotFoundError(f"Codex auth file not found: {auth_path}")

    data = json.loads(auth_path.read_text(encoding="utf-8"))
    tokens = data.get("tokens", {})
    token = tokens.get("access_token")
    if not token:
        raise ValueError("No access_token found in ~/.codex/auth.json. Run 'codex --login' first.")
    return token


def resolve_openai_api_key(cfg: dict) -> str:
    """Resolve the OpenAI API key: config api_key > env var > codex oauth fallback."""
    import os

    key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    # Fallback: try codex oauth
    try:
        return get_codex_oauth_token()
    except (FileNotFoundError, ValueError):
        raise ValueError(
            "OpenAI API key not set. Set api_key in config.yaml, "
            "OPENAI_API_KEY env var, or run 'codex --login' for OAuth."
        )
