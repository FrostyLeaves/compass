from __future__ import annotations

import yaml
from pathlib import Path

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

_config_cache: dict | None = None


def load_config(path: str | Path | None = None) -> dict:
    global _config_cache
    if _config_cache is not None and path is None:
        return _config_cache
    p = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path is None:
        _config_cache = cfg
    return cfg


def get_qdrant_config() -> dict:
    """Return Qdrant connection config: {path, url, api_key}."""
    cfg = load_config()
    qcfg = cfg.get("qdrant", {})
    result = {}
    if qcfg.get("url"):
        result["url"] = qcfg["url"]
        if qcfg.get("api_key"):
            result["api_key"] = qcfg["api_key"]
    else:
        base = Path(__file__).parent.parent
        default_path = cfg.get("storage", {}).get("qdrant_path", "./data/qdrant")
        path = (base / default_path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        result["path"] = str(path)
    return result


def get_papers_dir() -> Path:
    cfg = load_config()
    base = Path(__file__).parent.parent
    papers_path = cfg.get("storage", {}).get("papers_dir", "./data/papers")
    result = (base / papers_path).resolve()
    result.mkdir(parents=True, exist_ok=True)
    return result


def get_api_host() -> str:
    cfg = load_config()
    return cfg.get("api", {}).get("host", "localhost")


def get_api_port() -> int:
    cfg = load_config()
    return cfg.get("api", {}).get("port", 8000)


def get_project_root() -> Path:
    return Path(__file__).parent.parent.resolve()


def to_relative(path: str | Path) -> str:
    """Convert an absolute path to a project-relative path string."""
    try:
        return Path(path).resolve().relative_to(get_project_root()).as_posix()
    except ValueError:
        return str(path)


def reload_config() -> dict:
    """Clear the config cache and reload from disk."""
    global _config_cache
    _config_cache = None
    return load_config()


def to_absolute(rel_path: str) -> str:
    """Convert a project-relative path string to an absolute path string."""
    p = Path(rel_path)
    if p.is_absolute():
        return str(p)
    return str(get_project_root() / p)
