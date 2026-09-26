"""Central configuration loader.

Reads API keys from config.json at the project root, so keys can be
configured in one common place instead of being hardcoded in the code.

Priority (first non-empty wins):
  1. Environment variable (e.g. DASHSCOPE_API_KEY) — useful for CI/containers.
  2. config.json -> {"api_keys": {"dashscope": "...", "openai": "..."}}
"""

import json
import os
from functools import lru_cache
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


@lru_cache(maxsize=1)
def _load_config() -> dict:
    """Load config.json once; return {} if missing or invalid."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def get_api_key(name: str, *env_vars: str) -> str:
    """Return an API key by config name, falling back to env vars.

    Checks each env var first, then config.json's "api_keys" section.
    Returns "" when nothing is configured.
    """
    for var in env_vars:
        val = os.environ.get(var, "").strip()
        if val:
            return val
    keys = _load_config().get("api_keys", {})
    if isinstance(keys, dict):
        return str(keys.get(name, "") or "").strip()
    return ""


def get_qwen_api_key() -> str:
    """DashScope/Qwen key: DASHSCOPE_API_KEY / QWEN_API_KEY env, or config.json."""
    return get_api_key("dashscope", "DASHSCOPE_API_KEY", "QWEN_API_KEY")


def get_openai_api_key() -> str:
    """OpenAI key: OPENAI_API_KEY env, or config.json."""
    return get_api_key("openai", "OPENAI_API_KEY")
