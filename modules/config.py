"""Central configuration loader.

Reads API keys and provider settings from config.json at the project root,
so they can be configured in one common place instead of being hardcoded
in the code.

Priority (first non-empty wins):
  1. Environment variable (e.g. OPENROUTER_API_KEY) — useful for CI/containers.
  2. config.json -> {"api_keys": {"openrouter": "...", "openai": "..."},
                     "openrouter": {"base_url": "...", "ocr_model": "...", ...}}
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
    """Qwen key served via OpenRouter: OPENROUTER_API_KEY env, or config.json.

    Falls back to a legacy DashScope key if no OpenRouter key is configured.
    """
    return get_api_key("openrouter", "OPENROUTER_API_KEY",
                       "DASHSCOPE_API_KEY", "QWEN_API_KEY") \
        or get_api_key("dashscope", "DASHSCOPE_API_KEY", "QWEN_API_KEY")


def get_openai_api_key() -> str:
    """OpenAI key: OPENAI_API_KEY env, or config.json."""
    return get_api_key("openai", "OPENAI_API_KEY")


def get_openrouter_base_url() -> str:
    """OpenRouter OpenAI-compatible base URL (configurable in config.json)."""
    section = _load_config().get("openrouter", {})
    if isinstance(section, dict):
        url = str(section.get("base_url", "") or "").strip()
        if url:
            return url
    return os.environ.get("OPENROUTER_BASE_URL",
                          "https://openrouter.ai/api/v1")


def get_openrouter_ocr_model() -> str:
    """Default Qwen vision model used for OCR via OpenRouter."""
    section = _load_config().get("openrouter", {})
    if isinstance(section, dict):
        model = str(section.get("ocr_model", "") or "").strip()
        if model:
            return model
    return "qwen/qwen2.5-vl-72b-instruct"


def get_openrouter_llm_model() -> str:
    """Default text LLM (Qwen on OpenRouter) used for Step 5 recommendations."""
    section = _load_config().get("openrouter", {})
    if isinstance(section, dict):
        model = str(section.get("llm_model", "") or "").strip()
        if model:
            return model
    return "qwen/qwen3-235b-a22b"
