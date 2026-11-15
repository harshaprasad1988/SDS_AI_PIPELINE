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


def get_openrouter_api_key() -> str:
    """OpenRouter key (used for Qwen, Phi-4, Mistral Small and any other
    OpenRouter-hosted model): OPENROUTER_API_KEY env, or config.json.

    Falls back to a legacy DashScope/Qwen key if no OpenRouter key is set.
    """
    return get_api_key("openrouter", "OPENROUTER_API_KEY",
                       "DASHSCOPE_API_KEY", "QWEN_API_KEY") \
        or get_api_key("dashscope", "DASHSCOPE_API_KEY", "QWEN_API_KEY")


# Backwards-compatible alias (older code / OCR module may still import this)
get_qwen_api_key = get_openrouter_api_key


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
    return "qwen/qwen3-235b-a22b-2507"


# OpenRouter model IDs that were retired/renamed and now fail with
# "<id> is not a valid model ID". Old config.json files or saved selections
# that still reference them are transparently mapped to a working model.
RETIRED_OPENROUTER_MODELS = {
    "qwen/qwen3-235b-a22b": "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen-vl-max": "qwen/qwen3-vl-235b-a22b-instruct",
    "qwen/qwen3-vl-32b-instruct": "qwen/qwen3-vl-30b-a3b-instruct",
    "qwen/qwen3-coder-plus": "qwen/qwen3-coder",
    "qwen/qwen-turbo": "qwen/qwen3-30b-a3b-instruct-2507",
    "moonshotai/kimi-k2-thinking-turbo": "moonshotai/kimi-k2-thinking",
    "mistralai/mistral-small-latest": "mistralai/mistral-small-3.2-24b-instruct",
}


def resolve_openrouter_model(model: str) -> str:
    """Map a retired OpenRouter model ID to its current replacement."""
    model = (model or "").strip()
    return RETIRED_OPENROUTER_MODELS.get(model, model)


# ── Model catalogue (single source of truth for the UI and checks) ──────────
# Every ID below was verified against https://openrouter.ai/api/v1/models.
# Run `python check_openrouter_models.py` to re-check them.
OPENROUTER_OCR_MODELS = [
    "qwen/qwen2.5-vl-72b-instruct",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "qwen/qwen3-vl-30b-a3b-instruct",
]

OPENROUTER_LLM_MODELS = [
    # Moonshot AI — Kimi
    "moonshotai/kimi-k2.5",
    "moonshotai/kimi-k2-thinking",
    # Qwen
    "qwen/qwen3-coder",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-plus",
    "qwen/qwen3-30b-a3b-instruct-2507",
    # Microsoft
    "microsoft/phi-4",
    # Mistral
    "mistralai/mistral-small-3.2-24b-instruct",
    # DeepSeek
    "deepseek/deepseek-v4-flash",
    # OpenAI (via OpenRouter)
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
]
