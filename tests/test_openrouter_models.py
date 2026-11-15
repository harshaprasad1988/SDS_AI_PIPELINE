"""Guards against OpenRouter model IDs that were retired (they fail with
'<id> is not a valid model ID')."""
import re
from pathlib import Path

from modules.config import RETIRED_OPENROUTER_MODELS, resolve_openrouter_model

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / "streamlit_app.py", ROOT / "config.json",
           *(p for p in (ROOT / "modules").glob("*.py") if p.name != "config.py")]


def test_no_retired_model_ids_in_code():
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for retired in RETIRED_OPENROUTER_MODELS:
            assert not re.search(rf'"{re.escape(retired)}"', text), \
                f"{path.name} still uses retired model {retired}"


def test_retired_ids_are_remapped():
    assert resolve_openrouter_model("qwen/qwen3-235b-a22b") == "qwen/qwen3-235b-a22b-2507"
    assert resolve_openrouter_model("mistralai/mistral-small-latest") == \
        "mistralai/mistral-small-3.2-24b-instruct"
    assert resolve_openrouter_model("openai/gpt-4o-mini") == "openai/gpt-4o-mini"


def test_call_openrouter_handles_none_content(monkeypatch):
    import types, sys
    from modules import step6_llm

    captured = {}

    class FakeClient:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=self)

        def create(self, **kw):
            captured.update(kw)
            msg = types.SimpleNamespace(content=None, reasoning="answer from reasoning")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeClient))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    out = step6_llm.call_openrouter("q", "sys", model="qwen/qwen3-235b-a22b")
    assert out == "answer from reasoning"
    assert captured["model"] == "qwen/qwen3-235b-a22b-2507"


def test_model_catalogue_has_no_retired_ids():
    from modules.config import OPENROUTER_LLM_MODELS, OPENROUTER_OCR_MODELS
    for m in OPENROUTER_LLM_MODELS + OPENROUTER_OCR_MODELS:
        assert m not in RETIRED_OPENROUTER_MODELS, m
