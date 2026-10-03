"""
═══════════════════════════════════════════════════════════════════
STEP 8 — DEEPEVAL RAG QUALITY EVALUATION
═══════════════════════════════════════════════════════════════════

PURPOSE:
    Adopt the DeepEval framework (https://github.com/confident-ai/deepeval)
    to score the quality of the Step-7 RAG answers over the SDS sheet.

HOW IT WORKS:
    * Each user question + generated answer + retrieved passages is
      wrapped in a DeepEval ``LLMTestCase``.
    * LLM-as-judge metrics are measured with a custom
      ``DeepEvalBaseLLM`` adapter that routes evaluation calls through
      OpenRouter (the same API key / base URL configured in config.json),
      so no separate OpenAI key is required for the judge.
    * Metrics produced:
        - Faithfulness          : is the answer grounded in the retrieved
                                  SDS passages (hallucination check)?
        - Answer Relevancy      : does the answer actually address the
                                  question?
        - Contextual Recall     : do the retrieved passages contain the
                                  information needed to answer?
        - GEval "Regulatory Accuracy": custom criterion checking numeric
                                  limits / CAS numbers / regulatory claims
                                  against the retrieved SDS context.

The results are returned as plain dicts so the Streamlit UI can render
them without knowing anything about DeepEval internals.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Metric display order + short descriptions used by the UI
METRIC_INFO = {
    "Faithfulness":
        "Claims in the answer vs. the retrieved SDS passages (hallucination check).",
    "Answer Relevancy":
        "How relevant the generated answer is to the user's question.",
    "Contextual Recall":
        "Did retrieval pull the passages needed to fully answer the question?",
    "Regulatory Accuracy":
        "GEval criterion: are CAS numbers, % concentrations and limit values "
        "correct w.r.t. the retrieved SDS context?",
}


# ── OpenRouter-backed DeepEval judge model ───────────────────────────────────

def _build_or_model(model_name: str):
    """Return a DeepEvalBaseLLM whose generate()/a_generate() call any
    OpenRouter-hosted model (Qwen / Phi-4 / Mistral Small / DeepSeek /
    OpenAI GPT …) using the project's shared config loader."""
    from deepeval.models.base_model import DeepEvalBaseLLM

    class OpenRouterJudgeModel(DeepEvalBaseLLM):
        def __init__(self, model: str):
            super().__init__(model)

        def load_model(self):
            return self

        def _call(self, prompt: Optional[str] = None,
                  messages=None, schema=None, **kwargs) -> str:
            from modules.step6_llm import call_openrouter
            # DeepEval metrics pass a Pydantic `schema` they expect back as
            # JSON. Many OpenRouter models don't support native structured
            # output reliably, so we inject the JSON schema into the prompt
            # and ask for plain JSON — deepeval parses the raw text either way.
            if schema is not None:
                try:
                    schema_json = json.dumps(schema.model_json_schema(),
                                             ensure_ascii=False)
                    schema_hint = ("\n\nRespond ONLY with a valid JSON object "
                                   "matching this JSON schema (no markdown, no "
                                   f"extra text):\n{schema_json}")
                except Exception:
                    schema_hint = "\n\nRespond ONLY with a valid JSON object."
                if messages:
                    messages = [dict(m) if isinstance(m, dict) else m
                                for m in messages]
                    last_user = next((i for i in range(len(messages) - 1, -1, -1)
                                      if (messages[i].get("role")
                                          if isinstance(messages[i], dict)
                                          else messages[i].role) == "user"), None)
                    if last_user is not None:
                        c = (messages[last_user].get("content")
                             if isinstance(messages[last_user], dict)
                             else messages[last_user].content)
                        newc = (c or "") + schema_hint
                        if isinstance(messages[last_user], dict):
                            messages[last_user]["content"] = newc
                        else:
                            messages[last_user].content = newc
                elif prompt is not None:
                    prompt = (prompt or "") + schema_hint
            if messages:
                # Normalise to (system, prompt) form expected by our helper
                system = "You are a strict evaluation assistant."
                parts = []
                for m in messages:
                    role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
                    content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                    if role == "system":
                        system = content
                    else:
                        parts.append(content)
                return call_openrouter("\n\n".join(parts), system,
                                       model=self.model_name_str,
                                       temperature=0.0, max_tokens=2048)
            return call_openrouter(prompt or "",
                                   "You are a strict evaluation assistant.",
                                   model=self.model_name_str,
                                   temperature=0.0, max_tokens=2048)

        @property
        def model_name_str(self) -> str:
            # `self.model` is set by __init__ via load_model(); we keep the
            # requested id on the instance instead.
            return getattr(self, "_model_id", "qwen/qwen3-235b-a22b")

        def generate(self, *args, **kwargs) -> str:
            return self._call(*args, **kwargs)

        async def a_generate(self, *args, **kwargs) -> str:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self._call(*args, **kwargs))

        def get_model_name(self) -> str:
            return f"openrouter/{self.model_name_str}"

    m = OpenRouterJudgeModel(model_name)
    m._model_id = model_name
    return m


# ── Evaluation runner ────────────────────────────────────────────────────────

def evaluate_rag_answer(question: str, answer: str,
                        retrieved_passages: List[str],
                        judge_model: str,
                        threshold: float = 0.5) -> Dict:
    """Run DeepEval metrics over one RAG Q&A pair.

    Returns::
        {
          "ok": bool, "error": str|None, "judge": "openrouter/<model>",
          "results": [ {"metric","score","threshold","passed","reason"}, ... ]
        }
    """
    out = {"ok": False, "error": None, "results": [],
           "judge": f"openrouter/{judge_model}"}
    try:
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import (FaithfulnessMetric,
                                      AnswerRelevancyMetric,
                                      ContextualRecallMetric, GEval)
        try:  # newer deepeval renamed this enum
            from deepeval.test_case import SingleTurnParams as TP
        except ImportError:
            from deepeval.test_case import LLMTestCaseParams as TP
    except ImportError as exc:
        out["error"] = ("deepeval package not installed. "
                        "Run: pip install deepeval")
        logger.warning(out["error"])
        return out

    if not answer or not retrieved_passages:
        out["error"] = "Nothing to evaluate (empty answer or no retrieved passages)."
        return out

    from .config import get_openrouter_api_key
    if not get_openrouter_api_key():
        out["error"] = ("OpenRouter API key not configured — add it to "
                        "config.json (api_keys.openrouter) to run DeepEval "
                        "LLM-as-judge metrics.")
        return out

    tc = LLMTestCase(
        input=question,
        actual_output=answer,
        # Contextual Recall requires an expected_output reference; the SDS
        # passages themselves act as the gold reference for this document QA.
        expected_output=" ".join(p for p in retrieved_passages if p and p.strip())[:1500],
        retrieval_context=[p for p in retrieved_passages if p and p.strip()],
    )

    model = _build_or_model(judge_model)

    metrics = [
        ("Faithfulness",       FaithfulnessMetric(model=model, threshold=threshold,
                                                  async_mode=False)),
        ("Answer Relevancy",   AnswerRelevancyMetric(model=model, threshold=threshold,
                                                     async_mode=False)),
        ("Contextual Recall",  ContextualRecallMetric(model=model, threshold=threshold,
                                                      async_mode=False)),
        ("Regulatory Accuracy", GEval(
            name="Regulatory Accuracy",
            criteria=("The answer must state CAS numbers, weight percentages, "
                      "regulatory limit values and section references exactly "
                      "as supported by the retrieved SDS context. Any number, "
                      "CAS or regulation cited that contradicts or is absent "
                      "from the context lowers the score."),
            evaluation_params=[TP.INPUT, TP.ACTUAL_OUTPUT, TP.RETRIEVAL_CONTEXT],
            model=model, threshold=threshold, async_mode=False)),
    ]

    results = []
    errors = []
    for name, metric in metrics:
        try:
            metric.measure(tc)
            score = float(metric.score or 0.0)
            results.append({
                "metric": name,
                "score": round(score, 3),
                "threshold": threshold,
                "passed": score >= threshold,
                "reason": (getattr(metric, "reason", "") or "").strip(),
            })
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            logger.warning(f"DeepEval metric {name} failed: {exc}")

    out["results"] = results
    out["ok"] = bool(results)
    if errors and not results:
        out["error"] = "; ".join(errors[:2])
    elif errors:
        out["partial_errors"] = errors
    return out
