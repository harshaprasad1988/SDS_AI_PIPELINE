"""═══════════════════════════════════════════════════════════════════
RAG QUALITY TEST SUITE — DeepEval framework
═══════════════════════════════════════════════════════════════════

Standalone pytest suite that runs the REAL project pipeline modules
(Step 7 RAG engine + Step 8 DeepEval metrics) over an SDS document and
scores every question with LLM-as-judge metrics:

    * Faithfulness        — answer grounded in retrieved SDS passages?
    * Answer Relevancy    — answer actually addresses the question?
    * Contextual Recall   — retrieval pulled what's needed to answer?
    * Regulatory Accuracy — GEval criterion for CAS numbers / % limits.

HOW TO RUN
----------
1. Configure your OpenRouter API key (one of):
       - config.json  -> {"api_keys": {"openrouter": "sk-or-..."}}
       - export OPENROUTER_API_KEY="sk-or-..."

2. Run either way:
       pytest test_rag_quality.py -v
       deepeval test run test_rag_quality.py --verbose

The judge model defaults to the one configured in config.json
(openrouter.llm_model); override with env DEEPEVAL_JUDGE_MODEL.

If no API key is configured, tests SKIP gracefully (no hard failures),
so CI stays green while keys are being set up.
"""

import os
import sys

import pytest

# Make sure project root is importable when run via `deepeval test run` too
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.config import get_openrouter_api_key, get_openrouter_llm_model
from modules.step7_rag import SdsRagEngine
from modules.step8_deepeval import evaluate_rag_answer, METRIC_INFO

# ── Fixtures ────────────────────────────────────────────────────────────────

# Minimal but realistic SDS excerpt used when no Streamlit session state is
# available (i.e. running this file standalone). Mirrors the shape of the
# text produced by Steps 1–3 of the pipeline.
SAMPLE_SDS_TEXT = """
SAFETY DATA SHEET — PRODUCT: CircuitBoard Flux Cleaner FC-200
Section 3: Composition / Information on Ingredients
Substance: 4-Methoxyphenol (MEHQ). CAS No: 150-75-4. Concentration: 2.5 %.
Substance: Lead (Pb). CAS No: 7439-92-1. Concentration: 0.012 % (1200 ppm).
Substance: Mercury (Hg). CAS No: 7439-97-6. Concentration: < 5 ppm.
Substance: Hexavalent chromium (Cr VI). CAS No: 18540-29-9. Concentration: 0.0008 %.
Substance: Decabromodiphenyl ether (Deca-BDE). CAS No: 1163-19-5. Concentration: 0.02 %.
Section 15: Regulatory Information
RoHS Directive 2011/65/EU Annex II limit for Lead: 0.1 % (1000 ppm) in homogeneous materials.
REACH SVHC list includes Lead monoxide; this article contains lead below 0.1 % w/w.
Section 11: Toxicological Information
MEHQ causes skin irritation; LD50 (oral, rat) 300 mg/kg.
"""

QUESTIONS = [
    "What is the concentration of lead in this product and does it exceed the RoHS limit?",
    "List all substances with their CAS numbers.",
    "Is mercury present above 5 ppm?",
]


@pytest.fixture(scope="module")
def rag_engine():
    """Build the RAG index from the uploaded SDS if a Streamlit session is
    running; otherwise fall back to the bundled sample SDS text."""
    try:  # auto-detect Streamlit session state (running inside the app)
        import streamlit as st
        sds_text = (st.session_state.get("ocr_full_text")
                    or st.session_state.get("full_text") or "")
        mds = (st.session_state.get("structured_mds")
               or st.session_state.get("mds") or [])
        if str(sds_text).strip():
            return SdsRagEngine(sds_text, structured_mds=mds)
    except Exception:
        pass
    return SdsRagEngine(SAMPLE_SDS_TEXT, structured_mds=[])


@pytest.fixture(scope="module")
def judge_model() -> str:
    return os.environ.get("DEEPEVAL_JUDGE_MODEL", "") or get_openrouter_llm_model()


@pytest.fixture(autouse=True)
def require_api_key(request):
    """Skip LLM-judge tests when no OpenRouter key is configured.

    Tests explicitly marked with @pytest.mark.llm (or the DeepEval
    end-to-end test) need a key; deterministic tests such as the TF-IDF
    retriever check run regardless, so retrieval quality can be verified
    even without an API key.
    """
    needs_key = request.node.get_closest_marker("llm") is not None
    if needs_key and not get_openrouter_api_key():
        pytest.skip("OpenRouter API key not configured "
                    "(set config.json api_keys.openrouter or "
                    "export OPENROUTER_API_KEY)")


# ── Tests ───────────────────────────────────────────────────────────────────

@pytest.mark.llm
@pytest.mark.parametrize("question", QUESTIONS)
def test_rag_answer_faithful_and_relevant(question, rag_engine, judge_model):
    """End-to-end: retrieve → generate → DeepEval metrics must pass."""
    result = rag_engine.answer(question, model=judge_model, top_k=4)
    answer = result.answer
    passages = [h.text for h in result.sources]

    assert answer, "RAG engine returned an empty answer"
    assert passages, "RAG retriever found no relevant passages"

    eval_out = evaluate_rag_answer(
        question=question,
        answer=answer,
        retrieved_passages=passages,
        judge_model=judge_model,
        threshold=float(os.environ.get("DEEPEVAL_THRESHOLD", "0.5")),
    )

    assert eval_out["ok"], f"DeepEval run failed: {eval_out['error']}"

    scores = {r["metric"]: r for r in eval_out["results"]}
    # Core RAG metrics must be present and passing
    for metric in ("Faithfulness", "Answer Relevancy", "Contextual Recall"):
        assert metric in scores, f"Missing metric: {metric} ({METRIC_INFO.get(metric)})"
        row = scores[metric]
        assert row["passed"], (
            f"{metric} failed for Q: {question!r}\n"
            f"  score={row['score']:.2f} (threshold {row['threshold']})\n"
            f"  reason: {row.get('reason', 'n/a')[:300]}"
        )


def test_retriever_finds_lead_content(rag_engine):
    """Deterministic check (no LLM needed): TF-IDF retrieval quality."""
    hits = rag_engine.retrieve("lead concentration CAS RoHS limit", k=3)
    assert hits, "retriever returned nothing"
    joined = " ".join(h.text.lower() for h in hits)
    assert "lead" in joined or "7439-92-1" in joined, \
        "expected lead-related passage in top-3 results"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
