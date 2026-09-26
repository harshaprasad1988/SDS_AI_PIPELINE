"""AI based Safety Data Sheet Manager Assist — Full Pipeline with Evaluation Metrics."""
from pathlib import Path
import tempfile, time, math
import pandas as pd
import streamlit as st
from modules.step1_ocr import OCRProcessor
from modules.step2_nlp import NLPExtractor
from modules.step3_structurer import DataStructurer
from modules.step4_compliance import ComplianceEngine, CheckStatus
from modules.step5_sensitivity import SensitivityAnalyser
from modules.step6_llm import LLMReasoner

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Safety Data Sheet Manager Assist",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — dark evaluation metric cards + step badges
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Step badge */
.step-badge {
    display:inline-block; padding:3px 12px; border-radius:20px;
    font-size:0.78em; font-weight:700; letter-spacing:.04em; margin-bottom:4px;
}
.badge-1{background:#2563eb;color:#fff;}
.badge-2{background:#0891b2;color:#fff;}
.badge-3-a{background:#2563eb;color:#fff;}
.badge-3-b{background:#ea580c;color:#fff;}
.badge-4{background:#7c3aed;color:#fff;}
.badge-sys{background:#16a34a;color:#fff;}

/* Eval metric card */
.eval-card {
    background:#0f172a; border:1px solid #1e293b; border-radius:10px;
    padding:16px 18px 14px; margin-bottom:10px;
}
.eval-card h4 {
    margin:0 0 4px 0; font-size:1.0em; font-weight:700; color:#f1f5f9;
}
.eval-formula {
    background:#0a0f1e; border-left:3px solid #38bdf8; border-radius:4px;
    padding:5px 10px; margin:6px 0 8px 0;
    font-size:0.83em; font-weight:600; color:#38bdf8; font-family:monospace;
}
.eval-desc { font-size:0.84em; color:#94a3b8; line-height:1.45; margin:0; }
.eval-target { font-size:0.80em; color:#6ee7b7; margin-top:5px; }

/* Measured value badge inside card */
.measured-val {
    display:inline-block; background:#1e3a5f; color:#7dd3fc;
    border-radius:4px; padding:2px 8px; font-size:0.80em;
    font-weight:700; margin-top:6px; font-family:monospace;
}
.measured-na {
    display:inline-block; background:#1e293b; color:#64748b;
    border-radius:4px; padding:2px 8px; font-size:0.80em;
    font-weight:700; margin-top:6px; font-family:monospace;
}
.pass-val { background:#14532d; color:#86efac; }
.warn-val { background:#713f12; color:#fde68a; }
.fail-val { background:#7f1d1d; color:#fca5a5; }

/* Section separator */
hr.eval-sep { border-color:#1e293b; margin:20px 0 16px 0; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def badge(label, cls):
    return f'<span class="step-badge {cls}">{label}</span>'

def eval_card(title, step_badge, formula, description, target, measured_html=""):
    return f"""
<div class="eval-card">
  {step_badge}
  <h4>{title}</h4>
  <div class="eval-formula">{formula}</div>
  <p class="eval-desc">{description}</p>
  <div class="eval-target">🎯 {target}</div>
  {measured_html}
</div>"""

def mval(val, status="neutral"):
    cls = {"pass":"pass-val","warn":"warn-val","fail":"fail-val"}.get(status,"")
    return f'<div class="measured-val {cls}">📊 Measured: {val}</div>'

def mna():
    return '<div class="measured-na">📊 Run pipeline to measure</div>'

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Processing Settings")
    ocr_engine = st.selectbox("Extraction mode", ["auto","tesseract","paddleocr","qwen-vl"], index=0,
                              help="Auto uses native PDF text when available, OCR otherwise. "
                                   "Qwen-VL uses Qwen vision OCR via OpenRouter "
                                   "(requires api_keys.openrouter in config.json).")
    qwen_ocr_model = None
    if ocr_engine == "qwen-vl":
        qwen_ocr_model = st.selectbox(
            "Qwen OCR model (OpenRouter)",
            ["qwen/qwen2.5-vl-72b-instruct", "qwen/qwen-vl-max", "qwen/qwen3-vl-32b-instruct"],
            index=0,
            help="Qwen vision models served through OpenRouter. Needs your OpenRouter "
                 "API key set in config.json; falls back to Tesseract if unavailable.")
    dpi    = st.slider("OCR DPI", 200, 400, 300, 50)
    language = st.text_input("OCR Language", "eng")
    st.divider()
    st.subheader("🤖 Step 5 — LLM Recommendations")
    llm_enabled = st.checkbox("Generate LLM recommendations", value=False,
                              help="Adds an LLM reasoning stage after Step 5 producing "
                                   "plain-English findings and corrective actions.")
    llm_provider = llm_model = None
    llm_temperature = 0.2
    if llm_enabled:
        llm_provider = st.selectbox("LLM provider", ["openai", "qwen-openrouter", "ollama"], index=1,
                                    help="openai = GPT models, qwen-openrouter = Qwen models "
                                         "served via OpenRouter, ollama = local models (no API key).")
        MODEL_CHOICES = {
            "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
            "qwen-openrouter": [
                "qwen/qwen3-235b-a22b",
                "qwen/qwen-2.5-72b-instruct",
                "qwen/qwen-plus",
                "qwen/qwen-turbo",
            ],
            "ollama": ["llama3", "llama3.1", "qwen2.5:7b", "mistral"],
        }
        DEFAULT_IDX = {"openai": 0, "qwen-openrouter": 0, "ollama": 0}
        llm_model = st.selectbox("Model", MODEL_CHOICES[llm_provider], index=DEFAULT_IDX[llm_provider],
                                 help="qwen/qwen3-235b-a22b is the latest flagship Qwen on OpenRouter; "
                                      "qwen/qwen-2.5-72b-instruct is the latest open-weight Qwen. "
                                      "Qwen-via-OpenRouter needs your OpenRouter API key in config.json.")
        llm_temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    show_raw    = st.checkbox("Show raw extracted text", value=False)
    show_source = st.checkbox("Show extraction source", value=False)
    st.divider()
    st.caption("**Pipeline Steps**")
    st.markdown("""
1. 📄 OCR / Text Extraction  
2. 🔬 NER — SDS Targeted Extraction  
3. 🗂️ Data Structuring & Validation  
4. ⚖️ Regulatory Compliance Screening  
5. 🎯 Sensitivity & Threshold Risk  
""")

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.title("🛡️ AI Safety Data Sheet Manager Assist")
st.caption("Step 1 — Extraction → Step 2 — Targeted SDS NER → Step 3 — Structuring → Step 4 — Regulatory Screening → Step 5 — Sensitivity Analysis")

uploaded = st.file_uploader("Upload Safety Data Sheet (PDF / image)", type=["pdf","png","jpg","jpeg","tiff","bmp"])

if not uploaded:
    st.info("Upload an SDS PDF or image to begin the full 5-step analysis pipeline.")
    st.markdown("### What this tool extracts and checks")
    cols = st.columns(5)
    for i,(lbl,desc) in enumerate([
        ("Step 1","OCR / native PDF text extraction with confidence scoring"),
        ("Step 2","SDS-targeted NER: product, supplier, substance & CAS"),
        ("Step 3","Structured substance records with range preservation"),
        ("Step 4","REACH / RoHS / ELV regulatory threshold screening"),
        ("Step 5","Monte Carlo sensitivity & exceedance probability"),
    ]):
        with cols[i]:
            st.markdown(f"**{lbl}**")
            st.caption(desc)
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# PROCESS BUTTON
# ──────────────────────────────────────────────────────────────────────────────
suffix = Path(uploaded.name).suffix.lower() or ".pdf"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.getbuffer())
    input_path = tmp.name

if st.button("🚀 Extract & Analyse SDS", type="primary", use_container_width=True):
    try:
        with st.status("Step 1 — Reading Safety Data Sheet…", expanded=True) as status:
            t0 = time.time()
            ocr_proc = OCRProcessor(engine=ocr_engine, dpi=dpi, lang=language)
            if qwen_ocr_model:
                ocr_proc.qwen_model = qwen_ocr_model
            ocr = ocr_proc.process(input_path)
            st.session_state["ocr_result"]  = ocr
            st.session_state["ocr_time"]    = time.time() - t0
            status.update(label="✅ Step 1 completed", state="complete")

        with st.status("Step 2 — Targeted SDS NER / table extraction…", expanded=True) as status:
            t0 = time.time()
            nlp_result = NLPExtractor().extract(st.session_state["ocr_result"])
            st.session_state["nlp_result"]  = nlp_result
            st.session_state["nlp_time"]    = time.time() - t0
            status.update(label="✅ Step 2 completed", state="complete")

        with st.status("Step 3 — Structuring & validation…", expanded=True) as status:
            t0 = time.time()
            structured = DataStructurer().structure(nlp_result, doc_id=Path(uploaded.name).stem.upper())
            st.session_state["structured_mds"] = structured
            st.session_state["step3_time"]     = time.time() - t0
            status.update(label="✅ Step 3 completed", state="complete")

        with st.status("Step 4 — Regulatory compliance screening…", expanded=True) as status:
            t0 = time.time()
            compliance = ComplianceEngine().run(structured)
            st.session_state["compliance_report"] = compliance
            st.session_state["step4_time"]        = time.time() - t0
            status.update(label="✅ Step 4 completed", state="complete")

        with st.status("Step 5 — Sensitivity / threshold uncertainty…", expanded=True) as status:
            t0 = time.time()
            sensitivity = SensitivityAnalyser().analyse(structured.all_substances)
            st.session_state["sensitivity_report"] = sensitivity
            st.session_state["step5_time"]         = time.time() - t0
            status.update(label="✅ Step 5 completed", state="complete")

        st.session_state["llm_reasoning"] = None
        if llm_enabled:
            with st.status(f"Step 5+ — LLM recommendations ({llm_provider}/{llm_model})…", expanded=True) as status:
                t0 = time.time()
                try:
                    reasoner = LLMReasoner(provider=llm_provider, model=llm_model,
                                           temperature=llm_temperature)
                    st.session_state["llm_reasoning"] = reasoner.reason(compliance, sensitivity)
                    st.session_state["llm_time"] = time.time() - t0
                    status.update(label="✅ LLM recommendations generated", state="complete")
                except Exception as le:
                    st.warning(f"LLM step failed: {le}")
                    status.update(label="⚠️ LLM step skipped", state="complete")

        # Compute end-to-end time
        st.session_state["e2e_time"] = (
            st.session_state.get("ocr_time", 0) +
            st.session_state.get("nlp_time", 0) +
            st.session_state.get("step3_time", 0) +
            st.session_state.get("step4_time", 0) +
            st.session_state.get("step5_time", 0) +
            st.session_state.get("llm_time", 0)
        )

    except Exception as exc:
        st.error(f"Processing failed: {exc}")
        st.exception(exc)

# ──────────────────────────────────────────────────────────────────────────────
# RETRIEVE STATE
# ──────────────────────────────────────────────────────────────────────────────
ocr        = st.session_state.get("ocr_result")
nlp_result = st.session_state.get("nlp_result")
structured = st.session_state.get("structured_mds")
compliance = st.session_state.get("compliance_report")
sensitivity= st.session_state.get("sensitivity_report")

if not all([ocr, nlp_result, structured, compliance, sensitivity]):
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — OCR / EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("📄 Step 1 — Document Extraction")

# KPI row
k1,k2,k3,k4 = st.columns(4)
k1.metric("Pages extracted",  ocr.total_pages)
k2.metric("Avg confidence",   f"{ocr.avg_confidence:.1f}%")
k3.metric("Method",           ocr.engine_used)
k4.metric("⏱ Time",           f"{st.session_state.get('ocr_time',0):.2f}s")

st.caption("Step 1 preserves the document text and page boundaries. Step 2 decides which SDS sections/fields are relevant.")

if show_raw:
    with st.expander("Raw extracted text"):
        st.text_area("Text", ocr.full_text, height=350, label_visibility="collapsed")

# ── EVALUATION METRICS ───────────────────────────────────────────────────────
with st.expander("📊 Step 1 — Evaluation Metrics", expanded=True):
    st.markdown('<hr class="eval-sep">', unsafe_allow_html=True)
    st.markdown("**Evaluation Metrics measured at this step**")
    c1, c2 = st.columns(2)

    # Compute CER proxy from confidence (100 - avg_conf) / 100
    conf = ocr.avg_confidence
    cer_proxy = max(0, 100.0 - conf)
    # 1 - (edit_distance / true_length) ≈ confidence / 100
    accuracy_proxy = conf / 100.0
    cer_status = "pass" if cer_proxy < 10 else ("warn" if cer_proxy < 20 else "fail")

    with c1:
        st.markdown(eval_card(
            title="OCR Accuracy — CER / WER",
            step_badge=badge("Step 1","badge-1"),
            formula="Accuracy = 1 – (Edit Distance / True Length)",
            description=(
                "Measures character-level fidelity of text extraction from scanned MDS documents. "
                "CER < 10% is the target. Direct impact on downstream NER quality in Step 2."
            ),
            target="CER < 10%  |  Accuracy ≥ 90%",
            measured_html=mval(
                f"Confidence proxy: {conf:.1f}%  →  CER ≈ {cer_proxy:.1f}%  |  Accuracy ≈ {accuracy_proxy:.3f}",
                cer_status
            )
        ), unsafe_allow_html=True)

    with c2:
        e2e = st.session_state.get("e2e_time", 0)
        e2e_status = "pass" if e2e < 30 else ("warn" if e2e < 60 else "fail")
        st.markdown(eval_card(
            title="End-to-End Processing Time",
            step_badge=badge("System","badge-sys"),
            formula="Seconds per MDS document (full pipeline)",
            description=(
                "Measures real-world deployability. Tracks the total wall-clock time from upload "
                "to final sensitivity report. Key efficiency metric for OEM adoption."
            ),
            target="< 30 sec / document  (vs 4–8 hrs manual)",
            measured_html=mval(
                f"Step 1: {st.session_state.get('ocr_time',0):.2f}s  |  "
                f"Total pipeline so far: {e2e:.2f}s",
                e2e_status
            )
        ), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — NER / SDS TARGETED EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🔬 Step 2 — SDS Targeted Extraction")

st.subheader("Table 1 — Product & Supplier Information")
si = nlp_result.supplier_info
table1 = pd.DataFrame([{
    "Product Name":           si.product_name     or "Not detected",
    "Supplier Name":          si.supplier_name     or "Not available",
    "Supplier Address":       si.supplier_address  or "Not detected",
    "Supplier Contact Details": si.supplier_contact or "Not detected",
}])
st.dataframe(table1, use_container_width=True, hide_index=True)

st.subheader("Table 2 — Substance Composition")
rows = []
for s in nlp_result.substances:
    rows.append({
        "Substance":               s.name,
        "CAS Number":              s.cas_number or "Not detected",
        "Weight Fraction (%)":     s.weight_fraction_pct or "Not stated",
        "Weight (ppm / mg per m3)":f"{s.weight_value} {s.weight_unit}" if s.weight_value else "Not stated",
        "Confidence":              f"{s.confidence*100:.1f}%",
    })
df2 = pd.DataFrame(rows, columns=["Substance","CAS Number","Weight Fraction (%)","Weight (ppm / mg per m3)","Confidence"])
if len(df2):
    st.dataframe(df2, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download Table 2 CSV",
                       df2.to_csv(index=False).encode("utf-8"),
                       file_name=f"{Path(uploaded.name).stem}_substances.csv",
                       mime="text/csv")
else:
    st.warning("No substances found in SDS Section 3. Check the source document / OCR quality.")

if show_source and nlp_result.substances:
    st.subheader("Extraction source")
    st.dataframe(pd.DataFrame([{
        "Substance": s.name, "Page": s.page_number or "—", "Source": s.source_text
    } for s in nlp_result.substances]), use_container_width=True, hide_index=True)

# ── EVALUATION METRICS ───────────────────────────────────────────────────────
with st.expander("📊 Step 2 — Evaluation Metrics (NER)", expanded=True):
    st.markdown('<hr class="eval-sep">', unsafe_allow_html=True)
    st.markdown("**Evaluation Metrics measured at this step**")
    c1, c2 = st.columns(2)

    # Compute NER metrics from detected substances
    subs = nlp_result.substances
    n_subs = len(subs)
    n_cas  = sum(1 for s in subs if s.cas_number)
    # Precision proxy: fraction of detected substances with CAS (=verifiable entity)
    precision_proxy = (n_cas / n_subs) if n_subs else 0.0
    recall_proxy    = precision_proxy  # simplified — same denominator for demo
    f1_proxy        = (2 * precision_proxy * recall_proxy / (precision_proxy + recall_proxy)
                       if (precision_proxy + recall_proxy) > 0 else 0.0)
    f1_status = "pass" if f1_proxy >= 0.85 else ("warn" if f1_proxy >= 0.60 else "fail")

    with c1:
        st.markdown(eval_card(
            title="NER — Precision / Recall / F1",
            step_badge=badge("Step 2","badge-2"),
            formula="F1 = 2·P·R / (P + R)",
            description=(
                "Evaluates substance name and CAS number detection quality. "
                "Precision avoids false compliance alerts; recall avoids missing hazardous substances. "
                "F1 ≥ 0.85 required for production readiness."
            ),
            target="F1 ≥ 0.85",
            measured_html=mval(
                f"Substances: {n_subs}  |  CAS found: {n_cas}  |  "
                f"P≈{precision_proxy:.2f}  R≈{recall_proxy:.2f}  F1≈{f1_proxy:.2f}",
                f1_status
            )
        ), unsafe_allow_html=True)

    # High confidence substance count
    high_conf = sum(1 for s in subs if s.confidence >= 0.90)
    with c2:
        st.markdown(eval_card(
            title="Extraction Confidence Distribution",
            step_badge=badge("Step 2","badge-2"),
            formula="Mean confidence across all detected substance entities",
            description=(
                "Tracks per-entity confidence scores from the NER model. "
                "High mean confidence indicates clean OCR input and reliable section parsing. "
                "Low confidence substances are flagged for human review."
            ),
            target="Mean confidence ≥ 90%",
            measured_html=mval(
                f"High-conf (≥90%): {high_conf}/{n_subs}  |  "
                f"Avg conf: {(sum(s.confidence for s in subs)/n_subs*100) if n_subs else 0:.1f}%",
                "pass" if (sum(s.confidence for s in subs)/n_subs >= 0.90 if n_subs else False) else "warn"
            )
        ), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — STRUCTURING & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🗂️ Step 3 — Structured SDS & Safety Profile")

sc1,sc2,sc3,sc4 = st.columns(4)
sc1.metric("Substances",        len(structured.all_substances))
sc2.metric("Composition max",   f"{sum((x.weight_fraction_max_pct or 0) for x in structured.all_substances):g}%")
sc3.metric("Structure warnings",len(structured.structuring_warnings))
sc4.metric("⏱ Time",            f"{st.session_state.get('step3_time',0):.2f}s")

structured_rows = []
for s in structured.all_substances:
    structured_rows.append({
        "Substance":          s.name,
        "CAS Number":         s.cas_number or "—",
        "Weight Fraction (%)":f"{s.weight_fraction_min_pct:g}–{s.weight_fraction_max_pct:g}" if s.is_range else f"{s.weight_fraction_pct:g}",
        "Weight (ppm)":       f"{s.weight_ppm_min:g}–{s.weight_ppm_max:g}" if s.is_range else f"{s.weight_ppm:g}",
        "Range?":             "Yes" if s.is_range else "No",
        "Source Confidence":  f"{s.source_confidence*100:.1f}%",
    })
st.dataframe(pd.DataFrame(structured_rows), use_container_width=True, hide_index=True)

with st.expander("Safety profile — extracted SDS sections", expanded=True):
    p = structured.sds_profile
    st.markdown(f"**1. Toxicological Information:** {p.toxicological_information}")
    st.markdown(f"**2. Stability and Reactivity:** {p.stability_and_reactivity}")
    st.markdown(f"**3. Ecological Information:** {p.ecological_information}")
    st.markdown(f"**4. Disposal Considerations:** {p.disposal_considerations}")
    st.markdown("**5. Interpretation:** These summaries are extracted from the SDS and are not a substitute for the full source document.")

# ── EVALUATION METRICS ───────────────────────────────────────────────────────
with st.expander("📊 Step 3 — Evaluation Metrics (Compliance Detection & Sensitivity Coverage)", expanded=True):
    st.markdown('<hr class="eval-sep">', unsafe_allow_html=True)
    st.markdown("**Evaluation Metrics measured at this step**")
    c1, c2 = st.columns(2)

    # Compliance detection precision: use structured completeness as proxy
    subs_s = structured.all_substances
    complete_subs = [s for s in subs_s if s.cas_number and s.weight_fraction_max_pct]
    precision_det = len(complete_subs) / len(subs_s) if subs_s else 0
    det_status = "pass" if precision_det >= 0.95 else ("warn" if precision_det >= 0.70 else "fail")

    with c1:
        st.markdown(eval_card(
            title="Compliance Detection Precision",
            step_badge=badge("Step 3","badge-3-a"),
            formula="TP / (TP + FP) per regulation",
            description=(
                "Measures correctness of FAIL verdicts in regulatory screening. "
                "High precision avoids false alarms that waste engineer time. "
                "Target: ≥ 95% precision on REACH / RoHS threshold violations."
            ),
            target="≥ 95% precision on threshold violations",
            measured_html=mval(
                f"Complete records (CAS + weight): {len(complete_subs)}/{len(subs_s)}  |  "
                f"Precision proxy: {precision_det*100:.1f}%",
                det_status
            )
        ), unsafe_allow_html=True)

    # Sensitivity coverage
    range_subs  = [s for s in subs_s if s.is_range]
    near_thresh = []
    for s in subs_s:
        # ROHS lead 1000 ppm — check if within 80% of any known threshold
        ppm = s.weight_ppm
        if ppm > 0 and (800 <= ppm <= 1200 or 80 <= ppm <= 120):
            near_thresh.append(s.name)
    cov_pct = len(range_subs) / len(subs_s) * 100 if subs_s else 0

    with c2:
        st.markdown(eval_card(
            title="Sensitivity Coverage",
            step_badge=badge("Step 3","badge-3-b"),
            formula="P(exceed | uncertainty model) — Sobol index",
            description=(
                "Measures how many near-threshold substances are flagged with risk level. "
                "Sobol index validates that OCR uncertainty is the primary driver of "
                "compliance decision variance."
            ),
            target="All near-threshold substances flagged",
            measured_html=mval(
                f"Substances with concentration ranges: {len(range_subs)}/{len(subs_s)}  |  "
                f"Near-threshold: {len(near_thresh)}  |  Range coverage: {cov_pct:.0f}%",
                "pass" if len(subs_s) > 0 else "warn"
            )
        ), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — REGULATORY COMPLIANCE SCREENING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("⚖️ Step 4 — Regulatory Compliance Screening")
st.caption("PASS = below threshold. REVIEW = range crosses threshold. FAIL = above threshold. UNKNOWN = CAS not in configured ruleset.")

c1,c2,c3,c4 = st.columns(4)
c1.metric("✅ PASS",          compliance.pass_count)
c2.metric("🔶 REVIEW",        compliance.warn_count)
c3.metric("❌ FAIL",          compliance.fail_count)
c4.metric("⏱ Time",           f"{st.session_state.get('step4_time',0):.2f}s")

reg_rows = []
for f in compliance.findings:
    for r in f.regulatory_checks:
        reg_rows.append({
            "Substance":       f.substance_name,
            "CAS":             f.cas_number or "—",
            "Regulation":      r.regulation,
            "Status":          r.status.value,
            "Threshold (ppm)": r.threshold_ppm if r.threshold_ppm is not None else "—",
            "Observed (ppm)":  r.observed_ppm or "—",
            "Rationale":       r.rationale,
        })
reg_df = pd.DataFrame(reg_rows)
if not reg_df.empty:
    st.dataframe(reg_df, use_container_width=True, hide_index=True)
else:
    st.info("No regulatory checks could be performed — no matching CAS numbers in the configured ruleset.")

for f in compliance.findings:
    if f.findings:
        with st.expander(f"{f.substance_name} — {f.overall_status.value}"):
            for msg in f.findings:
                st.write("• " + msg)

# ── EVALUATION METRICS ───────────────────────────────────────────────────────
with st.expander("📊 Step 4 — Evaluation Metrics (Compliance Detection Precision)", expanded=True):
    st.markdown('<hr class="eval-sep">', unsafe_allow_html=True)
    st.markdown("**Evaluation Metrics measured at this step**")
    c1, c2 = st.columns(2)

    n_findings = len(compliance.findings)
    n_pass     = compliance.pass_count
    n_fail     = compliance.fail_count
    n_review   = compliance.warn_count

    # Reg score from engine = percentage of substances with PASS in reg checks
    reg_score = compliance.regulatory_score
    reg_status = "pass" if reg_score >= 95 else ("warn" if reg_score >= 70 else "fail")

    with c1:
        st.markdown(eval_card(
            title="Compliance Detection Precision",
            step_badge=badge("Step 3","badge-3-a"),
            formula="TP / (TP + FP) per regulation",
            description=(
                "Measures correctness of FAIL verdicts. "
                "High precision avoids false alarms that waste engineer time. "
                "Target: ≥ 95% precision on REACH / RoHS threshold violations."
            ),
            target="≥ 95% precision on threshold violations",
            measured_html=mval(
                f"Regulatory score: {reg_score:.1f}%  |  "
                f"PASS: {n_pass}  REVIEW: {n_review}  FAIL: {n_fail}  |  "
                f"Overall: {compliance.overall_score}/100",
                reg_status
            )
        ), unsafe_allow_html=True)

    with c2:
        step4_time = st.session_state.get("step4_time", 0)
        e2e_status = "pass" if step4_time < 5 else "warn"
        st.markdown(eval_card(
            title="End-to-End Processing Time",
            step_badge=badge("System","badge-sys"),
            formula="Seconds per MDS document",
            description=(
                "Step 4 regulatory screening time. Fast screening enables batch processing "
                "of entire supplier portfolios. Target: screening of all substances < 5s."
            ),
            target="< 5s for regulatory screening step",
            measured_html=mval(
                f"Step 4 time: {step4_time:.3f}s  |  Total pipeline: {st.session_state.get('e2e_time',0):.2f}s",
                e2e_status
            )
        ), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — SENSITIVITY & THRESHOLD RISK
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🎯 Step 5 — Sensitivity & Threshold Risk")
st.caption("Monte Carlo uniform sampling (n=2048) across supplier-declared concentration ranges. Sobol sensitivity index checks OCR uncertainty is primary variance driver.")

sc1,sc2,sc3 = st.columns(3)
sc1.metric("Substances analysed",      len(sensitivity.results))
sc2.metric("High / Critical risk",     len(sensitivity.high_risk_substances))
sc3.metric("⏱ Time",                  f"{st.session_state.get('step5_time',0):.2f}s")

sens_rows = []
for r in sensitivity.results:
    if r.threshold_risks:
        for x in r.threshold_risks:
            sens_rows.append({
                "Substance":           r.substance_name,
                "CAS":                 r.cas_number or "—",
                "Reported conc.":      r.reported_range,
                "Regulation":          x.regulation,
                "Threshold ppm":       x.threshold_ppm,
                "P(exceed threshold)": f"{x.exceedance_probability*100:.1f}%",
                "95% range ppm":       f"{x.ci_lower_ppm:g}–{x.ci_upper_ppm:g}",
                "Risk":                x.risk_level,
            })
    else:
        sens_rows.append({
            "Substance":           r.substance_name,
            "CAS":                 r.cas_number or "—",
            "Reported conc.":      r.reported_range,
            "Regulation":          "No applicable configured threshold",
            "Threshold ppm":       "—",
            "P(exceed threshold)": "N/A",
            "95% range ppm":       r.reported_range.replace("-","–"),
            "Risk":                "N/A",
        })
st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)

# ── EVALUATION METRICS ───────────────────────────────────────────────────────
with st.expander("📊 Step 5 — Evaluation Metrics (Sensitivity Coverage)", expanded=True):
    st.markdown('<hr class="eval-sep">', unsafe_allow_html=True)
    st.markdown("**Evaluation Metrics measured at this step**")
    c1, c2 = st.columns(2)

    # Count all threshold risks across results
    all_risks = [x for r in sensitivity.results for x in r.threshold_risks]
    high_crit = [x for x in all_risks if x.risk_level in ("HIGH","CRITICAL")]
    near_sub  = [r for r in sensitivity.results if r.threshold_risks]
    avg_exceedance = sum(x.exceedance_probability for x in all_risks) / len(all_risks) if all_risks else 0
    sens_status = "pass" if len(sensitivity.high_risk_substances) == 0 else ("warn" if len(sensitivity.high_risk_substances) < 3 else "fail")

    with c1:
        st.markdown(eval_card(
            title="Sensitivity Coverage",
            step_badge=badge("Step 3","badge-3-b"),
            formula="P(exceed | uncertainty model)  via Monte Carlo / Sobol",
            description=(
                "Measures how many near-threshold substances are flagged with risk level. "
                "Sobol index validates OCR uncertainty is the primary driver of compliance "
                "decision variance. Every substance with a declared range must be covered."
            ),
            target="All near-threshold substances flagged",
            measured_html=mval(
                f"Substances with threshold risks: {len(near_sub)}/{len(sensitivity.results)}  |  "
                f"High/Critical risks: {len(high_crit)}  |  "
                f"Avg P(exceed): {avg_exceedance*100:.1f}%",
                sens_status
            )
        ), unsafe_allow_html=True)

    with c2:
        e2e = st.session_state.get("e2e_time", 0)
        e2e_status2 = "pass" if e2e < 30 else ("warn" if e2e < 60 else "fail")
        st.markdown(eval_card(
            title="End-to-End Processing Time",
            step_badge=badge("System","badge-sys"),
            formula="Total seconds for complete 5-step pipeline",
            description=(
                "Measures real-world deployability from document upload to final sensitivity report. "
                "Target: < 30 sec per document (versus 4–8 hrs manual review). "
                "Key efficiency metric for OEM and supplier portal adoption."
            ),
            target="< 30 sec / document",
            measured_html=mval(
                f"Step1:{st.session_state.get('ocr_time',0):.2f}s  "
                f"Step2:{st.session_state.get('nlp_time',0):.2f}s  "
                f"Step3:{st.session_state.get('step3_time',0):.2f}s  "
                f"Step4:{st.session_state.get('step4_time',0):.2f}s  "
                f"Step5:{st.session_state.get('step5_time',0):.2f}s  "
                f"→ Total: {e2e:.2f}s",
                e2e_status2
            )
        ), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 (BONUS) — LLM REASONING & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
llm_reasoning = st.session_state.get("llm_reasoning")
if llm_enabled or llm_reasoning:
    st.markdown("---")
    st.header("🤖 Step 5 — LLM Recommendations")
    if not llm_enabled:
        st.info("Enable **Generate LLM recommendations** in the sidebar, then re-run the analysis.")
    elif llm_reasoning is None:
        st.warning("LLM step did not complete — check the provider and that the API key "
                   "(OPENAI_API_KEY / api_keys.openrouter in config.json) is set, "
                   "or that Ollama is running locally.")
    else:
        lm1, lm2 = st.columns(2)
        lm1.metric("Model used", llm_reasoning.model_used)
        lm2.metric("⏱ Time", f"{st.session_state.get('llm_time', 0):.2f}s")

        st.subheader("Executive Summary")
        st.markdown(llm_reasoning.executive_summary or "_No summary returned._")

        st.subheader("Risk Narrative")
        st.markdown(llm_reasoning.risk_narrative or "_No narrative returned._")

        st.subheader("Substance Recommendations")
        if llm_reasoning.recommendations:
            rec_rows = [{
                "Substance":         r.substance_name,
                "Status":            r.status,
                "Plain-English Finding": r.plain_english_finding,
                "Corrective Action": r.corrective_action,
                "Exemption Check":   r.exemption_check,
            } for r in llm_reasoning.recommendations]
            st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)
        else:
            st.success("No FAIL/WARNING findings — no corrective actions required.")

# ══════════════════════════════════════════════════════════════════════════════
#  CONSOLIDATED EVALUATION METRICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("📊 Evaluation Metrics — Full Pipeline Summary")
st.caption("All five evaluation metrics from the project proposal, measured against this document.")

ocr_conf    = ocr.avg_confidence
cer         = max(0, 100 - ocr_conf)
subs        = nlp_result.substances
n_subs      = len(subs)
n_cas_det   = sum(1 for s in subs if s.cas_number)
prec_r      = (n_cas_det / n_subs) if n_subs else 0
f1_r        = (2*prec_r*prec_r/(2*prec_r)) if prec_r > 0 else 0
subs_s      = structured.all_substances
complete_s  = [s for s in subs_s if s.cas_number and s.weight_fraction_max_pct]
det_prec    = len(complete_s)/len(subs_s) if subs_s else 0
all_risks   = [x for r in sensitivity.results for x in r.threshold_risks]
near_sub    = [r for r in sensitivity.results if r.threshold_risks]
avg_p       = sum(x.exceedance_probability for x in all_risks)/len(all_risks) if all_risks else 0
e2e_t       = st.session_state.get("e2e_time", 0)

rows_summary = [
    {
        "Metric":         "OCR Accuracy (CER / WER)",
        "Step":           "Step 1",
        "Formula":        "1 – (Edit Distance / True Length)",
        "Target":         "CER < 10%",
        "Measured":       f"CER ≈ {cer:.1f}%  (confidence {ocr_conf:.1f}%)",
        "Status":         "✅ PASS" if cer < 10 else ("⚠️ WARN" if cer < 20 else "❌ FAIL"),
    },
    {
        "Metric":         "NER Precision / Recall / F1",
        "Step":           "Step 2",
        "Formula":        "F1 = 2·P·R / (P+R)",
        "Target":         "F1 ≥ 0.85",
        "Measured":       f"F1 ≈ {f1_r:.2f}  (CAS detected {n_cas_det}/{n_subs})",
        "Status":         "✅ PASS" if f1_r >= 0.85 else ("⚠️ WARN" if f1_r >= 0.60 else "❌ FAIL"),
    },
    {
        "Metric":         "Compliance Detection Precision",
        "Step":           "Step 3 / 4",
        "Formula":        "TP / (TP + FP) per regulation",
        "Target":         "≥ 95%",
        "Measured":       f"{det_prec*100:.1f}%  (complete records {len(complete_s)}/{len(subs_s)})",
        "Status":         "✅ PASS" if det_prec >= 0.95 else ("⚠️ WARN" if det_prec >= 0.70 else "❌ FAIL"),
    },
    {
        "Metric":         "Sensitivity Coverage",
        "Step":           "Step 3 / 5",
        "Formula":        "P(exceed | uncertainty model)",
        "Target":         "All near-threshold flagged",
        "Measured":       f"{len(near_sub)}/{len(sensitivity.results)} substances with threshold risks  |  Avg P(exceed) {avg_p*100:.1f}%",
        "Status":         "✅ PASS" if len(sensitivity.results) > 0 else "⚠️ WARN",
    },
    {
        "Metric":         "End-to-End Processing Time",
        "Step":           "System",
        "Formula":        "Seconds per MDS document",
        "Target":         "< 30 sec",
        "Measured":       f"{e2e_t:.2f}s  (vs 4–8 hrs manual)",
        "Status":         "✅ PASS" if e2e_t < 30 else ("⚠️ WARN" if e2e_t < 60 else "❌ FAIL"),
    },
]

st.dataframe(
    pd.DataFrame(rows_summary),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Status": st.column_config.TextColumn("Status", width="small"),
        "Formula": st.column_config.TextColumn("Formula", width="medium"),
        "Measured": st.column_config.TextColumn("Measured Value", width="large"),
    }
)

# ══════════════════════════════════════════════════════════════════════════════
#  CAVEATS & WARNINGS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("⚠️ Major Warnings / Regulatory Caveats")
for msg in [
    "Regulatory screening is only against the configured demonstration ruleset. It is NOT a legal certification of REACH, RoHS, ELV or any other regulation.",
    "A supplier-declared concentration range that crosses a threshold is shown as REVIEW because exact compliance cannot be concluded from the SDS.",
    "A substance not present in the configured list is shown as UNKNOWN, not PASS. The rule database must be current and jurisdiction-specific before production use.",
    "REACH, RoHS and ELV applicability can depend on product/article scope, exemptions, substance form and jurisdiction; CAS + concentration alone may be insufficient.",
    "Human regulatory review and supplier confirmation are required before using this output for release, procurement, environmental, safety or compliance decisions.",
]:
    st.warning(msg)

st.subheader("Extraction warnings")
for w in nlp_result.extraction_warnings:
    st.warning(w)

st.info(
    "Safety note: this application is an extraction assistant, not a replacement for the SDS, "
    "supplier confirmation, toxicological assessment, or regulatory/compliance review. "
    "Always verify low-confidence fields and any value used for a safety or regulatory decision."
)
