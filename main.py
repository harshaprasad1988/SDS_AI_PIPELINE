"""
═══════════════════════════════════════════════════════════════════
main.py — MDS Checker Full Pipeline Entry Point
═══════════════════════════════════════════════════════════════════

Run with:
    python main.py --input data/sample_mds/sample.pdf --output outputs/
    python main.py --input doc.pdf --llm ollama --model llama3
    python main.py --input doc.pdf --llm openai --model gpt-4o-mini

All 7 steps run in order:
  1. OCR        → raw text from PDF/image
  2. NLP        → entities: substances, CAS, weights
  3. Structure  → typed StructuredMDS object
  4. Compliance → PASS/FAIL/WARN per substance per regulation
  5. Sensitivity→ Monte Carlo uncertainty analysis
  6. LLM        → plain-English findings + corrective actions
  7. Report     → JSON + HTML outputs
═══════════════════════════════════════════════════════════════════
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# Pipeline modules (one per step)
from modules.step1_ocr        import OCRProcessor
from modules.step2_nlp        import NLPExtractor
from modules.step3_structurer import DataStructurer
from modules.step4_compliance import ComplianceEngine
from modules.step5_sensitivity import SensitivityAnalyser
from modules.step6_llm        import LLMReasoner
from modules.step7_report     import ReportGenerator


# ── Logging Setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/pipeline.log", mode="a")
    ]
)
logger = logging.getLogger("mds_checker")


# ── Config Loader ─────────────────────────────────────────────────────────────

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(input_file: str, output_dir: str, llm_provider: str, llm_model: str) -> dict:
    """
    Execute all 7 pipeline steps sequentially.
    Returns a dict of output file paths.
    """
    config = load_config()
    total_start = time.time()
    
    logger.info("=" * 60)
    logger.info(f"MDS Checker Pipeline START: {Path(input_file).name}")
    logger.info("=" * 60)
    
    # ── STEP 1: OCR ──────────────────────────────────────────────────────────
    logger.info("\n📄 STEP 1: OCR / Vision Extraction")
    t0 = time.time()
    ocr_processor = OCRProcessor(
        engine=config["ocr"]["engine"],
        dpi=config["ocr"]["dpi"],
        lang=config["ocr"]["language"]
    )
    ocr_result = ocr_processor.process(input_file)
    logger.info(f"   ✓ Done in {time.time()-t0:.1f}s | Pages: {ocr_result.total_pages} "
                f"| Avg confidence: {ocr_result.avg_confidence:.1f}%")
    
    # ── STEP 2: NLP ──────────────────────────────────────────────────────────
    logger.info("\n🧠 STEP 2: NLP / NER Extraction")
    t0 = time.time()
    nlp_extractor = NLPExtractor()
    nlp_result = nlp_extractor.extract(ocr_result)
    logger.info(f"   ✓ Done in {time.time()-t0:.1f}s | "
                f"Substances: {len(nlp_result.substances)} | "
                f"Warnings: {len(nlp_result.extraction_warnings)}")
    if nlp_result.extraction_warnings:
        for w in nlp_result.extraction_warnings:
            logger.warning(f"   ⚠  {w}")
    
    # ── STEP 3: STRUCTURE ────────────────────────────────────────────────────
    logger.info("\n🗃  STEP 3: Data Structuring")
    t0 = time.time()
    structurer = DataStructurer()
    doc_id = Path(input_file).stem.upper()
    structured_mds = structurer.structure(nlp_result, doc_id=doc_id)
    logger.info(f"   ✓ Done in {time.time()-t0:.1f}s | "
                f"Resolved substances: {len(structured_mds.all_substances)}")
    if structured_mds.structuring_warnings:
        for w in structured_mds.structuring_warnings:
            logger.warning(f"   ⚠  {w}")
    
    # ── STEP 4: COMPLIANCE ───────────────────────────────────────────────────
    logger.info("\n⚖️  STEP 4: Compliance Rule Engine")
    t0 = time.time()
    engine = ComplianceEngine(score_weights=config["reporting"]["score_weights"])
    compliance_report = engine.run(structured_mds)
    logger.info(f"   ✓ Done in {time.time()-t0:.1f}s | "
                f"Score: {compliance_report.overall_score}/100 | "
                f"FAIL: {compliance_report.fail_count} | "
                f"WARN: {compliance_report.warn_count} | "
                f"PASS: {compliance_report.pass_count}")
    
    # ── STEP 5: SENSITIVITY ──────────────────────────────────────────────────
    logger.info("\n📊 STEP 5: Sensitivity Analysis")
    t0 = time.time()
    analyser = SensitivityAnalyser(
        uncertainty_pct=0.10,
        n_samples=config["sensitivity"]["num_samples"]
    )
    sensitivity_report = analyser.analyse(structured_mds.all_substances)
    logger.info(f"   ✓ Done in {time.time()-t0:.1f}s | "
                f"High-risk substances: {len(sensitivity_report.high_risk_substances)}")
    
    # ── STEP 6: LLM REASONING ────────────────────────────────────────────────
    logger.info(f"\n🤖 STEP 6: LLM Reasoning ({llm_provider}/{llm_model})")
    t0 = time.time()
    reasoner = LLMReasoner(provider=llm_provider, model=llm_model)
    try:
        llm_reasoning = reasoner.reason(compliance_report, sensitivity_report)
        logger.info(f"   ✓ Done in {time.time()-t0:.1f}s | "
                    f"Recommendations: {len(llm_reasoning.recommendations)}")
    except Exception as e:
        logger.warning(f"   ⚠  LLM step failed ({e}). Using rule-based summary only.")
        from modules.step6_llm import LLMReasoning
        llm_reasoning = LLMReasoning(
            executive_summary=compliance_report.summary,
            recommendations=[],
            risk_narrative="LLM unavailable — see rule-based findings.",
            model_used="fallback"
        )
    
    # ── STEP 7: REPORTING ─────────────────────────────────────────────────────
    logger.info("\n📈 STEP 7: Report Generation")
    t0 = time.time()
    generator = ReportGenerator(output_dir=output_dir)
    output_paths = generator.generate_all(compliance_report, sensitivity_report, llm_reasoning)
    logger.info(f"   ✓ Done in {time.time()-t0:.1f}s")
    
    # ── SUMMARY ───────────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ PIPELINE COMPLETE in {total_time:.1f}s")
    logger.info(f"   Overall Score : {compliance_report.overall_score}/100")
    logger.info(f"   FAIL / WARN / PASS : {compliance_report.fail_count} / {compliance_report.warn_count} / {compliance_report.pass_count}")
    for fmt, path in output_paths.items():
        logger.info(f"   {fmt.upper():6s} → {path}")
    logger.info("=" * 60)
    
    return output_paths


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-based MDS Compliance Checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/sample_mds/sample.pdf
  python main.py --input doc.pdf --llm ollama --model llama3
  python main.py --input doc.pdf --llm openai --model gpt-4o-mini --output results/
        """
    )
    parser.add_argument("--input",  required=True,  help="Path to MDS PDF or image file")
    parser.add_argument("--output", default="outputs/", help="Output directory (default: outputs/)")
    parser.add_argument("--llm",    default="openai", choices=["openai", "ollama"], help="LLM provider")
    parser.add_argument("--model",  default="gpt-4o-mini", help="LLM model name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output dir
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    output_paths = run_pipeline(
        input_file=args.input,
        output_dir=args.output,
        llm_provider=args.llm,
        llm_model=args.model
    )
    
    print(f"\nOutputs:")
    for fmt, path in output_paths.items():
        print(f"  {fmt}: {path}")
