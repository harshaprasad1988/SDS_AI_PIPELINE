"""
═══════════════════════════════════════════════════════════════════
STEP 7 — Report & Dashboard Generator
═══════════════════════════════════════════════════════════════════

PURPOSE:
    Generate three output formats from the full pipeline results:
    1. JSON  — machine-readable, for API consumers / IMDS upload
    2. HTML  — interactive dashboard with charts (Plotly)
    3. PDF   — printable audit-ready compliance report

WHAT IT CONTAINS:
    - Overall compliance score (gauge chart)
    - Per-check scores: Completeness / Consistency / Regulatory
    - Substance-level findings table with traffic-light status
    - Sensitivity analysis bar chart (risk levels per substance)
    - LLM recommendations section
    - Audit trail metadata

WHY THREE FORMATS?
    JSON → developers / automated systems
    HTML → engineers reviewing on screen
    PDF  → management, OEM submission, audit archive
═══════════════════════════════════════════════════════════════════
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Optional

from modules.step4_compliance import ComplianceReport, CheckStatus
from modules.step5_sensitivity import SensitivityReport
from modules.step6_llm import LLMReasoning

logger = logging.getLogger(__name__)

STATUS_EMOJI = {
    CheckStatus.PASS:    "✅",
    CheckStatus.FAIL:    "❌",
    CheckStatus.WARNING: "⚠️",
    CheckStatus.UNKNOWN: "❓",
}

STATUS_COLOR = {
    CheckStatus.PASS:    "#2E7D32",  # Green
    CheckStatus.FAIL:    "#B71C1C",  # Red
    CheckStatus.WARNING: "#E65100",  # Orange
    CheckStatus.UNKNOWN: "#546E7A",  # Gray
}

RISK_COLOR = {
    "LOW":      "#4CAF50",
    "MEDIUM":   "#FF9800",
    "HIGH":     "#F44336",
    "CRITICAL": "#880E4F",
}


# ── JSON Report ───────────────────────────────────────────────────────────────

def generate_json_report(
    compliance: ComplianceReport,
    sensitivity: SensitivityReport,
    reasoning: LLMReasoning,
    output_path: str
) -> str:
    """
    Produce a machine-readable JSON report.
    Structured for easy upload to IMDS or an internal compliance API.
    """
    report = {
        "metadata": {
            "document_id": compliance.document_id,
            "supplier": compliance.supplier,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "llm_model": reasoning.model_used,
        },
        "scores": {
            "overall":       compliance.overall_score,
            "completeness":  compliance.completeness_score,
            "consistency":   compliance.consistency_score,
            "regulatory":    compliance.regulatory_score,
        },
        "summary": compliance.summary,
        "executive_summary": reasoning.executive_summary,
        "substance_findings": [
            {
                "substance":          f.substance_name,
                "cas":                f.cas_number,
                "weight_ppm":         f.weight_ppm,
                "completeness":       f.completeness_status.value,
                "consistency":        f.consistency_status.value,
                "regulatory":         f.regulatory_status.value,
                "overall":            f.overall_status.value,
                "issues":             f.findings,
                "regulations_hit":    f.regulation_hits,
            }
            for f in compliance.findings
        ],
        "sensitivity_high_risk": sensitivity.high_risk_substances,
        "recommendations": [
            {
                "substance":          r.substance_name,
                "status":             r.status,
                "finding":            r.plain_english_finding,
                "corrective_action":  r.corrective_action,
                "exemption_check":    r.exemption_check,
            }
            for r in reasoning.recommendations
        ],
    }
    
    Path(output_path).write_text(json.dumps(report, indent=2))
    logger.info(f"JSON report written: {output_path}")
    return output_path


# ── HTML Report ───────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MDS Compliance Report — {doc_id}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0D1B2A; color: #E8EDF2; margin: 0; padding: 0; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 32px; }}
  h1 {{ color: #1E88E5; border-bottom: 2px solid #1E88E5; padding-bottom: 8px; }}
  h2 {{ color: #00ACC1; margin-top: 36px; }}
  .score-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }}
  .score-card {{ background: #132033; border-radius: 8px; padding: 20px; text-align: center; }}
  .score-card .score {{ font-size: 2.2em; font-weight: bold; }}
  .score-card .label {{ font-size: 0.85em; color: #B0BEC5; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; background: #132033; border-radius: 8px; overflow: hidden; }}
  th {{ background: #1E88E5; padding: 12px; text-align: left; font-size: 0.9em; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #0D1B2A; font-size: 0.88em; }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{ padding: 2px 10px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }}
  .rec-card {{ background: #132033; border-left: 4px solid #FF6F00; border-radius: 6px; padding: 16px; margin: 12px 0; }}
  .rec-card h4 {{ margin: 0 0 6px; color: #FF6F00; }}
  .summary-box {{ background: #132033; border-radius: 8px; padding: 20px; margin: 20px 0; line-height: 1.7; }}
  .meta {{ color: #546E7A; font-size: 0.8em; margin-top: 40px; }}
</style>
</head>
<body>
<div class="container">
  <h1>🧪 MDS Compliance Report</h1>
  <p><strong>Document:</strong> {doc_id} &nbsp;|&nbsp; <strong>Supplier:</strong> {supplier} &nbsp;|&nbsp; <strong>Generated:</strong> {timestamp}</p>

  <h2>Compliance Scores</h2>
  <div class="score-grid">
    <div class="score-card"><div class="score" style="color:{overall_color}">{overall}%</div><div class="label">Overall Score</div></div>
    <div class="score-card"><div class="score" style="color:#1E88E5">{completeness}%</div><div class="label">Completeness</div></div>
    <div class="score-card"><div class="score" style="color:#00ACC1">{consistency}%</div><div class="label">Consistency</div></div>
    <div class="score-card"><div class="score" style="color:{reg_color}">{regulatory}%</div><div class="label">Regulatory</div></div>
  </div>

  <div class="summary-box">
    <strong>Executive Summary:</strong><br>{executive_summary}
  </div>

  <h2>Substance Findings</h2>
  <table>
    <tr><th>Substance</th><th>CAS</th><th>Weight (ppm)</th><th>Completeness</th><th>Consistency</th><th>Regulatory</th><th>Regulations</th></tr>
    {findings_rows}
  </table>

  <h2>Recommendations</h2>
  {recommendations_html}

  <p class="meta">Report generated by AI-based MDS Checker | LLM: {llm_model} | MTech Final Year Project</p>
</div>
</body>
</html>
"""

def _status_badge(status: CheckStatus) -> str:
    emoji = STATUS_EMOJI.get(status, "❓")
    color = STATUS_COLOR.get(status, "#546E7A")
    return f'<span class="badge" style="background:{color}20; color:{color}">{emoji} {status.value}</span>'

def _score_color(score: float) -> str:
    if score >= 80: return "#2E7D32"
    if score >= 60: return "#E65100"
    return "#B71C1C"


def generate_html_report(
    compliance: ComplianceReport,
    sensitivity: SensitivityReport,
    reasoning: LLMReasoning,
    output_path: str
) -> str:
    """Generate an interactive HTML compliance dashboard."""
    
    # Build findings table rows
    rows = ""
    for f in compliance.findings:
        regs = ", ".join(f.regulation_hits) if f.regulation_hits else "—"
        rows += f"""
        <tr>
          <td><strong>{f.substance_name}</strong></td>
          <td><code>{f.cas_number or '—'}</code></td>
          <td>{f.weight_ppm:.1f}</td>
          <td>{_status_badge(f.completeness_status)}</td>
          <td>{_status_badge(f.consistency_status)}</td>
          <td>{_status_badge(f.regulatory_status)}</td>
          <td>{regs}</td>
        </tr>"""
    
    # Build recommendation cards
    rec_html = ""
    for r in reasoning.recommendations:
        rec_html += f"""
        <div class="rec-card">
          <h4>⚠️ {r.substance_name} — {r.status}</h4>
          <p><strong>Finding:</strong> {r.plain_english_finding}</p>
          <p><strong>Action:</strong> {r.corrective_action}</p>
          <p><strong>Exemption check:</strong> {r.exemption_check}</p>
        </div>"""
    
    if not rec_html:
        rec_html = "<p>✅ No corrective actions required — all substances pass.</p>"
    
    html = HTML_TEMPLATE.format(
        doc_id          = compliance.document_id,
        supplier        = compliance.supplier or "Unknown",
        timestamp       = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        overall         = compliance.overall_score,
        completeness    = compliance.completeness_score,
        consistency     = compliance.consistency_score,
        regulatory      = compliance.regulatory_score,
        overall_color   = _score_color(compliance.overall_score),
        reg_color       = _score_color(compliance.regulatory_score),
        executive_summary = reasoning.executive_summary or compliance.summary,
        findings_rows   = rows,
        recommendations_html = rec_html,
        llm_model       = reasoning.model_used,
    )
    
    Path(output_path).write_text(html, encoding="utf-8")
    logger.info(f"HTML report written: {output_path}")
    return output_path


# ── Facade: Generate All Reports ──────────────────────────────────────────────

class ReportGenerator:
    """
    Generates all output formats in one call.
    
    Usage:
        gen = ReportGenerator(output_dir="outputs/")
        paths = gen.generate_all(compliance, sensitivity, reasoning)
    """
    
    def __init__(self, output_dir: str = "outputs/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all(
        self,
        compliance: ComplianceReport,
        sensitivity: SensitivityReport,
        reasoning: LLMReasoning
    ) -> dict:
        """Generate JSON + HTML reports and return paths dict."""
        doc_id = compliance.document_id.replace(" ", "_")
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        paths = {}
        
        # JSON
        json_path = str(self.output_dir / f"{doc_id}_{ts}_report.json")
        paths["json"] = generate_json_report(compliance, sensitivity, reasoning, json_path)
        
        # HTML
        html_path = str(self.output_dir / f"{doc_id}_{ts}_report.html")
        paths["html"] = generate_html_report(compliance, sensitivity, reasoning, html_path)
        
        logger.info(f"Reports generated: {list(paths.values())}")
        return paths
