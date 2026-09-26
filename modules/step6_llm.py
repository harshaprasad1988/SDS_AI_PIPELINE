"""
═══════════════════════════════════════════════════════════════════
STEP 6 — LLM-based Reasoning & Recommendation Generation
═══════════════════════════════════════════════════════════════════

PURPOSE:
    Use a Large Language Model (GPT-4o-mini or local Llama) to:
    1. Interpret ambiguous compliance findings in plain English
    2. Suggest specific corrective actions for each FAIL/WARNING
    3. Generate a human-readable executive summary for engineers
    4. Handle edge cases that rule-based logic cannot reason about

WHY LLM AND NOT JUST RULES?
    Rule engines are binary: exceed threshold → FAIL. But real MDS
    review involves nuance:
    - "Is Lead in solder exempt under RoHS Annex III?"
    - "Is this substance used in medical equipment (different limits)?"
    - "What's the likely supplier intent here?"
    An LLM can contextualise these questions. Rules cannot.

INTEGRATION:
    Supports two modes:
    - cloud: OpenAI API (GPT-4o-mini, fast, requires API key)
    - local:  Ollama (Llama3 running locally, no API key needed)
═══════════════════════════════════════════════════════════════════
"""

import logging
import json
from dataclasses import dataclass, field
from typing import List, Optional

from modules.step4_compliance import ComplianceReport, SubstanceFinding, CheckStatus
from modules.step5_sensitivity import SensitivityReport

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class SubstanceRecommendation:
    """LLM-generated recommendation for a single flagged substance."""
    substance_name: str
    status: str              # FAIL / WARNING
    plain_english_finding: str   # What went wrong, in simple language
    corrective_action: str       # Specific, actionable next step
    exemption_check: str         # Does any RoHS/REACH exemption apply?

@dataclass
class LLMReasoning:
    """Full LLM output for the document."""
    executive_summary: str
    recommendations: List[SubstanceRecommendation]
    risk_narrative: str          # Paragraph explaining overall risk
    model_used: str


# ── Prompt Templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert automotive materials compliance engineer specialising in 
REACH, RoHS, and ELV regulations. You review Material Data Sheet (MDS) 
compliance check results and provide clear, actionable guidance.

Rules:
- Be specific and technical — engineers are your audience
- Reference the exact regulation and article number when possible
- Suggest concrete corrective actions (e.g., "Replace CAS 7439-92-1 with 
  tin-silver solder (CAS 7440-31-5) per IPC-4101 spec")
- Check whether RoHS Annex III / Annex IV exemptions might apply
- Keep each recommendation under 100 words
- Always respond in valid JSON
"""

FINDING_PROMPT_TEMPLATE = """
The following substances were found in an automotive MDS compliance check.
Analyse each FAIL and WARNING entry and provide recommendations.

Document ID: {doc_id}
Supplier: {supplier}
Overall Score: {score}/100

Findings (FAIL and WARNING only):
{findings_json}

Sensitivity Analysis (high-risk substances):
{sensitivity_json}

Respond ONLY with a valid JSON object in this exact structure:
{{
  "executive_summary": "2-3 sentence summary for management",
  "risk_narrative": "1 paragraph explaining the overall risk picture",
  "recommendations": [
    {{
      "substance_name": "...",
      "status": "FAIL or WARNING",
      "plain_english_finding": "What this means in simple terms",
      "corrective_action": "Specific action the engineer should take",
      "exemption_check": "Any applicable RoHS/REACH exemption, or 'None identified'"
    }}
  ]
}}
"""


# ── LLM Clients ──────────────────────────────────────────────────────────────

def call_openai(prompt: str, system: str, model: str = "gpt-4o-mini",
                temperature: float = 0.2, max_tokens: int = 1500) -> str:
    """Call OpenAI API and return raw response text."""
    try:
        from openai import OpenAI
        from .config import get_openai_api_key
        client = OpenAI(api_key=get_openai_api_key() or None)  # config.json, then OPENAI_API_KEY env
        
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise


def call_ollama(prompt: str, system: str, model: str = "llama3",
                base_url: str = "http://localhost:11434") -> str:
    """Call a local Ollama instance (no API key needed)."""
    import requests
    
    payload = {
        "model": model,
        "prompt": f"{system}\n\n{prompt}",
        "stream": False,
        "format": "json"
    }
    
    try:
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"]
    except requests.ConnectionError:
        raise RuntimeError(
            f"Could not connect to Ollama at {base_url}. "
            "Start it with: ollama serve"
        )


def call_openrouter(prompt: str, system: str,
                    model: str = "qwen/qwen3-235b-a22b",
                    temperature: float = 0.2, max_tokens: int = 1500) -> str:
    """Call ANY OpenRouter-hosted model via OpenRouter's OpenAI-compatible
    endpoint (https://openrouter.ai/api/v1). The `model` argument is passed
    straight through, so Qwen ("qwen/..."), Microsoft Phi-4
    ("microsoft/phi-4"), Mistral Small ("mistralai/mistral-small-latest"),
    DeepSeek ("deepseek/deepseek-v4-flash"), OpenAI ("openai/gpt-4o-mini") and
    every other OpenRouter model ID are all handled by this single function."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    from .config import get_openrouter_api_key, get_openrouter_base_url
    api_key = get_openrouter_api_key()
    if not api_key:
        raise RuntimeError(
            f"OpenRouter model '{model}' requires an API key. Add it to "
            "config.json (api_keys.openrouter) or set the OPENROUTER_API_KEY "
            "environment variable. Get one from https://openrouter.ai/keys"
        )

    client = OpenAI(api_key=api_key, base_url=get_openrouter_base_url())
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenRouter ({model}) API error: {e}")
        raise


# ── Main LLM Reasoner ────────────────────────────────────────────────────────

class LLMReasoner:
    """
    Uses an LLM to generate human-readable findings and recommendations.
    
    Usage:
        reasoner = LLMReasoner(provider="openai")
        reasoning = reasoner.reason(compliance_report, sensitivity_report)
    """
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini",
                 temperature: float = 0.2, ollama_url: str = "http://localhost:11434",
                 openrouter_base_url: str = None, max_tokens: int = 1500):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ollama_url = ollama_url
        if openrouter_base_url is None:
            from .config import get_openrouter_base_url
            openrouter_base_url = get_openrouter_base_url()
        self.openrouter_base_url = openrouter_base_url
    
    def reason(self, compliance: ComplianceReport, sensitivity: SensitivityReport) -> LLMReasoning:
        """Generate LLM reasoning over compliance and sensitivity results."""
        
        # Only send FAIL and WARNING findings to the LLM (save tokens)
        flagged = [f for f in compliance.findings
                   if f.overall_status in (CheckStatus.FAIL, CheckStatus.WARNING)]
        
        if not flagged:
            return LLMReasoning(
                executive_summary="All substances pass compliance checks. No corrective actions required.",
                recommendations=[],
                risk_narrative="No regulatory violations or near-threshold values detected.",
                model_used=self.model
            )
        
        # Build the prompt
        findings_json = json.dumps([self._finding_to_dict(f) for f in flagged], indent=2)
        sensitivity_json = json.dumps(sensitivity.high_risk_substances, indent=2)
        
        prompt = FINDING_PROMPT_TEMPLATE.format(
            doc_id=compliance.document_id,
            supplier=compliance.supplier or "Unknown",
            score=compliance.overall_score,
            findings_json=findings_json,
            sensitivity_json=sensitivity_json
        )
        
        logger.info(f"Sending {len(flagged)} findings to LLM ({self.provider}/{self.model})...")
        
        # Call the chosen LLM
        raw_response = self._call_llm(prompt)
        
        # Parse JSON response
        parsed = self._parse_response(raw_response)
        
        # Build typed recommendations
        recommendations = [
            SubstanceRecommendation(
                substance_name      = r.get("substance_name", ""),
                status              = r.get("status", ""),
                plain_english_finding = r.get("plain_english_finding", ""),
                corrective_action   = r.get("corrective_action", ""),
                exemption_check     = r.get("exemption_check", "None identified")
            )
            for r in parsed.get("recommendations", [])
        ]
        
        return LLMReasoning(
            executive_summary = parsed.get("executive_summary", ""),
            recommendations   = recommendations,
            risk_narrative    = parsed.get("risk_narrative", ""),
            model_used        = self.model
        )
    
    def _call_llm(self, prompt: str) -> str:
        """Dispatch to the correct LLM provider."""
        if self.provider == "openai":
            return call_openai(prompt, SYSTEM_PROMPT, self.model, self.temperature,
                               max_tokens=self.max_tokens)
        elif self.provider == "ollama":
            return call_ollama(prompt, SYSTEM_PROMPT, self.model, self.ollama_url)
        elif self.provider in ("qwen", "qwen-openrouter", "openrouter"):
            return call_openrouter(prompt, SYSTEM_PROMPT, self.model, self.temperature,
                             max_tokens=self.max_tokens)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")
    
    def _parse_response(self, raw: str) -> dict:
        """Safely parse the LLM's JSON response."""
        try:
            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[1:])
            if cleaned.endswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[:-1])
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON response — using fallback structure.")
            return {
                "executive_summary": raw[:300] if raw else "LLM response could not be parsed.",
                "risk_narrative": "",
                "recommendations": []
            }
    
    def _finding_to_dict(self, finding: SubstanceFinding) -> dict:
        return {
            "substance": finding.substance_name,
            "cas": finding.cas_number,
            "weight_ppm": finding.weight_ppm,
            "status": finding.overall_status,
            "findings": finding.findings,
            "regulations_triggered": finding.regulation_hits
        }
