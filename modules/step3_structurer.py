"""Step 3 — SDS structuring and validation.

Converts Step-2 SDS entities into typed records suitable for compliance and
sensitivity analysis. Composition ranges are preserved instead of forcing an
arbitrary single value.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from modules.step2_nlp import NLPResult, SubstanceEntity

logger = logging.getLogger(__name__)

CAS_TO_NAME: Dict[str, str] = {
    "7439-92-1": "Lead", "7440-43-9": "Cadmium", "7439-97-6": "Mercury",
    "18540-29-9": "Chromium VI (Hexavalent Chromium)", "7440-38-2": "Arsenic",
    "7440-41-7": "Beryllium", "36483-57-5": "Polybrominated Biphenyl (PBB)",
    "32534-81-9": "Polybrominated Diphenyl Ether (PBDE)", "10124-36-4": "Cadmium Sulfate",
    "7440-31-5": "Tin", "1332-21-4": "Asbestos (Chrysotile)", "7782-49-2": "Selenium",
}

@dataclass
class StructuredSubstance:
    name: str
    cas_number: Optional[str]
    weight_fraction_pct: float
    weight_fraction_min_pct: Optional[float]
    weight_fraction_max_pct: Optional[float]
    weight_ppm: float
    weight_ppm_min: Optional[float]
    weight_ppm_max: Optional[float]
    is_range: bool = False
    is_hazardous_candidate: bool = False
    source_confidence: float = 1.0

@dataclass
class StructuredComponent:
    component_name: str
    substances: List[StructuredSubstance]
    total_weight_pct: float
    completeness_ok: bool = True

@dataclass
class SDSProfile:
    ecological_information: str = "Not detected"
    disposal_considerations: str = "Not detected"
    toxicological_information: str = "Not detected"
    stability_and_reactivity: str = "Not detected"

@dataclass
class StructuredMDS:
    document_id: str
    supplier: Optional[str]
    components: List[StructuredComponent]
    all_substances: List[StructuredSubstance]
    sds_profile: SDSProfile = field(default_factory=SDSProfile)
    structuring_warnings: List[str] = field(default_factory=list)


def _parse_range(value: Optional[str]) -> Tuple[float, float, bool]:
    if not value:
        return 0.0, 0.0, False
    raw = str(value).strip().replace("–", "-").replace(" ", "")
    raw = raw.replace("<", "")
    try:
        if "-" in raw:
            a, b = raw.split("-", 1)
            return float(a), float(b), True
        x = float(raw)
        return x, x, False
    except ValueError:
        return 0.0, 0.0, False


def _section(text: str, number: int, next_number: Optional[int] = None) -> str:
    import re
    # Supports both 'SECTION N:' (Format B) and 'N.' (Format A) headings.
    if next_number:
        pat_b = rf"(?is)SECTION\s+{number}\s*:[^\n]*(.*?)(?=SECTION\s+{next_number}\s*:|\Z)"
        pat_a = rf"(?ims)^\s*{number}\.\s+[^\n]+\n(.*?)(?=^\s*{next_number}\.\s+[^\n]+$|\Z)"
    else:
        pat_b = rf"(?is)SECTION\s+{number}\s*:[^\n]*(.*?)$"
        pat_a = rf"(?ims)^\s*{number}\.\s+[^\n]+\n(.*)$"
    for pat in (pat_b, pat_a):
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return ""


def _summary(text: str, section_no: int, next_no: int, max_chars: int = 650) -> str:
    sec = _section(text, section_no, next_no)
    if not sec:
        return "Not detected"
    import re
    lines = [re.sub(r"\s+", " ", x).strip() for x in sec.splitlines() if x.strip()]
    # Remove page/header noise and keep the first useful descriptive sentences.
    lines = [x for x in lines if not re.match(r"^Page \d+ / \d+$", x, re.I)]
    out = " ".join(lines)
    return out[:max_chars] + ("…" if len(out) > max_chars else "")


class DataStructurer:
    def __init__(self, cas_lookup: Dict[str, str] = None):
        self.cas_lookup = cas_lookup or CAS_TO_NAME

    def structure(self, nlp_result: NLPResult, doc_id: str = "UNKNOWN") -> StructuredMDS:
        warnings = list(nlp_result.extraction_warnings)
        resolved = [self._resolve_substance(s) for s in nlp_result.substances]
        resolved = self._deduplicate(resolved)
        total_max = sum(s.weight_fraction_max_pct or 0 for s in resolved)
        total_nominal = sum(s.weight_fraction_pct for s in resolved)
        if total_max > 100:
            warnings.append(f"Maximum extracted composition ({total_max:.2f}%) exceeds 100%; review extraction/template alignment.")
        if any(s.is_range for s in resolved):
            warnings.append("Composition ranges are preserved. Compliance is marked REVIEW when a range crosses a regulatory threshold.")
        component = StructuredComponent(
            component_name="SDS Product Composition",
            substances=resolved,
            total_weight_pct=total_nominal,
            completeness_ok=bool(resolved),
        )
        text = getattr(nlp_result, "raw_text", "") or ""
        profile = SDSProfile()
        if text:
            profile = SDSProfile(
                ecological_information=_summary(text, 12, 13),
                disposal_considerations=_summary(text, 13, 14),
                toxicological_information=_summary(text, 11, 12),
                stability_and_reactivity=_summary(text, 10, 11),
            )
        return StructuredMDS(
            document_id=doc_id,
            supplier=nlp_result.supplier_info.supplier_name,
            components=[component],
            all_substances=resolved,
            sds_profile=profile,
            structuring_warnings=warnings,
        )

    def _resolve_substance(self, entity: SubstanceEntity) -> StructuredSubstance:
        name = entity.name
        if entity.cas_number and entity.cas_number in self.cas_lookup:
            name = self.cas_lookup[entity.cas_number]
        lo, hi, is_range = _parse_range(entity.weight_fraction_pct)
        # If only a single value is available, nominal is that value. For a range,
        # use midpoint only for display/uncertainty calculations; never replace the range.
        nominal = (lo + hi) / 2 if is_range else lo
        ppm_lo, ppm_hi = lo * 10000, hi * 10000
        return StructuredSubstance(
            name=name, cas_number=entity.cas_number,
            weight_fraction_pct=nominal,
            weight_fraction_min_pct=lo, weight_fraction_max_pct=hi,
            weight_ppm=nominal * 10000,
            weight_ppm_min=ppm_lo, weight_ppm_max=ppm_hi,
            is_range=is_range,
            is_hazardous_candidate=bool(entity.cas_number and entity.cas_number in self.cas_lookup),
            source_confidence=entity.confidence,
        )

    def _deduplicate(self, substances: List[StructuredSubstance]) -> List[StructuredSubstance]:
        unique = {}
        for s in sorted(substances, key=lambda x: x.source_confidence, reverse=True):
            key = s.cas_number or s.name.lower()
            if key not in unique:
                unique[key] = s
        return list(unique.values())
