"""
STEP 2 - SDS targeted extraction (NER + deterministic table/section parsing).

Supports TWO common SDS formats:
  FORMAT A (numbered dot):  "1. IDENTIFICATION" / "Supplier Address\nCompany\nStreet…"
  FORMAT B (SECTION colon): "SECTION 1: IDENTIFICATION" / "Supplier\nCompany\nStreet…"
                             CAS presented as "CAS-No.: NNNN-NN-N" on its own line.

Extraction is restricted to:
  Table 1: product/supplier/contact from Section 1.
  Table 2: substance/CAS/composition from Section 3.

Ranges are preserved exactly as printed.
"""
import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List
from modules.step1_ocr import OCRResult

logger = logging.getLogger(__name__)

# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class SubstanceEntity:
    name: str
    cas_number: Optional[str] = None
    weight_fraction_pct: Optional[str] = None
    weight_value: Optional[str] = None
    weight_unit: Optional[str] = None
    confidence: float = 0.0
    source_text: str = ""
    page_number: Optional[int] = None

@dataclass
class SupplierInfo:
    product_name: Optional[str] = None
    supplier_name: Optional[str] = None
    supplier_address: Optional[str] = None
    supplier_contact: Optional[str] = None

@dataclass
class NLPResult:
    supplier_info: SupplierInfo
    substances: List[SubstanceEntity]
    extraction_warnings: List[str] = field(default_factory=list)
    source_section: str = "SDS Section 1 and Section 3"
    raw_text: str = ""

    @property
    def suppliers(self):
        return [self.supplier_info.supplier_name] if self.supplier_info.supplier_name else []

    @property
    def components(self):
        return []

    @property
    def raw_entity_count(self):
        return len(self.substances)

# ── Regex constants ───────────────────────────────────────────────────────────

CAS_PATTERN  = re.compile(r"\b\d{1,7}-\d{2}-\d\b")
PHONE_PATTERN = re.compile(r"(?:(?:\+?\d[\d\s().-]{6,}\d))")
WEIGHT_RANGE  = r"(?:<\s*)?\d+(?:\.\d+)?(?:\s*[-–]\s*(?:<\s*)?\d+(?:\.\d+)?)?"
WT_PATTERN    = re.compile(rf"({WEIGHT_RANGE})\s*(?:wt\.?\s*)?%", re.I)
PPM_PATTERN   = re.compile(rf"({WEIGHT_RANGE})\s*ppm\b", re.I)
MG_M3_PATTERN = re.compile(rf"({WEIGHT_RANGE})\s*mg\s*/\s*m(?:3|³)", re.I)

# Lines that are table header artefacts to skip during composition parsing
_SEC3_SKIP = {
    "name", "product identifier", "ghs classification", "%",
    "composition comments", "the information given in this safety data sheet is for guidance only.",
    "substance", "cas number", "cas no", "weight fraction",
}

def _norm(s):
    return re.sub(r"[ \t]+", " ", s or "").strip()

# ── Format detection ──────────────────────────────────────────────────────────

def _detect_format(text: str) -> str:
    """Return 'B' if SDS uses 'SECTION N:' headings, else 'A' (uses 'N.' headings)."""
    if re.search(r"(?i)SECTION\s+\d+\s*:", text):
        return "B"
    return "A"

# ── Section extraction — handles both formats ─────────────────────────────────

def _section(text: str, number: int, next_number: Optional[int] = None) -> str:
    """Extract section N text, supporting both 'N.' and 'SECTION N:' headings."""
    # Format B: SECTION N: heading
    if next_number:
        pat_b = rf"(?is)SECTION\s+{number}\s*:[^\n]*(.*?)(?=SECTION\s+{next_number}\s*:|\Z)"
        pat_a = rf"(?is)\b{number}\.\s+[A-Z][^\n]*?(.*?)(?=\n\s*{next_number}\.\s+[A-Z])"
    else:
        pat_b = rf"(?is)SECTION\s+{number}\s*:[^\n]*(.*?)$"
        pat_a = rf"(?is)\b{number}\.\s+[A-Z][^\n]*(.*)$"

    for pat in (pat_b, pat_a):
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return ""

def _section_summary(text: str, section_no: int, next_no: int, max_chars: int = 650) -> str:
    sec = _section(text, section_no, next_no)
    if not sec:
        return "Not detected"
    lines = [re.sub(r"\s+", " ", x).strip() for x in sec.splitlines() if x.strip()]
    lines = [x for x in lines if not re.match(r"^Page \d+", x, re.I)]
    out = " ".join(lines)
    return out[:max_chars] + ("…" if len(out) > max_chars else "")

def _page_for_text(ocr, snippet):
    if not snippet:
        return None
    for p in ocr.pages:
        if snippet in p.raw_text:
            return p.page_number
    return None

# ── Supplier / Section 1 parsing ──────────────────────────────────────────────

def _parse_supplier_format_a(sec1: str) -> SupplierInfo:
    """Format A: 'Supplier Address' label followed by company/address lines."""
    info = SupplierInfo()
    lines = [_norm(x) for x in sec1.splitlines() if _norm(x)]

    for i, line in enumerate(lines):
        low = line.lower()
        if low.startswith("product name"):
            info.product_name = _norm(re.sub(r"^product name\s*", "", line, flags=re.I)).lstrip(":")
        elif low.startswith("supplier address"):
            following = []
            for nxt in lines[i+1:]:
                nlow = nxt.lower()
                if re.match(r"^(emergency telephone|company phone|telephone|phone|fax|email|e-mail|recommended use|sds\s*#)", nlow):
                    break
                if nxt:
                    following.append(nxt)
            if following:
                if not info.supplier_name:
                    info.supplier_name = following[0]
                if len(following) > 1:
                    info.supplier_address = ", ".join(following[1:])
        elif low.startswith("company phone number"):
            info.supplier_contact = _norm(re.sub(r"^company phone number\s*", "", line, flags=re.I)).lstrip(":")
        elif low.startswith("phone") or low.startswith("telephone"):
            candidate = _norm(re.sub(r"^(?:phone|telephone)\s*", "", line, flags=re.I)).lstrip(":")
            if candidate and any(ch.isdigit() for ch in candidate):
                info.supplier_contact = candidate

    # Fallbacks
    if not info.product_name:
        m = re.search(r"(?im)^\s*Product Name\s*:?\s*(.+)$", sec1)
        if m: info.product_name = _norm(m.group(1))
    if not info.supplier_contact:
        m = re.search(r"(?im)^\s*(?:Company Phone Number|Phone|Telephone)\s*:?\s*(.+)$", sec1)
        if m: info.supplier_contact = _norm(m.group(1))
    if not info.supplier_address:
        m = re.search(r"(?is)Supplier Address\s*(.*?)(?=\b(?:Emergency telephone|Company Phone Number|Recommended use)\b)", sec1)
        if m:
            info.supplier_address = _norm(m.group(1).replace("\n", ", ")).strip(" ,:")
    return info


def _parse_supplier_format_b(sec1: str) -> SupplierInfo:
    """
    Format B: key-value pairs on alternating lines.
    e.g.  'Product Name\\nGMW 14872 TESTING SOLUTION\\nSupplier\\nAuto Technology\\n...'
    """
    info = SupplierInfo()
    lines = [_norm(x) for x in sec1.splitlines() if _norm(x)]

    i = 0
    while i < len(lines):
        low = lines[i].lower()

        # Key-on-one-line, value-on-next-line pattern
        if low in ("product name", "product code"):
            if i + 1 < len(lines):
                if low == "product name":
                    info.product_name = lines[i+1]
            i += 2; continue

        if low in ("supplier", "company", "manufacturer"):
            # Collect following lines until a recognisable field label
            addr_lines = []
            j = i + 1
            while j < len(lines):
                nl = lines[j].lower()
                if re.match(r"^(contact person|emergency telephone|product code|identifier|product name|section\s+\d|revision)", nl):
                    break
                addr_lines.append(lines[j])
                j += 1
            if addr_lines:
                info.supplier_name = addr_lines[0]
                # Collect address (non-phone lines)
                addr_parts = []
                for al in addr_lines[1:]:
                    if re.search(r"tel\s*[:(]|fax\s*[:(]|phone\s*[:(]|\+\d|\(\d{3}\)", al, re.I):
                        if not info.supplier_contact:
                            # strip label
                            info.supplier_contact = re.sub(r"^Tel\s*:\s*", "", al, flags=re.I).strip()
                    else:
                        addr_parts.append(al)
                if addr_parts:
                    info.supplier_address = ", ".join(addr_parts)
            i = j; continue

        if low in ("contact person", "contact"):
            if i + 1 < len(lines):
                val = lines[i+1]
                # If the next line looks like email or phone, use it directly
                if "@" in val or any(ch.isdigit() for ch in val):
                    info.supplier_contact = val
                else:
                    # Contact Person is a person; next might be email
                    info.supplier_contact = val
            i += 2; continue

        if re.match(r"^(emergency telephone|emergency phone)", low):
            # skip — emergency line not the regular contact
            i += 2; continue

        i += 1

    # Inline fallbacks for lines like "Tel: (1-440-572-7800)" that may appear
    # inside the supplier block rather than as a separate key-value pair
    if not info.supplier_contact:
        m = re.search(r"(?im)^.*?Tel\s*:\s*(.+)$", sec1)
        if m:
            info.supplier_contact = _norm(m.group(1))
    if not info.product_name:
        m = re.search(r"(?im)^Product Name\s*\n(.+)$", sec1)
        if m:
            info.product_name = _norm(m.group(1))

    return info


def _parse_supplier_section(sec1: str, fmt: str) -> SupplierInfo:
    if fmt == "B":
        return _parse_supplier_format_b(sec1)
    return _parse_supplier_format_a(sec1)

# ── Composition / Section 3 parsing ──────────────────────────────────────────

def _convert_pct_to_ppm(pct: str):
    raw = pct.replace(" ", "").replace("–", "-").replace("<", "")
    try:
        if "-" in raw:
            a, b = raw.split("-", 1)
            return f"{float(a)*10000:g}-{float(b)*10000:g}", "ppm"
        return f"{float(raw)*10000:g}", "ppm"
    except ValueError:
        return None, None

def _weight_value_from_row(text: str):
    ppm = PPM_PATTERN.search(text)
    if ppm:
        return ppm.group(1).replace(" ", ""), "ppm"
    mg = MG_M3_PATTERN.search(text)
    if mg:
        return mg.group(1).replace(" ", ""), "mg/m3"
    wt = WT_PATTERN.search(text)
    if wt:
        return _convert_pct_to_ppm(wt.group(1))
    return None, None

def _is_table_header_line(line: str) -> bool:
    return line.lower().strip() in _SEC3_SKIP

def _parse_composition_format_b(sec3: str) -> List[SubstanceEntity]:
    """
    Format B layout (lines):
      <substance name>
      CAS-No.: NNNN-NN-N
      EC No.: NNN-NNN-N          ← skip
      [GHS classification]        ← optional, skip
      NN-NN%                     ← weight
    """
    lines = [_norm(x) for x in sec3.splitlines() if _norm(x)]
    substances = []
    seen_cas = set()

    i = 0
    while i < len(lines):
        line = lines[i]
        cas_prefix = re.search(r"CAS-No\.:\s*(\d{1,7}-\d{2}-\d)", line)
        if not cas_prefix:
            i += 1; continue

        cas = cas_prefix.group(1)
        if cas in seen_cas:
            i += 1; continue

        # Substance name = previous non-header, non-skip line
        name = None
        for back in range(i-1, max(i-4, -1), -1):
            candidate = lines[back]
            if _is_table_header_line(candidate):
                continue
            if re.search(r"CAS-No\.|EC No\.|revision date", candidate, re.I):
                continue
            # Must look like a chemical name (not a pure number/pct)
            if re.fullmatch(r"[\d\s%<>\-–.]+", candidate):
                continue
            name = candidate
            break

        if not name:
            i += 1; continue

        # Search forward for weight (skip EC/GHS lines, stop at next CAS or section)
        wt_pct = None
        j = i + 1
        while j < len(lines) and j < i + 6:
            nxt = lines[j]
            if re.search(r"CAS-No\.|SECTION\s+\d", nxt, re.I):
                break
            if re.search(r"EC No\.", nxt, re.I):
                j += 1; continue
            wt_m = WT_PATTERN.search(nxt)
            if wt_m:
                wt_pct = wt_m.group(1).replace(" ", "")
                break
            j += 1

        ppm_val, ppm_unit = _convert_pct_to_ppm(wt_pct) if wt_pct else (None, None)
        conf = 0.97 if wt_pct else 0.80

        seen_cas.add(cas)
        substances.append(SubstanceEntity(
            name=name.strip(" |:-"),
            cas_number=cas,
            weight_fraction_pct=wt_pct,
            weight_value=ppm_val,
            weight_unit=ppm_unit or "ppm",
            confidence=conf,
            source_text=f"{name} | {cas} | {wt_pct or 'N/A'}%",
        ))
        i += 1

    return substances


def _parse_composition_format_a(sec3: str) -> List[SubstanceEntity]:
    """
    Format A: consecutive triplets (name / bare CAS / wt) or single-line rows.
    """
    lines = [_norm(x) for x in sec3.splitlines() if _norm(x)]
    substances = []

    # Triplet scan: name | CAS_fullmatch | wt_fullmatch
    for i in range(len(lines) - 2):
        name, cas_line, wt_line = lines[i:i+3]
        cas = CAS_PATTERN.fullmatch(cas_line)
        wt  = re.fullmatch(rf"({WEIGHT_RANGE})\s*%?", wt_line, re.I)
        if cas and wt:
            pct = wt.group(1).replace(" ", "")
            substances.append(SubstanceEntity(
                name=name.strip(" |:-"),
                cas_number=cas.group(0),
                weight_fraction_pct=pct,
                weight_value=_convert_pct_to_ppm(pct)[0],
                weight_unit="ppm",
                confidence=0.99,
                source_text=f"{name} | {cas.group(0)} | {wt_line}",
            ))

    # Single-line / table-row scan
    for line in lines:
        cas_match = CAS_PATTERN.search(line)
        if not cas_match:
            continue
        cas = cas_match.group(0)
        wt  = WT_PATTERN.search(line)
        before = _norm(line[:cas_match.start()]).strip("|:-")
        if not before:
            continue
        if any(s.cas_number == cas for s in substances):
            continue
        pct = wt.group(1).replace(" ", "") if wt else None
        value, unit = _weight_value_from_row(line)
        substances.append(SubstanceEntity(
            name=before, cas_number=cas, weight_fraction_pct=pct,
            weight_value=value, weight_unit=unit,
            confidence=0.96 if pct else 0.85, source_text=line
        ))

    # Deduplicate by CAS
    unique = {}
    for s in substances:
        old = unique.get(s.cas_number)
        if old is None or s.confidence > old.confidence:
            unique[s.cas_number] = s
    return list(unique.values())


def _parse_composition(sec3: str, fmt: str) -> List[SubstanceEntity]:
    if fmt == "B":
        subs = _parse_composition_format_b(sec3)
        # If format B parser found nothing, fall back to format A
        if not subs:
            subs = _parse_composition_format_a(sec3)
        return subs
    return _parse_composition_format_a(sec3)

# ── Main extraction entry point ───────────────────────────────────────────────

def extract(ocr_result: OCRResult) -> NLPResult:
    text = ocr_result.full_text
    warnings = []

    fmt = _detect_format(text)
    logger.info(f"Detected SDS format: {fmt}")

    sec1 = _section(text, 1, 2)
    sec3 = _section(text, 3, 4)

    if not sec1:
        warnings.append("SDS Section 1 (Identification) was not confidently located.")
    if not sec3:
        warnings.append("SDS Section 3 (Composition/Information on Ingredients) was not confidently located.")

    info = _parse_supplier_section(sec1, fmt)
    substances = _parse_composition(sec3, fmt)

    # Attach page numbers
    for s in substances:
        s.page_number = (
            _page_for_text(ocr_result, s.source_text)
            or _page_for_text(ocr_result, s.cas_number or "")
        )

    # Warnings
    if not info.product_name:
        warnings.append("Product Name not detected.")
    if not info.supplier_name:
        warnings.append("Supplier Name not detected.")
    if not info.supplier_address:
        warnings.append("Supplier Address not detected.")
    if not info.supplier_contact:
        warnings.append("Supplier contact/phone not detected.")
    if not substances:
        warnings.append("No composition substances were detected in SDS Section 3.")
    if sec3 and re.search(r"trade secret", sec3, re.I):
        warnings.append("The SDS states a chemical identity/percentage may be withheld as trade secret.")
    if any(s.weight_fraction_pct and "-" in s.weight_fraction_pct for s in substances):
        warnings.append("One or more composition values are ranges. Exact concentration is not inferred.")
    warnings.append("ppm values in Table 2 are derived from SDS weight-% when ppm is not explicitly stated.")
    warnings.append("mg/m3 exposure limits from SDS Section 8 are not composition weights and are not included in Table 2.")

    return NLPResult(info, substances, warnings, raw_text=text)


class NLPExtractor:
    def __init__(self):
        pass
    def extract(self, ocr_result):
        return extract(ocr_result)
