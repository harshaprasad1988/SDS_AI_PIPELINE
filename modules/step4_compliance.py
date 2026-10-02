"""Step 4 — Regulatory screening engine.

IMPORTANT: This is a configurable screening engine, not a legal certification.
The built-in lists are a limited demonstration ruleset and must be replaced or
validated against the current applicable regulatory datasets before production use.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
from modules.step3_structurer import StructuredMDS, StructuredSubstance

class CheckStatus(str, Enum):
    PASS="PASS"; FAIL="FAIL"; WARNING="WARNING"; UNKNOWN="UNKNOWN"; REVIEW="REVIEW"

ROHS_THRESHOLDS = {
    "7439-92-1": {"name":"Lead", "threshold_ppm":1000},
    "7439-97-6": {"name":"Mercury", "threshold_ppm":1000},
    "7440-43-9": {"name":"Cadmium", "threshold_ppm":100},
    "18540-29-9": {"name":"Hexavalent Chromium", "threshold_ppm":1000},
    "36483-57-5": {"name":"PBB", "threshold_ppm":1000},
    "32534-81-9": {"name":"PBDE", "threshold_ppm":1000},
}
REACH_SVHC_CAS = {
    "7439-92-1":"Lead", "7440-43-9":"Cadmium", "7440-38-2":"Arsenic",
    "7440-41-7":"Beryllium", "18540-29-9":"Hexavalent Chromium",
    "1332-21-4":"Asbestos", "7782-49-2":"Selenium",
}
REACH_THRESHOLD_PPM=1000
ELV_THRESHOLDS={
    "7439-92-1":{"name":"Lead","threshold_ppm":1000},
    "7439-97-6":{"name":"Mercury","threshold_ppm":1000},
    "7440-43-9":{"name":"Cadmium","threshold_ppm":100},
    "18540-29-9":{"name":"Hexavalent Chromium","threshold_ppm":1000},
}

@dataclass
class RegulatoryCheck:
    regulation: str
    status: CheckStatus
    threshold_ppm: Optional[float]=None
    observed_ppm: Optional[str]=None
    rationale: str=""
    source_basis: str="Configured screening ruleset"

@dataclass
class SubstanceFinding:
    substance_name: str
    cas_number: Optional[str]
    weight_ppm: float
    completeness_status: CheckStatus=CheckStatus.UNKNOWN
    consistency_status: CheckStatus=CheckStatus.UNKNOWN
    regulatory_status: CheckStatus=CheckStatus.UNKNOWN
    findings: List[str]=field(default_factory=list)
    regulation_hits: List[str]=field(default_factory=list)
    regulatory_checks: List[RegulatoryCheck]=field(default_factory=list)
    @property
    def overall_status(self):
        statuses=[self.completeness_status,self.consistency_status,self.regulatory_status]
        if CheckStatus.FAIL in statuses: return CheckStatus.FAIL
        if CheckStatus.REVIEW in statuses or CheckStatus.WARNING in statuses: return CheckStatus.REVIEW
        if CheckStatus.UNKNOWN in statuses: return CheckStatus.UNKNOWN
        return CheckStatus.PASS

@dataclass
class ComplianceReport:
    document_id: str
    supplier: Optional[str]
    findings: List[SubstanceFinding]
    completeness_score: float=0.0
    consistency_score: float=0.0
    regulatory_score: float=0.0
    overall_score: float=0.0
    summary: str=""
    @property
    def fail_count(self): return sum(f.overall_status==CheckStatus.FAIL for f in self.findings)
    @property
    def warn_count(self): return sum(f.overall_status in (CheckStatus.REVIEW,CheckStatus.WARNING) for f in self.findings)
    @property
    def pass_count(self): return sum(f.overall_status==CheckStatus.PASS for f in self.findings)


def _range(s):
    lo=s.weight_ppm_min if s.weight_ppm_min is not None else s.weight_ppm
    hi=s.weight_ppm_max if s.weight_ppm_max is not None else s.weight_ppm
    return lo,hi

def check_completeness(s):
    issues=[]
    if not s.cas_number: issues.append("Missing CAS number — regulatory lookup cannot be completed.")
    if s.weight_fraction_max_pct in (None,0): issues.append("Missing weight fraction — threshold assessment cannot be completed.")
    if not s.name: issues.append("Missing substance name.")
    return (CheckStatus.FAIL if len(issues)>=2 else CheckStatus.WARNING if issues else CheckStatus.PASS), issues

def check_consistency(s, all_substances):
    lo,hi=_range(s); expected=s.weight_ppm
    issues=[]
    if abs(expected - ((lo+hi)/2)) > 1: issues.append("Nominal ppm is inconsistent with the preserved wt-% range.")
    if lo>hi: issues.append("Invalid composition range.")
    return (CheckStatus.FAIL if issues else CheckStatus.PASS), issues

def threshold_check(s, regulation, threshold, name):
    lo,hi=_range(s)
    if lo > threshold:
        return RegulatoryCheck(regulation,CheckStatus.FAIL,threshold,f"{lo:g}-{hi:g}" if s.is_range else f"{lo:g}",f"Entire reported concentration is above {threshold:g} ppm for {name}."), True
    if hi <= threshold:
        return RegulatoryCheck(regulation,CheckStatus.PASS,threshold,f"{lo:g}-{hi:g}" if s.is_range else f"{lo:g}",f"Reported concentration is at or below {threshold:g} ppm for {name}."), False
    return RegulatoryCheck(regulation,CheckStatus.REVIEW,threshold,f"{lo:g}-{hi:g}",f"Reported range crosses the {threshold:g} ppm threshold; exact compliance cannot be determined from the SDS range."), True

def check_regulatory(s, rohs=None, svhc=None, elv=None):
    rohs = ROHS_THRESHOLDS if rohs is None else rohs
    svhc = REACH_SVHC_CAS if svhc is None else svhc
    elv = ELV_THRESHOLDS if elv is None else elv
    if not s.cas_number:
        return CheckStatus.UNKNOWN,["No CAS number — regulatory screening skipped."],[],[]
    checks=[]; findings=[]; hits=[]
    cas=s.cas_number
    if cas in svhc:
        c,h=threshold_check(s,"REACH (SVHC screening)",REACH_THRESHOLD_PPM,svhc[cas]); checks.append(c)
        if c.status!=CheckStatus.PASS: findings.append(c.rationale); hits.append("REACH")
    else:
        checks.append(RegulatoryCheck("REACH (configured SVHC list)",CheckStatus.UNKNOWN,None,None,"Substance is not in the configured SVHC subset; this is not proof of REACH compliance."))
    if cas in rohs:
        r=rohs[cas]; c,h=threshold_check(s,"RoHS",r["threshold_ppm"],r["name"]); checks.append(c)
        if c.status!=CheckStatus.PASS: findings.append(c.rationale); hits.append("RoHS")
    else:
        checks.append(RegulatoryCheck("RoHS (configured restricted list)",CheckStatus.UNKNOWN,None,None,"CAS is not in the configured RoHS subset; this is not proof of RoHS compliance."))
    if cas in elv:
        r=elv[cas]; c,h=threshold_check(s,"ELV",r["threshold_ppm"],r["name"]); checks.append(c)
        if c.status!=CheckStatus.PASS: findings.append(c.rationale); hits.append("ELV")
    else:
        checks.append(RegulatoryCheck("ELV (configured restricted list)",CheckStatus.UNKNOWN,None,None,"CAS is not in the configured ELV subset; this is not proof of ELV compliance."))
    statuses=[c.status for c in checks]
    if CheckStatus.FAIL in statuses: status=CheckStatus.FAIL
    elif CheckStatus.REVIEW in statuses: status=CheckStatus.REVIEW
    elif CheckStatus.UNKNOWN in statuses: status=CheckStatus.UNKNOWN
    else: status=CheckStatus.PASS
    return status,findings,hits,checks

class ComplianceEngine:
    """Deterministic rule-based regulatory screening engine (REACH / RoHS / ELV).

    An optional ``ruleset`` dict can override the built-in thresholds; when
    supplied, :attr:`engine_name` reflects the custom configuration so the UI
    can print the exact evaluation engine that was used.
    """
    def __init__(self, score_weights=None, ruleset=None):
        self.weights=score_weights or {"completeness":.30,"consistency":.25,"regulatory":.45}
        self.ruleset=ruleset
        if ruleset is None:
            self.engine_name="Rule-based deterministic engine (REACH / RoHS / ELV threshold ruleset)"
        else:
            self.engine_name=("Custom rule-based deterministic engine "
                              f"({len(ruleset.get('rohs', ROHS_THRESHOLDS))} RoHS · "
                              f"{len(ruleset.get('reach_svhc', REACH_SVHC_CAS))} REACH SVHC · "
                              f"{len(ruleset.get('elv', ELV_THRESHOLDS))} ELV entries)")
    def run(self,mds):
        findings=[]
        for s in mds.all_substances:
            a,ai=check_completeness(s); b,bi=check_consistency(s,mds.all_substances); c,ci,hits,checks=check_regulatory(
                s,
                rohs=(self.ruleset or {}).get("rohs", ROHS_THRESHOLDS),
                svhc=(self.ruleset or {}).get("reach_svhc", REACH_SVHC_CAS),
                elv=(self.ruleset or {}).get("elv", ELV_THRESHOLDS),
            )
            findings.append(SubstanceFinding(s.name,s.cas_number,s.weight_ppm,a,b,c,ai+bi+ci,hits,checks))
        n=len(findings) or 1
        comp=100*sum(f.completeness_status==CheckStatus.PASS for f in findings)/n
        cons=100*sum(f.consistency_status==CheckStatus.PASS for f in findings)/n
        # Unknown/review is not counted as PASS. This prevents an incomplete ruleset
        # from producing a falsely high compliance score.
        reg=100*sum(f.regulatory_status==CheckStatus.PASS for f in findings)/n
        overall=self.weights["completeness"]*comp+self.weights["consistency"]*cons+self.weights["regulatory"]*reg
        summary=f"Screening score: {overall:.1f}/100. PASS={sum(f.overall_status==CheckStatus.PASS for f in findings)}, REVIEW={sum(f.overall_status==CheckStatus.REVIEW for f in findings)}, FAIL={sum(f.overall_status==CheckStatus.FAIL for f in findings)}."
        return ComplianceReport(mds.document_id,mds.supplier,findings,round(comp,1),round(cons,1),round(reg,1),round(overall,1),summary)
