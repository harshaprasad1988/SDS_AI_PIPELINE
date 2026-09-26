"""Step 5 — Concentration uncertainty and regulatory sensitivity analysis."""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from modules.step3_structurer import StructuredSubstance
from modules.step4_compliance import ROHS_THRESHOLDS, ELV_THRESHOLDS, REACH_SVHC_CAS, REACH_THRESHOLD_PPM

@dataclass
class ThresholdRisk:
    regulation:str; threshold_ppm:float; exceedance_probability:float
    ci_lower_ppm:float; ci_upper_ppm:float; risk_level:str
@dataclass
class SensitivityResult:
    substance_name:str; cas_number:Optional[str]; nominal_ppm:float
    reported_range:str; uncertainty_pct:float; threshold_risks:List[ThresholdRisk]=field(default_factory=list)
    most_sensitive_regulation:Optional[str]=None
@dataclass
class SensitivityReport:
    results:List[SensitivityResult]; high_risk_substances:List[str]=field(default_factory=list)

def _risk(p, nominal, threshold):
    if nominal>threshold: return "CRITICAL"
    if p>.30: return "HIGH"
    if p>.05 or nominal/threshold>.80: return "MEDIUM"
    return "LOW"

def _mc(lo,hi,threshold,n=2048):
    # Uniform distribution is deliberately used for an SDS-declared range:
    # the system does not pretend to know where inside the supplier range the
    # true concentration lies.
    rng=np.random.default_rng(42); samples=rng.uniform(lo,hi,n)
    p=float(np.mean(samples>threshold))
    return p,float(np.percentile(samples,2.5)),float(np.percentile(samples,97.5))

class SensitivityAnalyser:
    def __init__(self,uncertainty_pct=.10,n_samples=2048): self.uncertainty_pct=uncertainty_pct; self.n_samples=n_samples
    def analyse(self,substances):
        results=[]; high=[]
        for s in substances:
            lo=s.weight_ppm_min if s.weight_ppm_min is not None else s.weight_ppm
            hi=s.weight_ppm_max if s.weight_ppm_max is not None else s.weight_ppm
            thresholds=[]
            if s.cas_number in ROHS_THRESHOLDS: thresholds.append(("RoHS",ROHS_THRESHOLDS[s.cas_number]["threshold_ppm"]))
            if s.cas_number in ELV_THRESHOLDS: thresholds.append(("ELV",ELV_THRESHOLDS[s.cas_number]["threshold_ppm"]))
            if s.cas_number in REACH_SVHC_CAS: thresholds.append(("REACH (SVHC screening)",REACH_THRESHOLD_PPM))
            risks=[]
            for reg,t in thresholds:
                p,cl,cu=_mc(lo,hi,t,self.n_samples)
                nominal=(lo+hi)/2
                risks.append(ThresholdRisk(reg,t,round(p,4),round(cl,2),round(cu,2),_risk(p,nominal,t)))
            worst=max(risks,key=lambda x:x.exceedance_probability).regulation if risks else None
            sr=SensitivityResult(s.name,s.cas_number,s.weight_ppm,f"{lo:g}-{hi:g} ppm" if s.is_range else f"{lo:g} ppm",self.uncertainty_pct,risks,worst)
            results.append(sr)
            if any(r.risk_level in ("HIGH","CRITICAL") for r in risks): high.append(s.name)
        return SensitivityReport(results,high)
