"""
services/combination_synergy.py
Chou-Talalay Combination Index (CI) engine based on the median-effect equation.

For each drug (and for the fixed-ratio combination) the median-effect equation
    fa / fu = (D / Dm)^m
is fitted to dose-response points by linear regression of log(fa/fu) on log(D). The dose that alone
produces effect fa is then  Dx = Dm * (fa / (1 - fa))^(1/m),  and the combination index is
    CI = d1/Dx1 + d2/Dx2 + alpha * (d1 * d2) / (Dx1 * Dx2)
with alpha = 0 for mutually exclusive drugs (same site) and alpha = 1 for mutually non-exclusive drugs.

Interpretation: CI < 0.3 very strong synergism · 0.3-0.7 strong · 0.7-0.9 moderate · 0.9-1.1 additive · > 1.1 antagonism.

Dose-response inputs come from data/dose_response_illustrative.json. Those points are ILLUSTRATIVE (see the file's
_note): the computation is real, the inputs are not measurements. Clinical outcome fields (mPFS, ORR, HR) are the
published trial results.
"""

import json
import os
from typing import Any, Dict, List

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSE_RESPONSE_FILE = os.path.join(BASE_DIR, "data", "dose_response_illustrative.json")
FA_LEVELS = (0.5, 0.75, 0.9)

# Published clinical outcomes shown next to each regimen (not used in the CI calculation).
TRIAL_OUTCOMES = {
    "Pair_A_MET_Bypass_Blockade": {
        "name": "Osimertinib + Savolitinib", "target": "EGFR Cys797 + c-MET Tyrosine Kinase",
        "trial": "SAVANNAH Phase II (PMID: 40461383)", "mPFS_months": 7.5, "orr_pct": 56.3,
    },
    "Pair_B_Bispecific_Dual_Receptor_Blockade": {
        "name": "Amivantamab + Lazertinib", "target": "Bispecific EGFR/MET Ectodomain + CNS-Penetrant 3rd-Gen TKI",
        "trial": "MARIPOSA Phase III (PMID: 38924756)", "mPFS_months": 23.7, "orr_pct": 86.0, "hazard_ratio": 0.70,
    },
    "Gefitinib_Osimertinib": {
        "name": "Osimertinib + Gefitinib", "target": "T790M/C797S allele phasing",
        "trial": "Zhou et al. (PMID: 33958875)",
    },
}


def categorize(ci: float) -> str:
    if ci < 0.3:
        return "Very Strong Synergism"
    if ci < 0.7:
        return "Strong Synergism"
    if ci < 0.9:
        return "Moderate Synergism"
    if ci <= 1.1:
        return "Additive Effect"
    return "Antagonism"


def fit_median_effect(doses: List[float], fa: List[float]) -> Dict[str, float]:
    """Least-squares fit of log(fa/fu) = m*log(D) - m*log(Dm). Returns Dm, m and the linear r."""
    d = np.asarray(doses, dtype=float)
    f = np.clip(np.asarray(fa, dtype=float), 1e-4, 1 - 1e-4)
    x, y = np.log(d), np.log(f / (1 - f))
    m, c = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return {"Dm": float(np.exp(-c / m)), "m": float(m), "r": r}


def dose_for_effect(fit: Dict[str, float], fa: float) -> float:
    return fit["Dm"] * (fa / (1 - fa)) ** (1 / fit["m"])


class CombinationSynergyEngine:

    @staticmethod
    def calculate_ci(d1: float, dx1: float, d2: float, dx2: float, alpha: float = 1.0) -> Dict[str, Any]:
        """Chou-Talalay Combination Index for doses d1, d2 given the single-agent equi-effective doses Dx1, Dx2."""
        term1 = d1 / max(dx1, 1e-12)
        term2 = d2 / max(dx2, 1e-12)
        term3 = alpha * term1 * term2
        ci = term1 + term2 + term3
        return {
            "combination_index": round(ci, 3),
            "category": categorize(ci),
            "is_synergistic": ci < 0.9,
            "formula_terms": {"term_d1": round(term1, 3), "term_d2": round(term2, 3), "term_interaction": round(term3, 3)},
        }

    @staticmethod
    def ci_from_dose_response(spec: Dict[str, Any], fa_levels=FA_LEVELS) -> Dict[str, Any]:
        """Fits the three median-effect curves in `spec` and returns CI at each fa level (Fa-CI table)."""
        f1 = fit_median_effect(spec["drug_1"]["dose"], spec["drug_1"]["fa"])
        f2 = fit_median_effect(spec["drug_2"]["dose"], spec["drug_2"]["fa"])
        combo = spec["combination"]
        fc = fit_median_effect(combo["total_dose"], combo["fa"])
        ratio = combo["ratio_d1_to_d2"]
        p1, p2 = ratio / (1 + ratio), 1 / (1 + ratio)

        table = []
        for fa in fa_levels:
            dc = dose_for_effect(fc, fa)
            res = CombinationSynergyEngine.calculate_ci(
                dc * p1, dose_for_effect(f1, fa), dc * p2, dose_for_effect(f2, fa), spec.get("alpha", 1.0))
            table.append({"fa": fa, **res})
        return {
            "fits": {"drug_1": f1, "drug_2": f2, "combination": fc},
            "fa_ci": table,
            "ci_at_ed50": table[0]["combination_index"],
            "context": spec.get("context", ""),
            "alpha": spec.get("alpha", 1.0),
        }

    @staticmethod
    def load_dose_response() -> Dict[str, Any]:
        with open(DOSE_RESPONSE_FILE, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def evaluate_regimen(pair_id: str, is_cis_c797s: bool = False, is_met_amplified: bool = False) -> Dict[str, Any]:
        """Computes the CI for a regimen in the patient's clonal context and attaches the trial outcomes."""
        data = CombinationSynergyEngine.load_dose_response()["regimens"]
        key = pair_id
        if pair_id.startswith("Gefitinib_Osimertinib"):
            key = "Gefitinib_Osimertinib|cis" if is_cis_c797s else "Gefitinib_Osimertinib|trans"
        elif "SAVANNAH" in pair_id:
            key = "Pair_A_MET_Bypass_Blockade"
        elif "MARIPOSA" in pair_id:
            key = "Pair_B_Bispecific_Dual_Receptor_Blockade"

        calc = CombinationSynergyEngine.ci_from_dose_response(data[key])
        ci = calc["ci_at_ed50"]
        base_id = key.split("|")[0]
        meta = TRIAL_OUTCOMES[base_id]

        if base_id == "Gefitinib_Osimertinib":
            verdict = ("CONTRAINDICATED: antagonistic against cis-C797S; route to 4th-Gen TKI" if ci > 1.1
                       else "TRANSIENT_BENEFIT: covers trans-C797S until a cis clonal shift")
        elif base_id == "Pair_A_MET_Bypass_Blockade":
            verdict = "RECOMMENDED FOR MET-AMPLIFIED BYPASS" if is_met_amplified else "CONSIDER IF MET AMPLIFICATION IS CONFIRMED"
        else:
            verdict = "FIRST_LINE_SUPERIOR_OVERALL_SURVIVAL"

        return {
            **meta,
            "ci": ci,
            "category": f"{categorize(ci)} (CI = {ci})",
            "verdict": verdict,
            "calculation": calc,
        }
