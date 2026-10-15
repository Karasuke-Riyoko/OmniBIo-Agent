"""
services/spatial.py
LDV Labs spatial multi-omics layer: escape scoring, Getis-Ord Gi* hotspot detection and a spatial
treatment flag, computed from per-spot expression.

Input spots come from data/visium_demo_spots.json. That file is SIMULATED (scripts/simulate_visium_demo.py places
Visium-geometry spots on the H&E tile and draws expression from documented rules); it is not a measured
10x Visium sample. Point SPOTS_FILE at a real Visium export (barcode, pixel x/y, log2 UMI per gene) to use real data.
"""

import json
import os
from typing import Any, Dict, List

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPOTS_FILE = os.path.join(BASE_DIR, "data", "visium_demo_spots.json")

ESCAPE_THRESHOLD = 0.75
# Logistic escape model on log2 UMI: bypass (MET), EMT (AXL) and immune shield (CD274) raise escape;
# retained EGFR addiction lowers it.
ESCAPE_WEIGHTS = {"MET": 0.9, "AXL": 0.6, "CD274": 0.3, "EGFR": -0.4}
ESCAPE_BIAS = -8.0
MET_HIGH_LOG2 = 8.0


def escape_score(expr: Dict[str, float]) -> float:
    z = ESCAPE_BIAS + sum(w * expr.get(g, 0.0) for g, w in ESCAPE_WEIGHTS.items())
    return float(1.0 / (1.0 + np.exp(-z)))


def alert_status(expr: Dict[str, float], score: float, hotspot: bool) -> str:
    if score < 0.5:
        return "BENIGN_STROMA" if expr.get("EGFR", 0) < 4 else "SENSITIVE_EGFR_ADDICTED"
    if expr.get("MET", 0) >= MET_HIGH_LOG2 and expr.get("AXL", 0) >= 5:
        status = "MET_AXL_COESCAPE"
    elif expr.get("MET", 0) >= MET_HIGH_LOG2:
        status = "MET_BYPASS"
    elif expr.get("AXL", 0) >= 6:
        status = "AXL_EMT"
    else:
        status = "RISING_ESCAPE"
    return status + ("_HOTSPOT" if hotspot else "")


def getis_ord_gi_star(xy: np.ndarray, values: np.ndarray, band: float) -> np.ndarray:
    """Gi* z-scores with binary distance-band weights (self included)."""
    n = len(values)
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    w = (d <= band).astype(float)
    x_bar, s = values.mean(), values.std()
    wsum = w.sum(1)
    num = w @ values - x_bar * wsum
    den = s * np.sqrt((n * (w ** 2).sum(1) - wsum ** 2) / (n - 1))
    return num / np.where(den > 0, den, 1.0)


def analyze(path: str = SPOTS_FILE) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    spots: List[Dict[str, Any]] = data["spots"]
    xy = np.array([[s["pxl_x"], s["pxl_y"]] for s in spots], dtype=float)
    scores = np.array([escape_score(s["expression_log2_umi"]) for s in spots])
    gi = getis_ord_gi_star(xy, scores, band=data["center_to_center_px"] * 1.5)

    for s, sc, z in zip(spots, scores, gi):
        hot = bool(z > 1.96 and sc >= ESCAPE_THRESHOLD)
        s["escape_score"] = round(float(sc), 3)
        s["gi_star_z"] = round(float(z), 2)
        s["hotspot"] = hot
        s["alert_status"] = alert_status(s["expression_log2_umi"], sc, hot)

    hot = [s for s in spots if s["hotspot"]]
    met_hot = [s for s in hot if s["expression_log2_umi"].get("MET", 0) >= MET_HIGH_LOG2]
    if met_hot:
        recommendation = {
            "spatial_met_niche": True,
            "summary": (f"{len(met_hot)} MET-high escape hotspot spots (Gi* z > 1.96). Confirm MET amplification on "
                        "tissue (FISH or IHC) before adding a MET inhibitor such as Savolitinib."),
        }
    else:
        recommendation = {"spatial_met_niche": False,
                          "summary": "No significant MET-high escape hotspot; no spatial case for MET co-targeting."}

    return {
        **{k: v for k, v in data.items() if k != "spots"},
        "escape_threshold": ESCAPE_THRESHOLD,
        "escape_model": {"weights": ESCAPE_WEIGHTS, "bias": ESCAPE_BIAS, "form": "sigmoid(bias + sum(w_g * log2UMI_g))"},
        "n_spots": len(spots),
        "n_escape": int((scores >= ESCAPE_THRESHOLD).sum()),
        "n_hotspots": len(hot),
        "recommendation": recommendation,
        "spots": spots,
    }
