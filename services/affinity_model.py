"""
services/affinity_model.py
EGFR potency model: ridge regression on hashed SMILES n-grams, trained on 5,568 measured
EGFR (CHEMBL203) IC50 values from ChEMBL (exported by the TeachOpenCADD T001 talktorial).

Train / re-evaluate with:  python scripts/train_affinity_model.py
The fitted weights live in models/egfr_affinity_ridge.npz; the cross-validated metrics in data/benchmarks.json.
"""

import os
from typing import Dict, Any

import numpy as np

from services.featurizers import smiles_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "egfr_affinity_ridge.npz")

# RT ln(10) at 298.15 K in kcal/mol: dG = -RT ln(10) * pK
RT_LN10_KCAL = 1.364


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> Dict[str, np.ndarray]:
    """Closed-form ridge regression with an unpenalised intercept."""
    x_mean, y_mean = x.mean(axis=0), y.mean()
    xc, yc = x - x_mean, y - y_mean
    # Dual form is cheaper when samples < features.
    if xc.shape[0] < xc.shape[1]:
        k = xc @ xc.T
        a = np.linalg.solve(k + alpha * np.eye(k.shape[0]), yc)
        w = xc.T @ a
    else:
        w = np.linalg.solve(xc.T @ xc + alpha * np.eye(xc.shape[1]), xc.T @ yc)
    return {"w": w, "b": np.array(y_mean - x_mean @ w)}


class EGFRAffinityModel:
    _instance = None

    def __init__(self, path: str = MODEL_PATH):
        data = np.load(path)
        self.w = data["w"]
        self.b = float(data["b"])
        self.dim = int(data["dim"])
        self.max_n = int(data["max_n"])
        self.train_range = (float(data["y_min"]), float(data["y_max"]))

    @classmethod
    def load(cls) -> "EGFRAffinityModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict(self, smiles: str) -> Dict[str, Any]:
        x = smiles_features(smiles, self.dim, self.max_n)
        pic50 = float(np.clip(x @ self.w + self.b, *self.train_range))
        return {
            "predicted_pic50": round(pic50, 2),
            "predicted_ic50_nm": round(10 ** (9 - pic50), 2),
            # pIC50 used as a stand-in for pKd, which is why this is labelled an estimate.
            "estimated_delta_g_kcal_mol": round(-RT_LN10_KCAL * pic50, 2),
            "model": "Ridge on SMILES n-grams · ChEMBL EGFR (CHEMBL203) IC50",
        }
