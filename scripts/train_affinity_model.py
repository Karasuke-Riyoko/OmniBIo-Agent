"""
scripts/train_affinity_model.py
Trains the EGFR potency model and writes the cross-validated metrics the README quotes.

Data: data/egfr_chembl_pic50.csv — 5,568 EGFR (CHEMBL203) IC50 measurements from ChEMBL, as exported by
TeachOpenCADD talktorial T001 (https://github.com/volkamerlab/teachopencadd, CC BY 4.0).

Run:  python scripts/train_affinity_model.py
"""

import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from services.affinity_model import MODEL_PATH, fit_ridge  # noqa: E402
from services.featurizers import smiles_features  # noqa: E402

DATA = os.path.join(ROOT, "data", "egfr_chembl_pic50.csv")
BENCH = os.path.join(ROOT, "data", "benchmarks.json")
DIM, MAX_N, ALPHA, FOLDS, SEED = 2048, 4, 1.0, 5, 7


def load():
    with open(DATA, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["smiles"] and r["pIC50"]]
    x = np.stack([smiles_features(r["smiles"], DIM, MAX_N) for r in rows])
    y = np.array([float(r["pIC50"]) for r in rows])
    return x, y


def predict(model, x):
    return x @ model["w"] + model["b"]


def main():
    x, y = load()
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(len(y)), FOLDS)

    preds = np.empty_like(y)
    for k in range(FOLDS):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(FOLDS) if j != k])
        model = fit_ridge(x[train], y[train], ALPHA)
        preds[test] = predict(model, x[test])

    r, _ = pearsonr(y, preds)
    rho, _ = spearmanr(y, preds)
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
    mae = float(np.mean(np.abs(y - preds)))
    baseline_rmse = float(np.sqrt(np.mean((y - y.mean()) ** 2)))

    final = fit_ridge(x, y, ALPHA)
    np.savez_compressed(MODEL_PATH, w=final["w"], b=final["b"], dim=DIM, max_n=MAX_N,
                        y_min=float(y.min()), y_max=float(y.max()))

    bench = {}
    if os.path.exists(BENCH):
        with open(BENCH, encoding="utf-8") as f:
            bench = json.load(f)
    bench["egfr_affinity"] = {
        "dataset": "ChEMBL EGFR (CHEMBL203) IC50, TeachOpenCADD T001 export",
        "n_compounds": int(len(y)),
        "protocol": f"{FOLDS}-fold random cross-validation (seed {SEED})",
        "model": f"Ridge (alpha={ALPHA}) on {DIM}-bucket hashed SMILES 1-{MAX_N}-grams",
        "pearson_r": round(float(r), 3),
        "spearman_rho": round(float(rho), 3),
        "rmse_pic50": round(rmse, 3),
        "mae_pic50": round(mae, 3),
        "predict_mean_baseline_rmse": round(baseline_rmse, 3),
        "generated": time.strftime("%Y-%m-%d"),
    }
    with open(BENCH, "w", encoding="utf-8") as f:
        json.dump(bench, f, indent=2)
    print(json.dumps(bench["egfr_affinity"], indent=2))


if __name__ == "__main__":
    main()
