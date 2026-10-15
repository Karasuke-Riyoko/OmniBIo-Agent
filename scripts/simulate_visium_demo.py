"""
scripts/simulate_visium_demo.py
Builds data/visium_demo_spots.json: SIMULATED Visium-style spots laid over the bundled H&E tile.

Geometry follows 10x Visium (hexagonal lattice, 55 um spots, 100 um centre-to-centre), scaled to the tile.
Expression (log2 UMI) is drawn from rules, seeded for reproducibility:
- EGFR tracks tumour nuclear density measured from the H&E patch under each spot.
- MET and AXL are raised in a Gaussian niche centred on the densest tumour region, so the demo contains one
  bypass-resistance niche to find.
- CD274 (PD-L1) rises with stroma at the invasive front.
This is demo data for exercising the spatial pipeline, not a measurement.

Run:  python scripts/simulate_visium_demo.py
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from services import histology  # noqa: E402

OUT = os.path.join(ROOT, "data", "visium_demo_spots.json")
SEED = 11
SPACING_PX = 96


def main():
    rng = np.random.default_rng(SEED)
    h = histology.extract(histology.SAMPLE_IMAGE)
    tokens, (rows, cols), patch = h["tokens"], h["grid"], h["patch_px"]
    names = h["feature_names"]
    nuc = tokens[:, names.index("nuclear_frac")].reshape(rows, cols)
    stroma = tokens[:, names.index("stroma_frac")].reshape(rows, cols)
    width, height = h["image_px"]

    # Niche centre: the 3x3-patch window with the highest nuclear density.
    pad = np.pad(nuc, 1, mode="edge")
    smooth = sum(pad[i:i + rows, j:j + cols] for i in range(3) for j in range(3)) / 9
    cy, cx = np.unravel_index(np.argmax(smooth), smooth.shape)
    niche = np.array([(cx + 0.5) * patch, (cy + 0.5) * patch])

    spots = []
    dy = SPACING_PX * np.sqrt(3) / 2
    r = 0
    y = SPACING_PX / 2
    while y < height - SPACING_PX / 3:
        x = SPACING_PX / 2 + (SPACING_PX / 2 if r % 2 else 0)
        c = 0
        while x < width - SPACING_PX / 3:
            pr, pc = min(int(y // patch), rows - 1), min(int(x // patch), cols - 1)
            n_local = float(nuc[max(pr - 1, 0):pr + 2, max(pc - 1, 0):pc + 2].mean())
            s_local = float(stroma[max(pr - 1, 0):pr + 2, max(pc - 1, 0):pc + 2].mean())
            g = float(np.exp(-np.sum((np.array([x, y]) - niche) ** 2) / (2 * (1.3 * SPACING_PX) ** 2)))
            expr = {
                "EGFR": 2.0 + 14.0 * n_local + rng.normal(0, 0.35),
                "MET": 1.6 + 7.8 * g + 2.0 * n_local + rng.normal(0, 0.3),
                "AXL": 1.5 + 4.8 * g + 3.0 * s_local + rng.normal(0, 0.3),
                "CD274": 2.0 + 3.5 * s_local + 1.5 * g + rng.normal(0, 0.3),
            }
            spots.append({
                "barcode": f"SIM{r:02d}{c:02d}-1",
                "array_row": r, "array_col": c,
                "pxl_x": round(float(x), 1), "pxl_y": round(float(y), 1),
                "tumor_cellularity": round(min(1.0, n_local * 2.5), 2),
                "expression_log2_umi": {k: round(max(0.0, v), 2) for k, v in expr.items()},
            })
            x += SPACING_PX
            c += 1
        y += dy
        r += 1

    out = {
        "slide_id": "SIMULATED-DEMO-ON-SAMPLE-HE",
        "simulated": True,
        "platform": "Simulated 10x Visium geometry (55 um spots, 100 um centre-to-centre)",
        "normalization": "log2(UMI_count + 1)",
        "image_px": [width, height],
        "center_to_center_px": SPACING_PX,
        "generator": "scripts/simulate_visium_demo.py (seed %d)" % SEED,
        "spots": spots,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {len(spots)} spots; niche centre px {niche.tolist()}")


if __name__ == "__main__":
    main()
