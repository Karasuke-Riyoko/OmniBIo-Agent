"""
services/histology.py
Turns a real H&E image into patch tokens for the Spatial Perceiver Resampler.

1. Colour deconvolution into haematoxylin (nuclei) and eosin (stroma/cytoplasm) optical density
   using the Ruifrok & Johnston stain vectors.
2. The image is tiled into PATCH x PATCH pixel patches; each patch becomes a 16-dim descriptor
   (stain densities, nuclear/stroma/lumen fractions, texture, colour).
3. Slide-level tissue composition (stroma %, nuclear %, lumen %) is summarised from the same masks
   and feeds the stromal penetration penalty in the orchestrator.

The bundled sample is a public H&E tile (AMIDA13 breast-cancer image from github.com/schaugf/HEnorm_python, MIT);
it is NOT this patient's lung biopsy. Upload your own tile from the UI or POST /api/histology.
"""

import io
import os
from typing import Any, Dict

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_IMAGE = os.path.join(BASE_DIR, "data", "histology", "sample_he_tile.jpg")
UPLOADED_IMAGE = os.path.join(BASE_DIR, "data", "histology", "uploaded_he_tile.png")

PATCH = 32
# Haematoxylin optical density above which a pixel counts as nuclear chromatin.
NUCLEAR_OD = 0.35
MAX_SIDE = 1024

# Ruifrok & Johnston (2001) H&E(+DAB) stain matrix; rows are unit OD vectors.
_STAIN_RGB = np.array([
    [0.650, 0.704, 0.286],   # haematoxylin
    [0.072, 0.990, 0.105],   # eosin
    [0.268, 0.570, 0.776],   # residual / DAB
])
_STAIN_RGB = _STAIN_RGB / np.linalg.norm(_STAIN_RGB, axis=1, keepdims=True)
_UNMIX = np.linalg.inv(_STAIN_RGB)

FEATURE_NAMES = [
    "h_od_mean", "h_od_std", "e_od_mean", "e_od_std", "nuclear_frac", "stroma_frac", "lumen_frac",
    "h_to_e_ratio", "grad_energy", "h_grad_energy", "r_mean", "g_mean", "b_mean", "sat_mean",
    "h_od_p90", "e_od_p90",
]


def current_image_path() -> str:
    return UPLOADED_IMAGE if os.path.exists(UPLOADED_IMAGE) else SAMPLE_IMAGE


def save_upload(raw: bytes) -> str:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.save(UPLOADED_IMAGE)
    return UPLOADED_IMAGE


def reset_upload() -> None:
    if os.path.exists(UPLOADED_IMAGE):
        os.remove(UPLOADED_IMAGE)


def _load_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    scale = MAX_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.BILINEAR)
    w, h = (img.size[0] // PATCH) * PATCH, (img.size[1] // PATCH) * PATCH
    return np.asarray(img.crop((0, 0, w, h)), dtype=np.float64) / 255.0


def stain_densities(rgb: np.ndarray) -> np.ndarray:
    od = -np.log(np.clip(rgb, 1.0 / 255.0, 1.0))
    return np.clip(od.reshape(-1, 3) @ _UNMIX, 0, None).reshape(rgb.shape)


def extract(path: str = None) -> Dict[str, Any]:
    """Returns patch tokens (N x 16), the patch grid shape and slide-level composition."""
    path = path or current_image_path()
    rgb = _load_rgb(path)
    stains = stain_densities(rgb)
    h_od, e_od = stains[..., 0], stains[..., 1]

    brightness = rgb.mean(axis=-1)
    sat = rgb.max(axis=-1) - rgb.min(axis=-1)
    lumen = (brightness > 0.85) & (sat < 0.12)
    nuclear = (h_od > NUCLEAR_OD) & ~lumen
    stroma = (e_od > h_od * 1.2) & ~nuclear & ~lumen

    gy, gx = np.gradient(brightness)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    hgy, hgx = np.gradient(h_od)
    hgrad = np.sqrt(hgx ** 2 + hgy ** 2)

    rows, cols = rgb.shape[0] // PATCH, rgb.shape[1] // PATCH

    def patches(a):
        return a.reshape(rows, PATCH, cols, PATCH).swapaxes(1, 2).reshape(rows * cols, PATCH * PATCH)

    ph, pe = patches(h_od), patches(e_od)
    tokens = np.stack([
        ph.mean(1), ph.std(1), pe.mean(1), pe.std(1),
        patches(nuclear.astype(float)).mean(1), patches(stroma.astype(float)).mean(1),
        patches(lumen.astype(float)).mean(1),
        ph.mean(1) / (pe.mean(1) + 1e-3),
        patches(grad).mean(1), patches(hgrad).mean(1),
        patches(rgb[..., 0]).mean(1), patches(rgb[..., 1]).mean(1), patches(rgb[..., 2]).mean(1),
        patches(sat).mean(1),
        np.percentile(ph, 90, axis=1), np.percentile(pe, 90, axis=1),
    ], axis=1)

    tissue = (~lumen).sum()
    composition = {
        "stroma_pct_of_tissue": round(float(100.0 * stroma.sum() / max(tissue, 1)), 1),
        "nuclear_pct_of_tissue": round(float(100.0 * nuclear.sum() / max(tissue, 1)), 1),
        "lumen_pct_of_image": round(float(100.0 * lumen.mean()), 1),
    }
    nuclear_density_grid = tokens[:, FEATURE_NAMES.index("nuclear_frac")].reshape(rows, cols)

    return {
        "source": "uploaded" if path == UPLOADED_IMAGE else "bundled_sample",
        "source_note": ("Your uploaded H&E tile" if path == UPLOADED_IMAGE else
                        "Public H&E sample (AMIDA13 breast tissue via HEnorm_python, MIT), not this patient's biopsy"),
        "image_px": [int(rgb.shape[1]), int(rgb.shape[0])],
        "patch_px": PATCH,
        "grid": [rows, cols],
        "n_patches": int(rows * cols),
        "feature_names": FEATURE_NAMES,
        "tokens": tokens,
        "composition": composition,
        "nuclear_density_grid": nuclear_density_grid,
    }


def summary(path: str = None) -> Dict[str, Any]:
    """JSON-safe view of extract() for the UI."""
    out = extract(path)
    grid = out.pop("nuclear_density_grid")
    out.pop("tokens")
    out["nuclear_density_grid"] = np.round(grid, 3).tolist()
    return out
