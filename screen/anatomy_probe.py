"""exploratory chest x-ray anatomy metrics.

no public asthma-labeled pediatric imaging corpus exists, so this module does
not make risk claims and does not run a trained classifier. it measures a few
basic signals from an x-ray image: lung field transparency, left/right
asymmetry, hilar texture energy, and lower lung dominance. the numbers are
conversation starters for a clinician, nothing more.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_xray(path: str | Path) -> np.ndarray:
    """grayscale image in [0, 1], autocontrast so any export format reads well."""
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img)
    return np.asarray(img, dtype=np.float32) / 255.0


def _grayscale_energy(patch: np.ndarray) -> float:
    gx = np.abs(np.diff(patch, axis=1)).mean()
    gy = np.abs(np.diff(patch, axis=0)).mean()
    return float((gx + gy) / 2)


def anatomy_metrics(img: np.ndarray) -> dict:
    """compute the four signals on the torso region, all floats in [0, 1]."""
    h, w = img.shape
    # crop to the central torso region to skip borders and text markers
    img = img[int(h * 0.08) : int(h * 0.95), int(w * 0.10) : int(w * 0.90)]
    h, w = img.shape
    mid_y0, mid_y1 = int(h * 0.25), int(h * 0.75)

    # lung fields, skipping the central mediastinal shadow (middle 12%)
    center_cut = int(w * 0.06)
    left_field = img[mid_y0:mid_y1, : w // 2 - center_cut]
    right_field = img[mid_y0:mid_y1, w // 2 + center_cut :]
    field_mean = (left_field.mean() + right_field.mean()) / 2

    transparency = float(np.clip(field_mean / max(img.mean(), 1e-6), 0.0, 2.0) / 2.0)
    asymmetry = float(
        np.clip(abs(left_field.mean() - right_field.mean()) / max(field_mean, 1e-6), 0.0, 1.0)
    )

    # hilar region: upper central quadrants on each side of the spine
    hilum = img[int(h * 0.30) : int(h * 0.55), :]
    side_w = w // 2 - center_cut
    hilum_texture = float(
        np.clip(
            max(
                _grayscale_energy(hilum[:, :side_w]),
                _grayscale_energy(hilum[:, w - side_w :]),
            )
            * 4.0,
            0.0,
            1.0,
        )
    )

    # lower vs upper lung brightness; hyperinflation tends to brighten lowers
    upper = img[int(h * 0.25) : int(h * 0.50)]
    lower = img[int(h * 0.50) : int(h * 0.85)]
    lower_dominance = float(np.clip((lower.mean() - upper.mean()) / max(upper.mean(), 1e-6) + 0.5, 0.0, 1.0))

    return {
        "transparency": round(transparency, 3),
        "asymmetry": round(asymmetry, 3),
        "hilar_texture": round(hilum_texture, 3),
        "lower_dominance": round(lower_dominance, 3),
    }


def observations(metrics: dict) -> list[str]:
    """plain sentences describing the metrics. deliberately cautious wording."""
    notes = []
    t = metrics["transparency"]
    if t < 0.4:
        notes.append("lung fields read darker than usual - possible reduced air content")
    elif t > 0.65:
        notes.append("lung fields read brighter than usual - possible hyperinflation")
    else:
        notes.append("lung field brightness sits in the typical range")

    a = metrics["asymmetry"]
    if a > 0.25:
        notes.append(f"left/right brightness differs ({a:.0%}) - worth a clinician's eye")
    else:
        notes.append("left and right lung fields look symmetric")

    ht = metrics["hilar_texture"]
    if ht > 0.6:
        notes.append("hilar texture is busy - possible prominent markings")
    else:
        notes.append("hilar texture looks unremarkable")

    ld = metrics["lower_dominance"]
    if ld > 0.65:
        notes.append("lower fields dominate the brightness profile")
    return notes
