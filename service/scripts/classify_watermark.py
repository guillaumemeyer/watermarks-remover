#!/usr/bin/env python3
"""Watermark type classifier for the WAVES robustness-evaluation toolchain.

Identifies which watermarking scheme was used on an image by analysing spatial
and frequency-domain features.  The classifier routes images to the appropriate
removal engine in the attack chain coordinator.

Exit codes:
  0  classified successfully
  1  runtime error
  2  bad input (missing/unreadable image)
  3  classifier unavailable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import emit_json, eprint  # noqa: E402

VALID_SCHEMES = frozenset(
    {
        "tree_ring",
        "gaussian_shading",
        "stegastamp",
        "stable_signature",
        "synthid",
        "hidden",
        "trustmark",
        "unknown",
        "none",
    }
)

PIXEL_BASED_SCHEMES = frozenset({"stegastamp", "stable_signature", "hidden", "trustmark"})

LATENT_ONLY_ENGINES = frozenset({"marknull", "nfpa", "crack_bark"})


class ClassifyResult(NamedTuple):
    scheme: str
    confidence: float


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_spatial_features(image: np.ndarray) -> np.ndarray:
    """Spatial-domain features: edge density, local variance, histogram stats."""
    from scipy import ndimage

    gray = (
        np.mean(image.astype(np.float64), axis=2) if image.ndim == 3 else image.astype(np.float64)
    )

    sx = ndimage.sobel(gray, axis=0)
    sy = ndimage.sobel(gray, axis=1)
    edges = np.hypot(sx, sy)
    edge_density = float(np.mean(edges > np.percentile(edges, 90)))

    blk = 16
    h, w = gray.shape
    variances = []
    for r in range(0, h - blk + 1, blk):
        for c in range(0, w - blk + 1, blk):
            variances.append(np.var(gray[r : r + blk, c : c + blk]))
    var_arr = np.array(variances) if variances else np.array([0.0])
    var_mean = float(np.mean(var_arr))
    var_std = float(np.std(var_arr))

    hist, _ = np.histogram(gray.ravel(), bins=64, range=(0, 255), density=True)
    mu = float(np.mean(hist))
    sigma = float(np.std(hist)) + 1e-10
    skew = float(np.mean(((hist - mu) / sigma) ** 3))
    kurt = float(np.mean(((hist - mu) / sigma) ** 4)) - 3.0

    return np.array([edge_density, var_mean, var_std, skew, kurt], dtype=np.float64)


def extract_frequency_features(image: np.ndarray) -> np.ndarray:
    """Frequency-domain features: radial power, ring energy, phase coherence."""
    gray = (
        np.mean(image.astype(np.float64), axis=2) if image.ndim == 3 else image.astype(np.float64)
    )

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    phase = np.angle(fshift)

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)
    radial_profile = np.zeros(max_r)
    for r in range(max_r):
        y, x = np.ogrid[-cy : h - cy, -cx : w - cx]
        mask = (y * y + x * x >= r * r) & (y * y + x * x < (r + 1) ** 2)
        if mask.any():
            radial_profile[r] = np.mean(mag[mask])

    safe_profile = radial_profile[1:]
    safe_profile = np.where(safe_profile > 0, safe_profile, 1e-10)
    log_profile = np.log10(safe_profile)
    radii = np.log10(np.arange(1, len(safe_profile) + 1).astype(np.float64))
    slope = float(np.polyfit(radii, log_profile, 1)[0]) if len(radii) > 1 else -2.0

    ring_bands = [
        (max_r // 8, max_r // 6),
        (max_r // 4, max_r // 3),
        (max_r // 2 - 5, max_r // 2 + 5),
    ]
    ring_energies = []
    for rlo, rhi in ring_bands:
        lo = max(0, rlo)
        hi = min(max_r - 1, rhi)
        y, x = np.ogrid[-cy : h - cy, -cx : w - cx]
        mask = (y * y + x * x >= lo * lo) & (y * y + x * x < hi * hi)
        ring_energies.append(float(np.mean(mag[mask])) if mask.any() else 0.0)

    mid_band = mag[cy - max_r // 4 : cy + max_r // 4, cx - max_r // 4 : cx + max_r // 4]
    mid_phase = phase[cy - max_r // 4 : cy + max_r // 4, cx - max_r // 4 : cx + max_r // 4]
    phase_coherence = float(np.abs(np.mean(np.exp(1j * mid_phase)))) if mid_band.size > 0 else 0.0

    return np.array([slope, *ring_energies, phase_coherence, float(np.mean(mag))], dtype=np.float64)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class WatermarkClassifier:
    """Heuristic or CNN-based watermark scheme classifier."""

    def __init__(self, model_path: str | None = None):
        self._model_path = model_path
        self._model = None

    def classify(self, image_path: str) -> ClassifyResult:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)

        freq = extract_frequency_features(arr)
        spat = extract_spatial_features(arr)

        if self._model_path is not None:
            return self._classify_cnn(arr)

        return self._classify_heuristic(freq, spat)

    def _classify_heuristic(self, freq: np.ndarray, spat: np.ndarray) -> ClassifyResult:
        slope = freq[0]
        ring_energies = freq[1:4]
        phase_coherence = freq[4]
        mean_mag = freq[5]
        edge_density = spat[0]

        ring_norm = np.array(ring_energies) / (mean_mag + 1e-10)
        if np.max(ring_norm) > 0.8 and phase_coherence > 0.05:
            return ClassifyResult("tree_ring", min(float(np.max(ring_norm)), 1.0))

        if abs(slope + 2.0) > 1.0 and phase_coherence < 0.02:
            return ClassifyResult("gaussian_shading", 0.6)

        if edge_density > 0.15 and np.std(ring_norm) > 0.1:
            return ClassifyResult("stegastamp", min(float(edge_density * 3), 1.0))

        if phase_coherence > 0.1:
            return ClassifyResult("synthid", min(float(phase_coherence * 5), 1.0))

        return ClassifyResult("unknown", 0.3)

    def _classify_cnn(self, image: np.ndarray) -> ClassifyResult:
        try:
            import torch

            model = torch.load(self._model_path, weights_only=False)
            model.eval()
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                logits = model(tensor)
            probs = torch.softmax(logits, dim=-1)
            idx = int(probs.argmax())
            schemes = sorted(VALID_SCHEMES - {"unknown", "none"})
            scheme = schemes[idx] if idx < len(schemes) else "unknown"
            return ClassifyResult(scheme, float(probs[0, idx]))
        except Exception as e:
            eprint(f"CNN classification failed: {e}")
            return self._classify_heuristic(
                extract_frequency_features(image),
                extract_spatial_features(image),
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Input image")
    p.add_argument("--model-path", type=str, default=None, help="Trained classifier model")
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    if not args.path.is_file():
        eprint(f"not a file: {args.path}")
        return 2

    try:
        clf = WatermarkClassifier(model_path=args.model_path)
        result = clf.classify(str(args.path))
    except Exception as e:
        eprint(f"classification error: {e}")
        return 1

    if args.json:
        emit_json({"scheme": result.scheme, "confidence": round(result.confidence, 4)})
    else:
        eprint(f"{result.scheme} (confidence: {result.confidence:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
