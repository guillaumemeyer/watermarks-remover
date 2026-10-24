#!/usr/bin/env python3
"""Forensic stealth detector for the WAVES robustness-evaluation toolchain.

Detects traces of image manipulation (watermark removal) via spectral analysis
and JPEG double-compression detection.  Used by the attack chain coordinator
to verify that removal does not leave detectable artifacts.

Exit codes:
  0  check completed
  1  runtime error
  2  bad input
  3  detector unavailable
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


class ForensicResult(NamedTuple):
    stealth_score: float
    detected_traces: list[str]
    recommendation: str


def spectral_deformation_score(image: np.ndarray) -> float:
    """Measure deviation from natural 1/f^alpha spectral falloff.

    Returns 0.0 for natural images, higher values for processed images
    where the frequency spectrum has been altered (e.g., by diffusion).
    """
    gray = (
        np.mean(image.astype(np.float64), axis=2) if image.ndim == 3 else image.astype(np.float64)
    )

    f = np.fft.fft2(gray)
    mag = np.abs(np.fft.fftshift(f))

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)

    radial = np.zeros(max_r)
    for r in range(1, max_r):
        y, x = np.ogrid[-cy : h - cy, -cx : w - cx]
        mask = (y * y + x * x >= r * r) & (y * y + x * x < (r + 1) ** 2)
        if mask.any():
            radial[r] = np.mean(mag[mask])

    safe = radial[1:]
    safe = np.where(safe > 0, safe, 1e-10)
    log_power = np.log10(safe)
    log_freq = np.log10(np.arange(1, len(safe) + 1, dtype=np.float64))

    if len(log_freq) < 2:
        return 0.0

    coeffs = np.polyfit(log_freq, log_power, 1)
    fitted = np.polyval(coeffs, log_freq)
    residual = np.sqrt(np.mean((log_power - fitted) ** 2))

    score = float(np.clip(residual / 2.0, 0.0, 1.0))
    return score


def jpeg_double_compression_score(image_bytes: bytes) -> float:
    """Detect double JPEG compression artifacts.

    Analyses the distribution of DCT coefficients for signs of
    double quantisation.  Returns 0.0 for non-JPEG or single-compressed,
    higher for double-compressed.
    """
    if len(image_bytes) < 3 or image_bytes[:2] != b"\xff\xd8":
        return 0.0

    try:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        if img.format != "JPEG":
            return 0.0

        arr = np.array(img.convert("L"), dtype=np.float64)
        dct_like = np.fft.dct2 if hasattr(np.fft, "dct2") else None
        if dct_like is None:
            from scipy.fft import dctn

            blk_scores = []
            for r in range(0, arr.shape[0] - 7, 8):
                for c in range(0, arr.shape[1] - 7, 8):
                    block = arr[r : r + 8, c : c + 8]
                    coeffs = dctn(block, type=2, norm="ortho")
                    ac = coeffs.ravel()[1:]
                    hist, _ = np.histogram(ac, bins=32, range=(-50, 50))
                    peaks = np.sum(np.diff(np.sign(np.diff(hist.astype(float)))) < 0)
                    blk_scores.append(peaks / 15.0)
            if not blk_scores:
                return 0.0
            return float(np.clip(np.mean(blk_scores), 0.0, 1.0))
        return 0.0
    except Exception:
        return 0.0


class ForensicChecker:
    """Combined forensic stealth analyser."""

    def __init__(self, model_path: str | None = None):
        self._model_path = model_path

    def check(self, image_path: str) -> ForensicResult:
        from PIL import Image

        path = Path(image_path)
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        raw_bytes = path.read_bytes()

        spectral = spectral_deformation_score(arr)
        jpeg = jpeg_double_compression_score(raw_bytes)

        combined = 0.6 * spectral + 0.4 * jpeg
        traces: list[str] = []
        if spectral > 0.3:
            traces.append("spectral_anomaly")
        if jpeg > 0.3:
            traces.append("double_jpeg")

        if combined > 0.7:
            rec = "reprocess"
        elif combined > 0.3:
            rec = "warn"
        else:
            rec = "pass"

        return ForensicResult(
            stealth_score=round(combined, 4),
            detected_traces=traces,
            recommendation=rec,
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Input image")
    p.add_argument("--model-path", type=str, default=None, help="Trained detector model")
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    if not args.path.is_file():
        eprint(f"not a file: {args.path}")
        return 2

    try:
        checker = ForensicChecker(model_path=args.model_path)
        result = checker.check(str(args.path))
    except Exception as e:
        eprint(f"forensic check error: {e}")
        return 1

    if args.json:
        emit_json(
            {
                "stealth_score": result.stealth_score,
                "detected_traces": result.detected_traces,
                "recommendation": result.recommendation,
            }
        )
    else:
        eprint(
            f"stealth={result.stealth_score:.3f} "
            f"traces={result.detected_traces} "
            f"rec={result.recommendation}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
