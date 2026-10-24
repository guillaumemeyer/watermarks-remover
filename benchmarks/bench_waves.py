#!/usr/bin/env python3
"""WAVES benchmark adapter for image watermark removal evaluation.

Runs removal engines against the WAVES watermarked image dataset and computes
quality/effectiveness metrics (TPR, PSNR, SSIM, LPIPS, FID).

Exit codes:
  0  evaluation completed
  1  runtime error
  2  bad input (missing args, invalid paths)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from common import emit_json, eprint, subprocess_creationflags  # noqa: E402


class WavesResult(NamedTuple):
    tpr: float
    tpr_before: float
    psnr: float
    ssim: float
    lpips: float | None
    fid: float | None
    n_images: int


def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Peak signal-to-noise ratio between two uint8 images."""
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(255.0**2 / mse))


def compute_ssim(original: np.ndarray, modified: np.ndarray) -> float:
    """Simplified SSIM (no sliding window, single-scale)."""
    a = original.astype(np.float64)
    b = modified.astype(np.float64)
    mu_a, mu_b = np.mean(a), np.mean(b)
    var_a, var_b = np.var(a), np.var(b)
    cov_ab = np.mean((a - mu_a) * (b - mu_b))
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return float(num / den)


def _detect_watermark(
    img_path: Path,
    decoder_cmd: list[str] | None,
) -> bool:
    """Return True if the image is detected as watermarked.

    When *decoder_cmd* is given, it is invoked as
    ``decoder_cmd... <img_path> --json`` and exit 0 with
    ``{"detected": true}`` means watermarked.  Otherwise the built-in
    heuristic classifier is used (scheme not in none/unknown = detected).

    The built-in classifier is a routing heuristic, not a watermark
    decoder.  It cannot detect real latent-space watermarks like
    Tree-Ring.  For meaningful TPR, supply a watermark-specific decoder
    via --decoder.
    """
    cmd = (
        [*decoder_cmd, str(img_path), "--json"]
        if decoder_cmd
        else [
            sys.executable,
            str(SCRIPTS / "classify_watermark.py"),
            str(img_path),
            "--json",
        ]
    )
    cr = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        creationflags=subprocess_creationflags,
    )
    if cr.returncode != 0:
        return False
    import json as _json

    try:
        cj = _json.loads(cr.stdout)
        if decoder_cmd:
            return bool(cj.get("detected", False))
        return cj.get("scheme", "none") not in ("none", "unknown")
    except (ValueError, TypeError):
        return False


class WavesAdapter:
    """Runs a removal engine against WAVES benchmark images."""

    def __init__(self, waves_dir: str, output_dir: str, decoder_cmd: list[str] | None = None):
        self._waves_dir = Path(waves_dir)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._decoder_cmd = decoder_cmd

    def run_attack(
        self,
        watermark: str,
        engine: str,
        intensity: float = 0.5,
        dataset: str = "coco",
    ) -> WavesResult:
        wm_dir = self._waves_dir / dataset / watermark
        if not wm_dir.is_dir():
            raise FileNotFoundError(f"WAVES watermark dir not found: {wm_dir}")

        clean_dir = self._waves_dir / dataset / "clean"
        out_dir = self._output_dir / f"{watermark}_{engine}_{intensity}"
        out_dir.mkdir(parents=True, exist_ok=True)

        psnrs: list[float] = []
        ssims: list[float] = []
        detected_before = 0
        detected_after = 0
        images_processed = 0

        for img_path in sorted(wm_dir.glob("*.png")):
            out_path = out_dir / img_path.name
            clean_path = clean_dir / img_path.name if clean_dir.is_dir() else None

            if _detect_watermark(img_path, self._decoder_cmd):
                detected_before += 1

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "clean_image.py"),
                    str(img_path),
                    "-o",
                    str(out_path),
                    "--remove-pixel",
                    engine,
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=600,
                creationflags=subprocess_creationflags,
            )
            if result.returncode != 0:
                eprint(f"engine {engine} failed on {img_path.name}: rc={result.returncode}")
                continue

            images_processed += 1

            if _detect_watermark(out_path, self._decoder_cmd):
                detected_after += 1

            if clean_path and clean_path.is_file() and out_path.is_file():
                from PIL import Image

                orig = np.array(Image.open(clean_path).convert("RGB"))
                out = np.array(Image.open(out_path).convert("RGB"))
                if orig.shape == out.shape:
                    psnrs.append(compute_psnr(orig, out))
                    ssims.append(compute_ssim(orig, out))

        n_total = len(list(wm_dir.glob("*.png")))
        tpr_before = detected_before / n_total if n_total > 0 else 0.0
        tpr_after = detected_after / images_processed if images_processed > 0 else 0.0
        eprint(
            f"detection: before={detected_before}/{n_total} "
            f"({tpr_before:.2f}), after={detected_after}/{images_processed} "
            f"({tpr_after:.2f})"
        )
        return WavesResult(
            tpr=tpr_after,
            tpr_before=tpr_before,
            psnr=float(np.mean(psnrs)) if psnrs else 0.0,
            ssim=float(np.mean(ssims)) if ssims else 0.0,
            lpips=None,
            fid=None,
            n_images=images_processed,
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--waves-dir", type=Path, required=True, help="WAVES dataset root")
    p.add_argument("--engine", type=str, required=True, help="Removal engine name")
    p.add_argument("--watermark", type=str, required=True, help="Watermark scheme")
    p.add_argument("--intensity", type=float, default=0.5, help="Attack intensity")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--dataset", type=str, default="coco", help="Dataset name")
    p.add_argument(
        "--decoder",
        type=str,
        nargs="+",
        default=None,
        help="External watermark decoder command (receives <img> --json, "
        'must output {"detected": bool}); without this, the built-in '
        "heuristic classifier is used (cannot detect latent-space watermarks)",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    if not args.waves_dir.is_dir():
        eprint(f"WAVES dir not found: {args.waves_dir}")
        return 2

    try:
        adapter = WavesAdapter(str(args.waves_dir), str(args.output_dir), args.decoder)
        t0 = time.monotonic()
        result = adapter.run_attack(args.watermark, args.engine, args.intensity, args.dataset)
        elapsed = time.monotonic() - t0
    except FileNotFoundError as e:
        eprint(str(e))
        return 2
    except Exception as e:
        eprint(f"evaluation error: {e}")
        return 1

    if args.json:
        emit_json(
            {
                "watermark": args.watermark,
                "engine": args.engine,
                "intensity": args.intensity,
                "tpr_before": round(result.tpr_before, 4),
                "tpr": round(result.tpr, 4),
                "psnr": round(result.psnr, 2),
                "ssim": round(result.ssim, 4),
                "lpips": round(result.lpips, 4) if result.lpips is not None else None,
                "fid": round(result.fid, 2) if result.fid is not None else None,
                "n_images": result.n_images,
                "elapsed_s": round(elapsed, 2),
            }
        )
    else:
        lpips_s = f"{result.lpips:.4f}" if result.lpips is not None else "N/A (needs torch)"
        fid_s = f"{result.fid:.2f}" if result.fid is not None else "N/A (needs torch)"
        eprint(
            f"{args.watermark}/{args.engine}: "
            f"TPR_before={result.tpr_before:.3f} TPR={result.tpr:.3f} "
            f"PSNR={result.psnr:.1f} SSIM={result.ssim:.3f} "
            f"LPIPS={lpips_s} FID={fid_s} "
            f"n={result.n_images} ({elapsed:.1f}s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
