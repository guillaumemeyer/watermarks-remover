#!/usr/bin/env python3
"""FMDiffWA frequency-domain watermark removal.

Tiles the image into overlapping patches, suppresses spectral bands and peaks
that stand out of the natural-image power-law trend, optionally refines each
patch with low-strength Stable Diffusion img2img, and merges the patches with
cosine feathering (see fmdiffwa_engine.py). Needs torch; refinement also needs
diffusers + transformers and SD 2.1 base weights, otherwise only the frequency
stage runs (reported as ``refined: false``).

Exit codes:
  0  removed successfully
  1  remover runtime error
  2  bad input (missing/unreadable image, bad args)
  3  remover unavailable (missing deps)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from clean_ctrlregen import save_image_bytes  # noqa: E402
from common import cleaned_path, emit_json, eprint, safe_write_bytes  # noqa: E402
from fmdiffwa_engine import MIN_FILTER_SIZE, FMDiffWAEngine  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Input image (PNG/JPEG/etc.)")
    p.add_argument("-o", "--output", type=Path, help="Output path (default: *.fmdiffwa.*)")
    p.add_argument(
        "--patch-size", type=int, default=256, help="Patch tile size in pixels (default: 256)"
    )
    p.add_argument(
        "--intensity",
        type=float,
        default=0.3,
        help="Diffusion refinement strength in [0, 1]; 0 = frequency stage only (default: 0.3)",
    )
    p.add_argument("--steps", type=int, default=50, help="Diffusion inference steps (default: 50)")
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps (default: auto)")
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    if not args.path.is_file():
        eprint(f"not a file: {args.path}")
        return 2
    if args.patch_size < MIN_FILTER_SIZE:
        eprint(f"patch-size must be >= {MIN_FILTER_SIZE}: {args.patch_size}")
        return 2
    if not 0 <= args.intensity <= 1:
        eprint(f"intensity must be in [0, 1]: {args.intensity}")
        return 2
    if args.steps < 1:
        eprint(f"steps must be >= 1: {args.steps}")
        return 2

    if not FMDiffWAEngine.available():
        eprint("FMDiffWA dependencies missing: install torch (diffusers optional)")
        return 3

    from PIL import Image

    try:
        image = Image.open(args.path).convert("RGB")
        image.load()
    except Exception as e:
        eprint(f"could not load image: {e}")
        return 2

    output = args.output or cleaned_path(args.path, ".fmdiffwa")
    engine = FMDiffWAEngine(patch_size=args.patch_size, device=args.device)
    try:
        result = engine.remove(image, intensity=args.intensity, steps=args.steps)
    except Exception as e:
        eprint(f"FMDiffWA error: {e}")
        return 1

    try:
        data = save_image_bytes(result, output)
        safe_write_bytes(output, data)
    except (OSError, ValueError) as e:
        eprint(f"cannot write output: {e}")
        return 1

    payload = {
        "available": True,
        "engine": "fmdiffwa",
        "model_id": engine.model_id,
        "refined": engine.refined,
        "output": str(output),
        "patch_size": args.patch_size,
        "intensity": args.intensity,
        "steps": args.steps,
        "device": engine.device,
        "input_size": list(image.size),
        "output_size": list(result.size),
        "bytes_out": len(data),
    }
    if args.json:
        emit_json(payload)
    else:
        stage = "frequency + diffusion" if engine.refined else "frequency only"
        print(
            f"FMDiffWA removed: {args.path} -> {output} "
            f"({payload['input_size']} -> {payload['output_size']}, {stage}, "
            f"device {engine.device})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
