#!/usr/bin/env python3
"""NFPA next-frame prediction watermark removal.

Inverts the image to its initial diffusion noise, warps that noise with an
edge-driven flow field and regenerates from it (see nfpa_engine.py). Needs the
optional torch + diffusers + transformers stack and SD 2.1 base weights.

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
from nfpa_engine import NFPAEngine  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Input image (PNG/JPEG/etc.)")
    p.add_argument("-o", "--output", type=Path, help="Output path (default: *.nfpa.*)")
    p.add_argument(
        "--flow-strength",
        type=float,
        default=0.05,
        help="Flow field strength in (0, 1] (default: 0.05)",
    )
    p.add_argument("--steps", type=int, default=50, help="DDIM steps per pass (default: 50)")
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps (default: auto)")
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    if not args.path.is_file():
        eprint(f"not a file: {args.path}")
        return 2
    if not 0 < args.flow_strength <= 1:
        eprint(f"flow-strength must be in (0, 1]: {args.flow_strength}")
        return 2
    if args.steps < 1:
        eprint(f"steps must be >= 1: {args.steps}")
        return 2

    if not NFPAEngine.available():
        eprint("NFPA dependencies missing: install torch, diffusers and transformers")
        return 3

    from PIL import Image

    try:
        image = Image.open(args.path).convert("RGB")
        image.load()
    except Exception as e:
        eprint(f"could not load image: {e}")
        return 2

    output = args.output or cleaned_path(args.path, ".nfpa")
    engine = NFPAEngine(device=args.device)
    try:
        result = engine.remove(image, flow_strength=args.flow_strength, steps=args.steps)
    except Exception as e:
        eprint(f"NFPA error: {e}")
        return 1

    try:
        data = save_image_bytes(result, output)
        safe_write_bytes(output, data)
    except (OSError, ValueError) as e:
        eprint(f"cannot write output: {e}")
        return 1

    payload = {
        "available": True,
        "engine": "nfpa",
        "model_id": engine.model_id,
        "output": str(output),
        "flow_strength": args.flow_strength,
        "steps": args.steps,
        "device": engine.device,
        "input_size": list(image.size),
        "output_size": list(result.size),
        "bytes_out": len(data),
    }
    if args.json:
        emit_json(payload)
    else:
        print(
            f"NFPA removed: {args.path} -> {output} "
            f"({payload['input_size']} -> {payload['output_size']}, "
            f"flow-strength {args.flow_strength}, device {engine.device})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
