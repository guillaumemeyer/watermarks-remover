#!/usr/bin/env python3
"""MarkNull latent-space watermark removal via NLAS optimisation.

Reimplemented from arXiv 2608.10166.  Part of the WAVES robustness-evaluation
toolchain for measuring watermark fragility against latent-space attacks.

Exit codes:
  0  removed successfully
  1  remover runtime error
  2  bad input (missing/unreadable image, bad args)
  3  remover unavailable (missing torch / diffusers)
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import cleaned_path, emit_json, eprint, safe_write_bytes  # noqa: E402


def _save_image_bytes(image, output: Path) -> bytes:
    buf = io.BytesIO()
    ext = output.suffix.lower()
    if ext in (".jpg", ".jpeg", ".jpe", ".jfif"):
        image = image.convert("RGB")
        image.save(buf, format="JPEG", quality=95)
    else:
        image.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Input image (PNG/JPEG/etc.)")
    p.add_argument("-o", "--output", type=Path, help="Output path (default: *.marknull.*)")
    p.add_argument(
        "--mode",
        choices=("optimize", "fast"),
        default="optimize",
        help="optimize: full NLAS gradient descent; fast: single VAE round-trip (default: optimize)",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Optimisation iterations (default: 100)",
    )
    p.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="HuggingFace model ID for VAE (default: stabilityai/stable-diffusion-2-1-base)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto|cpu|cuda|mps (default: auto)",
    )
    p.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds (default: 3600)")
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    if not args.path.is_file():
        eprint(f"not a file: {args.path}")
        return 2
    if args.iterations < 1:
        eprint(f"iterations must be >= 1: {args.iterations}")
        return 2

    try:
        from marknull_engine import MarkNullEngine
    except ImportError as e:
        eprint(f"MarkNull dependencies missing: {e}")
        return 3

    if not MarkNullEngine.available():
        eprint("MarkNull requires torch and diffusers; install them first")
        return 3

    try:
        from PIL import Image

        image = Image.open(args.path).convert("RGB")
        image.load()
    except Exception as e:
        eprint(f"could not load image: {e}")
        return 2

    model_id = args.model_id or os.environ.get(
        "MARKNULL_MODEL_ID", "stabilityai/stable-diffusion-2-1-base"
    )
    output = args.output or cleaned_path(args.path, ".marknull")
    bytes_in = args.path.stat().st_size

    engine = MarkNullEngine(model_id=model_id, device=args.device)

    t0 = time.monotonic()
    try:
        result = engine.remove(image, mode=args.mode, iterations=args.iterations)
    except Exception as e:
        eprint(f"MarkNull error: {e}")
        return 1
    elapsed = time.monotonic() - t0

    data = _save_image_bytes(result, output)
    safe_write_bytes(output, data)

    if args.json:
        emit_json(
            {
                "output": str(output),
                "mode": args.mode,
                "iterations": args.iterations,
                "device": engine._device,
                "model_id": model_id,
                "bytes_in": bytes_in,
                "bytes_out": len(data),
                "elapsed_s": round(elapsed, 2),
            }
        )
    else:
        eprint(f"wrote {output} ({len(data)} bytes, {elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
