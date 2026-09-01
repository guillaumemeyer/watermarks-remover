#!/usr/bin/env python3
"""Optional CtrlRegen pixel-watermark remover backed by an external noai-watermark checkout.

This script does NOT vendor upstream code. It imports ``CtrlRegenEngine`` from
a user-provided checkout of ``mertizci/noai-watermark`` at runtime, using that
environment's optional dependencies (torch, diffusers, transformers, etc.).

Exit codes:
  0  removed successfully
  1  remover runtime error
  2  bad input (missing/unreadable image, bad args)
  3  remover unavailable (not configured / missing checkout / missing deps)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import cleaned_path, safe_write_bytes  # noqa: E402

# Backend default guidance scale (CtrlRegenEngine.run() default). Kept
# internal: intensity is the user-facing knob, not the CFG scale.
DEFAULT_GUIDANCE_SCALE = 2.0


def resolve_upstream(raw: str | None) -> Path | None:
    if not raw:
        return None
    upstream = Path(raw).expanduser().resolve()
    if not upstream.is_dir():
        return None
    return upstream


def resolve_device(raw: str | None) -> str:
    """Resolve the ``auto`` device hint to a concrete torch device."""
    if raw and raw != "auto":
        return raw
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # noqa: S110 - optional torch device detection
        pass
    return "cpu"


def save_image_bytes(image, output: Path) -> bytes:
    """Encode the regenerated image, honoring a JPEG output suffix."""
    ext = output.suffix.lower()
    buf = io.BytesIO()
    if ext in (".jpg", ".jpeg", ".jpe", ".jfif"):
        image = image.convert("RGB")
        image.save(buf, format="JPEG", quality=95)
    else:
        image.save(buf, format="PNG")
    return buf.getvalue()


def _progress(message: str) -> None:
    print(f"[ctrlregen] {message}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Input image (PNG/JPEG/etc.)")
    p.add_argument("-o", "--output", type=Path, help="Output path (default: *.ctrlregen.*)")
    p.add_argument(
        "--upstream-dir",
        type=Path,
        default=None,
        help="noai-watermark checkout root (default: $NOAI_WATERMARK_DIR)",
    )
    p.add_argument(
        "--intensity",
        type=float,
        default=0.25,
        help="Regeneration intensity in (0, 1]; lower preserves more detail (default: 0.25)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Diffusion inference steps (default: 50; effective steps ~= steps * intensity)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto|cpu|cuda|mps (default: auto)",
    )
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    if not args.path.is_file():
        print(f"not a file: {args.path}", file=sys.stderr)
        return 2
    if not 0 < args.intensity <= 1:
        print(f"intensity must be in (0, 1]: {args.intensity}", file=sys.stderr)
        return 2
    if args.steps < 1:
        print(f"steps must be >= 1: {args.steps}", file=sys.stderr)
        return 2

    raw_upstream = args.upstream_dir or os.environ.get("NOAI_WATERMARK_DIR")
    upstream = resolve_upstream(str(raw_upstream) if raw_upstream else None)
    if upstream is None:
        print(
            "CtrlRegen not configured: set NOAI_WATERMARK_DIR or pass --upstream-dir",
            file=sys.stderr,
        )
        return 3

    src_dir = upstream / "src"
    if not src_dir.is_dir():
        print(f"upstream src dir not found: {src_dir}", file=sys.stderr)
        return 3

    sys.path.insert(0, str(src_dir))
    try:
        from ctrlregen.engine import CtrlRegenEngine, is_ctrlregen_available
        from PIL import Image
    except ImportError as e:
        print(f"CtrlRegen dependencies missing: {e}", file=sys.stderr)
        print("run setup_ctrlregen.sh first", file=sys.stderr)
        return 3

    if not is_ctrlregen_available():
        print(
            "CtrlRegen dependencies not installed; run setup_ctrlregen.sh first",
            file=sys.stderr,
        )
        return 3

    try:
        image = Image.open(args.path).convert("RGB")
        image.load()
    except Exception as e:
        print(f"could not load image: {e}", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    output = args.output or cleaned_path(args.path, ".ctrlregen")

    engine = CtrlRegenEngine(
        base_model_id=None,
        device=device,
        torch_dtype=None,
        hf_token=os.environ.get("HF_TOKEN"),
        progress_callback=_progress,
    )

    try:
        result = engine.run(
            image,
            strength=args.intensity,
            num_inference_steps=args.steps,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
            seed=args.seed,
        )
    except Exception as e:
        print(f"CtrlRegen error: {e}", file=sys.stderr)
        return 1

    try:
        data = save_image_bytes(result, output)
        safe_write_bytes(output, data)
    except (OSError, ValueError) as e:
        print(f"cannot write output: {e}", file=sys.stderr)
        return 1

    payload = {
        "available": True,
        "upstream_dir": str(upstream),
        "output": str(output),
        "intensity": args.intensity,
        "steps": args.steps,
        "device": device,
        "seed": args.seed,
        "input_size": list(image.size),
        "output_size": list(result.size),
        "bytes_out": len(data),
    }

    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(
            f"CtrlRegen removed: {args.path} -> {output} "
            f"({payload['input_size']} -> {payload['output_size']}, "
            f"intensity {args.intensity}, device {device})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
