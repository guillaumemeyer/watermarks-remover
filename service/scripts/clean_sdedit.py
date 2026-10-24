#!/usr/bin/env python3
"""Remove latent-space watermarks by SDEdit resampling (local diffusion img2img).

Encodes the image to latent space, renoises it to an intermediate timestep, and
denoises it back along a fresh noise path.  Watermarks embedded in the initial
diffusion noise (Tree-Ring, Gaussian Shading) do not survive the resampling,
and the output carries diffusion-native texture rather than additive noise.

Exit codes:
  0  attack completed
  1  runtime error
  2  bad input
  3  engine unavailable (torch/diffusers missing)
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import cleaned_path, emit_json, eprint, safe_write_bytes  # noqa: E402

# Matches the model the Tree-Ring tooling in this repo already pins
# (benchmarks/decode_treering.py, markdiffusion_harness.py).
DEFAULT_MODEL = os.environ.get("SDEDIT_MODEL_ID", "huanzi05/stable-diffusion-2-1-base")

# SD 2.1 base is trained at 512px; feeding it a much larger image duplicates
# subjects and can exhaust VRAM.  Resample at this size and restore after.
MAX_WORKING_PX = 768


_SUFFIX_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".bmp": "BMP",
    ".png": "PNG",
}


def _encoding_for(output: Path) -> str:
    """Pick the Pillow format matching `output`'s extension so the bytes match the name."""
    return _SUFFIX_FORMATS.get(output.suffix.lower(), "PNG")


def _save_options(fmt: str) -> dict:
    """Per-format save options; WEBP defaults to lossy, which would re-degrade output."""
    if fmt == "JPEG":
        return {"quality": 95}
    if fmt == "WEBP":
        return {"lossless": True}
    return {}


def resolve_device(raw: str | None) -> str:
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


def sdedit(
    image_path: Path,
    strength: float,
    steps: int,
    guidance: float,
    prompt: str,
    model_id: str,
    device: str,
    seed: int | None,
):
    """Resample `image_path` through diffusion at `strength`; returns a PIL image."""
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline
    from PIL import Image

    use_fp16 = device.split(":", 1)[0] == "cuda"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    src = Image.open(image_path).convert("RGB")
    work = src
    if max(src.size) > MAX_WORKING_PX:
        scale = MAX_WORKING_PX / max(src.size)
        work = src.resize(
            (max(8, int(src.width * scale)), max(8, int(src.height * scale))),
            Image.Resampling.LANCZOS,
        )

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    out = pipe(
        prompt=prompt,
        image=work,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]

    if out.size != src.size:
        out = out.resize(src.size, Image.Resampling.LANCZOS)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Input image")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output path")
    p.add_argument(
        "--strength",
        type=float,
        default=0.35,
        help="Denoising strength 0-1; higher destroys more watermark (default: 0.35)",
    )
    p.add_argument("--steps", type=int, default=50, help="Inference steps (default: 50)")
    p.add_argument("--guidance", type=float, default=3.0, help="CFG scale (default: 3.0)")
    p.add_argument(
        "--prompt",
        type=str,
        default="a high quality photograph",
        help="Guidance prompt (default: generic photo prompt)",
    )
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Diffusion model id")
    p.add_argument("--device", type=str, default=None, help="Device (auto|cpu|cuda|mps)")
    p.add_argument("--seed", type=int, default=None, help="Sampling seed for reproducible runs")
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    if not args.path.is_file():
        eprint(f"not a file: {args.path}")
        return 2
    if not 0.0 < args.strength <= 1.0:
        eprint(f"--strength must be in (0, 1]: {args.strength}")
        return 2
    if args.steps < 1:
        eprint(f"--steps must be >= 1: {args.steps}")
        return 2
    if not math.isfinite(args.guidance) or args.guidance < 0:
        eprint(f"--guidance must be a finite, non-negative number: {args.guidance}")
        return 2
    # Diffusers derives the denoising step count from strength * steps; a product
    # below 1 silently performs no denoising at all.
    if int(args.steps * args.strength) < 1:
        eprint(
            f"--strength {args.strength} x --steps {args.steps} yields no denoising steps; "
            "raise one of them"
        )
        return 2

    # Validate the input before the heavy imports, so a bad image reports exit 2
    # even where torch/diffusers are absent (CI installs neither).
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        eprint(f"sdedit unavailable: {exc}")
        return 3

    try:
        # verify() only checks headers; decoding is what catches truncated data,
        # so force a full decode here rather than failing later at inference.
        with Image.open(args.path) as probe:
            probe.convert("RGB").load()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        # Pillow raises SyntaxError for some corrupt chunk structures, which is
        # still bad input rather than a crash.
        eprint(f"not a readable image: {args.path} ({exc})")
        return 2

    try:
        import torch  # noqa: F401
        from diffusers import StableDiffusionImg2ImgPipeline  # noqa: F401
    except ImportError as exc:
        eprint(f"sdedit unavailable: {exc}")
        return 3

    device = resolve_device(args.device)
    output = args.output or cleaned_path(args.path, ".sdedit")

    try:
        out_img = sdedit(
            args.path,
            strength=args.strength,
            steps=args.steps,
            guidance=args.guidance,
            prompt=args.prompt,
            model_id=args.model,
            device=device,
            seed=args.seed,
        )
        buf = io.BytesIO()
        fmt = _encoding_for(output)
        out_img.save(buf, format=fmt, **_save_options(fmt))
        safe_write_bytes(output, buf.getvalue())
    except Exception as e:
        eprint(f"sdedit error: {e}")
        return 1

    if args.json:
        emit_json(
            {
                "output": str(output),
                "strength": args.strength,
                "steps": args.steps,
                "guidance": args.guidance,
                "device": device,
                "size": list(out_img.size),
            }
        )
    else:
        eprint(f"sdedit(strength={args.strength}): {args.path} -> {output} on {device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
