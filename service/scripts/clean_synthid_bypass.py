#!/usr/bin/env python3
"""SynthID-Image robustness pipeline (reverse-SynthID V4 style spectral attack).

Runs an image through a configurable chain of lossy transforms and writes the
result, so a SynthID detector can then measure whether the watermark survived.
Stages run in the order given:

  vae      encode/decode through a Stable Diffusion VAE (AutoencoderKL); falls
           back to a JPEG q=85 round trip when torch/diffusers are not installed
  warp     smooth random elastic deformation (scipy.ndimage)
  squeeze  scale pixel values by a factor slightly below 1
  jpeg     JPEG round trip at --jpeg-quality

Exit codes:
  0  pipeline ran and the output was written
  1  runtime error
  2  bad input (missing/unreadable image, bad args)
  3  unavailable (a required dependency is missing)
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
from numpy import ndarray

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from clean_ctrlregen import resolve_device, save_image_bytes  # noqa: E402
from common import cleaned_path, emit_json, eprint, safe_write_bytes  # noqa: E402
from markdiffusion_harness import DEFAULT_MODEL  # noqa: E402

VAE_FALLBACK_JPEG_QUALITY = 85
SQUEEZE_FACTOR = 0.95
WARP_STRENGTH = 0.01
STAGES = ("vae", "warp", "squeeze", "jpeg")
DEFAULT_PIPELINE = "warp,squeeze,jpeg"


def _vae_deps_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "diffusers"))


def _jpeg_roundtrip(image: ndarray, quality: int) -> ndarray:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as decoded:
        return np.asarray(decoded)


def vae_roundtrip(image: ndarray, device: str = "cpu") -> ndarray:
    """Encode an HxWx3 uint8 image through the SD VAE latent space and decode it back."""
    try:
        import torch
        from diffusers import AutoencoderKL
    except ImportError:
        return _jpeg_roundtrip(image, VAE_FALLBACK_JPEG_QUALITY)

    vae = AutoencoderKL.from_pretrained(DEFAULT_MODEL, subfolder="vae").to(device).eval()
    # The VAE downsamples by 8; pad to a multiple of 8 and crop back afterwards.
    h, w = image.shape[:2]
    padded = np.pad(image, ((0, -h % 8), (0, -w % 8), (0, 0)), mode="edge")
    x = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float()
    x = (x / 127.5 - 1.0).to(device)
    with torch.no_grad():
        latents = vae.encode(x).latent_dist.mode()
        decoded = vae.decode(latents).sample
    out = ((decoded.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
    return out.squeeze(0).permute(1, 2, 0).cpu().numpy()[:h, :w]


def elastic_warp(image: ndarray, strength: float = 0.01) -> ndarray:
    """Displace pixels along a smooth random field.

    ``strength`` is the peak displacement as a fraction of the longer side
    (0.01 on a 512px image moves pixels by at most ~5px).
    """
    from scipy.ndimage import gaussian_filter, map_coordinates

    h, w = image.shape[:2]
    rng = np.random.default_rng()
    sigma = max(h, w) / 16
    peak = strength * max(h, w)

    def _field() -> ndarray:
        field = gaussian_filter(rng.uniform(-1.0, 1.0, (h, w)), sigma, mode="reflect")
        scale = np.abs(field).max()
        return field * (peak / scale) if scale > 0 else field

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.stack([yy + _field(), xx + _field()])
    channels = image.reshape(h, w, -1).astype(np.float64)
    warped = np.stack(
        [
            map_coordinates(channels[..., c], coords, order=1, mode="reflect")
            for c in range(channels.shape[2])
        ],
        axis=-1,
    )
    return np.clip(np.rint(warped), 0, 255).astype(image.dtype).reshape(image.shape)


def channel_squeeze(image: ndarray, factor: float = 0.95) -> ndarray:
    """Scale every channel by ``factor`` to compress the dynamic range."""
    squeezed = np.clip(image.astype(np.float32) * factor, 0, 255)
    return np.rint(squeezed).astype(np.uint8)


def parse_pipeline(raw: str) -> list[str]:
    """Split a ``,``- or ``+``-separated stage list; raise ValueError on bad input."""
    stages = [s.strip().lower() for s in raw.replace("+", ",").split(",") if s.strip()]
    if not stages:
        raise ValueError("pipeline is empty")
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown pipeline stage(s): {', '.join(unknown)}")
    return stages


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("path", type=Path, help="Input image (PNG/JPEG/etc.)")
    p.add_argument("-o", "--output", type=Path, help="Output path (default: *.synthid-bypass.*)")
    p.add_argument(
        "--pipeline",
        default=DEFAULT_PIPELINE,
        help=f"Stages from {{{','.join(STAGES)}}}, comma- or plus-separated "
        f"(default: {DEFAULT_PIPELINE})",
    )
    p.add_argument(
        "--jpeg-quality", type=int, default=90, help="Quality for the jpeg stage (default: 90)"
    )
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps for the vae stage")
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    if not args.path.is_file():
        eprint(f"not a file: {args.path}")
        return 2
    try:
        stages = parse_pipeline(args.pipeline)
    except ValueError as e:
        eprint(f"bad --pipeline: {e}")
        return 2
    if not 1 <= args.jpeg_quality <= 100:
        eprint(f"jpeg-quality must be in [1, 100]: {args.jpeg_quality}")
        return 2

    if importlib.util.find_spec("PIL") is None:
        eprint("Pillow is required: pip install pillow")
        return 3
    if "warp" in stages and importlib.util.find_spec("scipy") is None:
        eprint("the warp stage needs scipy: pip install scipy")
        return 3

    from PIL import Image

    try:
        with Image.open(args.path) as img:
            image = np.asarray(img.convert("RGB"))
    except Exception as e:
        eprint(f"could not load image: {e}")
        return 2

    # Only the vae stage can use an accelerator; the numpy/PIL stages run on the CPU.
    device = resolve_device(args.device) if "vae" in stages else "cpu"
    vae_backend = None
    if "vae" in stages:
        vae_backend = "autoencoderkl" if _vae_deps_available() else "jpeg-fallback"

    result = image
    try:
        for stage in stages:
            if stage == "vae":
                result = vae_roundtrip(result, device=device)
            elif stage == "warp":
                result = elastic_warp(result, strength=WARP_STRENGTH)
            elif stage == "squeeze":
                result = channel_squeeze(result, factor=SQUEEZE_FACTOR)
            else:
                result = _jpeg_roundtrip(result, args.jpeg_quality)
    except Exception as e:
        eprint(f"pipeline error in stage {stage!r}: {e}")
        return 1

    output = args.output or cleaned_path(args.path, ".synthid-bypass")
    out_image = Image.fromarray(result)
    try:
        data = save_image_bytes(out_image, output)
        safe_write_bytes(output, data)
    except (OSError, ValueError) as e:
        eprint(f"cannot write output: {e}")
        return 1

    payload = {
        "available": True,
        "engine": "synthid-bypass",
        "output": str(output),
        "pipeline": stages,
        "jpeg_quality": args.jpeg_quality if "jpeg" in stages else None,
        "vae_backend": vae_backend,
        "device": device,
        "input_size": [image.shape[1], image.shape[0]],
        "output_size": list(out_image.size),
        "bytes_out": len(data),
    }
    if args.json:
        emit_json(payload)
    else:
        print(
            f"SynthID bypass pipeline: {args.path} -> {output} "
            f"(stages {'+'.join(stages)}, {payload['input_size']} -> {payload['output_size']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
