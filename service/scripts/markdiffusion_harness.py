#!/usr/bin/env python3
"""Optional MarkDiffusion image-watermark harness backed by the external
THU-BPM/MarkDiffusion package (Apache-2.0).

This script does NOT vendor upstream code. It imports ``markdiffusion`` either
from a user-provided checkout (--upstream-dir / $MARKDIFFUSION_DIR) or from the
environment's installed package (``pip install markdiffusion[optional]``). The
checkout/venv environment must supply torch + diffusers.

MarkDiffusion is a *generative watermarking* toolkit (it embeds marks). We use
it as a research/verification harness for controlled experiments on images you
own: watermark a test image, run removal, and re-detect with the SAME scheme
config. Detection is only valid against the same scheme config, model, and keys
used at generation — it cannot certify that a vendor detector will fail on the
given image.

Subcommands:
  watermark  generate a watermarked (and optionally unwatermarked) image from
             a prompt, for controlled before/after experiments
  detect     run same-scheme detection on an image
  purify     run the DiffusionPurification regeneration attack on an image
             (optional pixel-watermark removal engine)

Exit codes:
  0  success
  1  runtime error (model load, detection/generation failure)
  2  bad input (missing/unreadable file, binary input, bad args)
  3  unavailable (not configured / missing package / missing deps)
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import emit_json, eprint, read_text_input, safe_write_bytes  # noqa: E402

# User-facing scheme names -> MarkDiffusion algorithm names (image-only; the
# video algorithms VideoShield/VideoMark are out of scope here). Canonical
# upstream names are accepted as-is.
SCHEMES = {
    "tr": "TR",
    "ringid": "RI",
    "robin": "ROBIN",
    "wind": "WIND",
    "sfw": "SFW",
    "gaussianshading": "GS",
    "gaussmarker": "GM",
    "prc": "PRC",
    "seal": "SEAL",
}

IMAGE_SCHEMES = {"TR", "RI", "ROBIN", "WIND", "SFW", "GS", "GM", "PRC", "SEAL"}

DEFAULT_MODEL = "huanzi05/stable-diffusion-2-1-base"

# Algorithm configs are a few hundred bytes (TR.json/GS.json). Cap well above
# that so a crafted or accidental huge file is refused before either this script
# or upstream reads it into memory.
MAX_CONFIG_BYTES = 1 << 20


class _Unavailable(RuntimeError):
    """Backend present but unusable (missing package/checkout, missing deps)."""


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


def normalize_scheme(raw: str) -> str:
    up = raw.upper()
    if up in IMAGE_SCHEMES:
        return up
    if raw.lower() in SCHEMES:
        return SCHEMES[raw.lower()]
    raise ValueError(f"unknown scheme {raw!r}; image schemes: " + ", ".join(sorted(SCHEMES)))


def _import_markdiffusion(upstream: Path | None) -> Any:
    """Import the ``markdiffusion`` module from a checkout or the environment."""
    if upstream is not None:
        sys.path.insert(0, str(upstream))
    try:
        import markdiffusion
    except ImportError as e:
        raise _Unavailable(
            "markdiffusion not importable: "
            + str(e)
            + " (set MARKDIFFUSION_DIR / --upstream-dir to a checkout, or "
            "pip install markdiffusion[optional])"
        ) from e
    return markdiffusion


def _load_diffusion(model: str, device: str, offline: bool, size: int):
    """Load the Stable Diffusion pipeline and scheduler used by the harness."""
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    load_kwargs = {"local_files_only": True} if offline else {}

    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        model, subfolder="scheduler", **load_kwargs
    )
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        scheduler=scheduler,
        torch_dtype=dtype,
        safety_checker=None,
        **load_kwargs,
    ).to(device)
    return pipe, scheduler


def _resolve_config(upstream: Path | None, config: str | None) -> str | None:
    if not config:
        return None
    path = Path(config).expanduser().resolve()
    if not path.is_file():
        raise _Unavailable(f"MarkDiffusion config not found: {path}")
    try:
        size = path.stat().st_size
    except OSError as e:
        raise _Unavailable(f"cannot stat MarkDiffusion config {path}: {e}") from e
    if size > MAX_CONFIG_BYTES:
        raise _Unavailable(
            f"MarkDiffusion config too large ({size} bytes > {MAX_CONFIG_BYTES}): {path}"
        )
    return str(path)


def _json_safe(obj: Any) -> Any:
    """Convert numpy/torch scalars so the payload is JSON-serializable."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:  # noqa: S110 - optional numpy import
        pass
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.item()
    except Exception:  # noqa: S110 - optional torch import
        pass
    try:
        return float(obj)
    except Exception:
        return str(obj)


def _detect_payload(result: dict, config_dict: dict | None) -> dict[str, Any]:
    is_wm = bool(result.get("is_watermarked", False))
    metrics = _json_safe(result)
    score: float | None = None
    for key in ("p_value", "l1_distance", "distance", "score", "acc", "corr"):
        v = metrics.get(key)
        if isinstance(v, (int, float)):
            score = float(v)
            break
    if score is None:
        for v in metrics.values():
            if isinstance(v, (int, float)) and str(v) != str(is_wm):
                score = float(v)
                break
    cfg = config_dict or {}
    threshold = cfg.get("threshold")
    if not isinstance(threshold, (int, float)):
        threshold = None
    threshold_p = cfg.get("threshold_p_value")
    if not isinstance(threshold_p, (int, float)):
        threshold_p = None
    return {
        "is_watermarked": is_wm,
        "score": score,
        "threshold": threshold,
        "threshold_p_value": threshold_p,
        "metrics": metrics,
    }


def _save_png(image: Any, path: str) -> None:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    safe_write_bytes(path, buf.getvalue())


def _cmd_watermark(args: argparse.Namespace, upstream: Path | None, scheme: str) -> int:
    prompt = read_text_input(args.prompt, allow_binary=args.force_text)

    device = resolve_device(args.device)
    config_path = _resolve_config(upstream, args.config)

    try:
        _import_markdiffusion(upstream)
        from markdiffusion.utils import DiffusionConfig
        from markdiffusion.watermark import AutoWatermark

        pipe, scheduler = _load_diffusion(args.model, device, args.offline, args.size)
        diffusion_config = DiffusionConfig(
            scheduler=scheduler,
            pipe=pipe,
            device=device,
            image_size=(args.size, args.size),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            gen_seed=args.seed,
            inversion_type="ddim",
        )
        wm = AutoWatermark.load(
            scheme,
            algorithm_config=config_path,
            diffusion_config=diffusion_config,
        )
        if args.seed is not None:
            import torch

            torch.manual_seed(args.seed)
        watermarked = wm.generate_watermarked_media(
            prompt,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            height=args.size,
            width=args.size,
        )
        unwatermarked = None
        if args.unwatermarked_output:
            unwatermarked = wm.generate_unwatermarked_media(prompt)
    except _Unavailable as e:
        eprint(str(e))
        return 3
    except Exception as e:
        eprint(f"generation error: {e}")
        return 1

    wm_out = args.watermarked_output
    _save_png(watermarked, wm_out)
    if unwatermarked is not None:
        _save_png(unwatermarked, args.unwatermarked_output)

    payload = {
        "available": True,
        "upstream_dir": str(upstream) if upstream else None,
        "scheme": scheme,
        "config": config_path,
        "model": args.model,
        "device": device,
        "watermarked_output": wm_out,
        "unwatermarked_output": args.unwatermarked_output,
    }

    if args.json:
        emit_json(payload)
    else:
        print(f"{scheme}: watermarked image -> {wm_out}")
        if unwatermarked is not None:
            print(f"      unwatermarked image -> {args.unwatermarked_output}")
    return 0


def _cmd_detect(args: argparse.Namespace, upstream: Path | None, scheme: str) -> int:
    if args.path != "-" and not Path(args.path).is_file():
        eprint(f"not a file: {args.path}")
        return 2

    device = resolve_device(args.device)
    config_path = _resolve_config(upstream, args.config)

    try:
        _import_markdiffusion(upstream)
        from markdiffusion.utils import DiffusionConfig
        from markdiffusion.watermark import AutoWatermark
        from PIL import Image

        pipe, scheduler = _load_diffusion(args.model, device, args.offline, args.size)
        diffusion_config = DiffusionConfig(
            scheduler=scheduler,
            pipe=pipe,
            device=device,
            image_size=(args.size, args.size),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            gen_seed=args.seed,
            inversion_type="ddim",
        )
        wm = AutoWatermark.load(
            scheme,
            algorithm_config=config_path,
            diffusion_config=diffusion_config,
        )
        image = Image.open(args.path).convert("RGB")
        kwargs: dict[str, Any] = {}
        if args.detector_type:
            kwargs["detector_type"] = args.detector_type
        result = wm.detect_watermark_in_media(image, prompt=args.prompt or "", **kwargs)
        config_dict = getattr(getattr(wm, "config", None), "config_dict", None)
    except _Unavailable as e:
        eprint(str(e))
        return 3
    except Exception as e:
        eprint(f"detection error: {e}")
        return 1

    det = _detect_payload(result, config_dict)
    payload = {
        "available": True,
        "upstream_dir": str(upstream) if upstream else None,
        "scheme": scheme,
        "config": config_path,
        "model": args.model,
        "device": device,
        **det,
    }

    if args.json:
        emit_json(payload)
    else:
        label = "watermarked" if det["is_watermarked"] else "not watermarked"
        score_txt = f"{det['score']:.4f}" if det["score"] is not None else "n/a"
        print(f"{scheme}: {label} (score {score_txt})")
    return 0


def _cmd_purify(args: argparse.Namespace, upstream: Path | None) -> int:
    if not Path(args.path).is_file():
        eprint(f"not a file: {args.path}")
        return 2

    device = resolve_device(args.device)

    try:
        _import_markdiffusion(upstream)
        from markdiffusion.evaluation.tools.image_editor import DiffusionPurification
        from markdiffusion.utils import DiffusionConfig
        from PIL import Image

        pipe, scheduler = _load_diffusion(args.model, device, args.offline, args.size)
        diffusion_config = DiffusionConfig(
            scheduler=scheduler,
            pipe=pipe,
            device=device,
            image_size=(args.size, args.size),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            inversion_type="ddim",
        )
        purifier = DiffusionPurification(
            diffusion_config,
            purification_strength=args.purification_intensity,
            prompt=args.prompt or "",
        )
        image = Image.open(args.path).convert("RGB")
        purified = purifier.edit(image)
    except _Unavailable as e:
        eprint(str(e))
        return 3
    except Exception as e:
        eprint(f"purification error: {e}")
        return 1

    _save_png(purified, args.output)
    payload = {
        "available": True,
        "upstream_dir": str(upstream) if upstream else None,
        "model": args.model,
        "device": device,
        "output": args.output,
        "purification_intensity": args.purification_intensity,
    }

    if args.json:
        emit_json(payload)
    else:
        print(f"purified image (intensity {args.purification_intensity}) -> {args.output}")
    return 0


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--upstream-dir",
        type=Path,
        default=None,
        help="MarkDiffusion checkout root (default: $MARKDIFFUSION_DIR); "
        "when unset the installed markdiffusion package is used",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("MARKDIFFUSION_MODEL", DEFAULT_MODEL),
        help=f"HF Stable Diffusion model (default: $MARKDIFFUSION_MODEL or {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="auto|cpu|cuda|mps (default: auto)",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Never contact the HF hub: load the model from the local cache only",
    )
    p.add_argument(
        "--force-text",
        action="store_true",
        help="Process input even when it looks like a binary container",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    wm = sub.add_parser("watermark", help="Generate a watermarked sample image")
    wm.add_argument("prompt", help="Prompt file, or - for stdin")
    wm.add_argument(
        "-o",
        "--watermarked-output",
        required=True,
        help="Output PNG path (images cannot go to stdout)",
    )
    wm.add_argument(
        "-o2",
        "--unwatermarked-output",
        default=None,
        help="Also write an unwatermarked sample to this path",
    )
    wm.add_argument(
        "--scheme", default="tr", help="Scheme: " + ", ".join(sorted(SCHEMES)) + " (default: tr)"
    )
    wm.add_argument("--config", default=None, help="Algorithm config JSON (default: bundled)")
    wm.add_argument("--size", type=int, default=512, help="Image size in px (default: 512)")
    wm.add_argument("--steps", type=int, default=50, help="Diffusion steps (default: 50)")
    wm.add_argument("--guidance", type=float, default=7.5, help="Guidance scale (default: 7.5)")
    wm.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    _add_common(wm)
    wm.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    wm.set_defaults(handler=_cmd_watermark)

    det = sub.add_parser("detect", help="Same-scheme detection on an image")
    det.add_argument("path", help="Image file to detect on")
    det.add_argument(
        "--scheme", default="tr", help="Scheme: " + ", ".join(sorted(SCHEMES)) + " (default: tr)"
    )
    det.add_argument("--config", default=None, help="Algorithm config JSON (default: bundled)")
    det.add_argument("--prompt", default=None, help="Optional prompt used at generation")
    det.add_argument(
        "--detector-type",
        default=None,
        help="Detector variant (e.g. l1_distance, p_value; scheme-dependent)",
    )
    det.add_argument("--size", type=int, default=512, help="Image size in px (default: 512)")
    det.add_argument("--steps", type=int, default=50, help="Diffusion steps (default: 50)")
    det.add_argument("--guidance", type=float, default=7.5, help="Guidance scale (default: 7.5)")
    det.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    _add_common(det)
    det.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    det.set_defaults(handler=_cmd_detect)

    pf = sub.add_parser("purify", help="Run the DiffusionPurification regeneration attack")
    pf.add_argument("path", help="Image file to purify")
    pf.add_argument("-o", "--output", required=True, help="Output PNG path")
    pf.add_argument(
        "--purification-intensity",
        type=float,
        default=0.3,
        help="Fraction of the diffusion schedule to regenerate in (0, 1] (default: 0.3)",
    )
    pf.add_argument("--prompt", default=None, help="Optional prompt for denoising (default: '')")
    pf.add_argument("--size", type=int, default=512, help="Image size in px (default: 512)")
    pf.add_argument("--steps", type=int, default=50, help="Diffusion steps (default: 50)")
    pf.add_argument("--guidance", type=float, default=7.5, help="Guidance scale (default: 7.5)")
    _add_common(pf)
    pf.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    pf.set_defaults(handler=_cmd_purify)

    args = p.parse_args()

    try:
        scheme = normalize_scheme(args.scheme) if args.cmd in ("watermark", "detect") else None
    except ValueError as e:
        eprint(str(e))
        return 2

    raw_upstream = args.upstream_dir or os.environ.get("MARKDIFFUSION_DIR")
    upstream = resolve_upstream(str(raw_upstream) if raw_upstream else None)

    if args.cmd == "purify":
        return args.handler(args, upstream)
    return args.handler(args, upstream, scheme)


if __name__ == "__main__":
    raise SystemExit(main())
