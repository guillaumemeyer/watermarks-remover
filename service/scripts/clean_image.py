#!/usr/bin/env python3
"""Strip C2PA and AI-related metadata from raster images (PNG/JPEG/WebP/AVIF/HEIC/BMP/GIF/TIFF)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import backup_path, cleaned_path, eprint
from image_meta import clean_image


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Input image (PNG/JPEG/WebP/AVIF/HEIC/BMP/GIF/TIFF)")
    p.add_argument("-o", "--output", type=Path, help="Output path (default: *.cleaned.*)")
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input (writes .bak backup first)",
    )
    p.add_argument(
        "--keep-non-ai-metadata",
        action="store_true",
        help="Only drop segments/chunks that look like C2PA/AI (less aggressive)",
    )
    p.add_argument("--json", action="store_true", help="JSON result on stdout")
    p.add_argument(
        "--synthid-dir",
        type=str,
        default=None,
        help="reverse-SynthID checkout root for optional pixel SynthID scoring",
    )
    p.add_argument(
        "--remove-pixel",
        choices=["ctrlregen", "diffusion"],
        default=None,
        help="Run optional pixel-watermark removal after metadata cleaning "
        "(ctrlregen = CtrlRegen regeneration; diffusion = MarkDiffusion "
        "DiffusionPurification regeneration)",
    )
    p.add_argument(
        "--ctrlregen-dir",
        type=str,
        default=None,
        help="noai-watermark checkout root (default: $NOAI_WATERMARK_DIR)",
    )
    p.add_argument(
        "--ctrlregen-intensity",
        type=float,
        default=0.25,
        help="CtrlRegen intensity in (0, 1] (default: 0.25, conservative)",
    )
    p.add_argument(
        "--ctrlregen-steps",
        type=int,
        default=50,
        help="CtrlRegen diffusion steps (default: 50)",
    )
    p.add_argument(
        "--ctrlregen-device",
        type=str,
        default=None,
        help="CtrlRegen device: auto|cpu|cuda|mps (default: auto)",
    )
    p.add_argument(
        "--ctrlregen-seed",
        type=int,
        default=None,
        help="Optional CtrlRegen RNG seed",
    )
    p.add_argument(
        "--ctrlregen-timeout",
        type=int,
        default=3600,
        help="CtrlRegen subprocess timeout in seconds (default: 3600)",
    )
    p.add_argument(
        "--markdiffusion-dir",
        type=str,
        default=None,
        help="MarkDiffusion bootstrap dir (default: $MARKDIFFUSION_DIR)",
    )
    p.add_argument(
        "--markdiffusion-intensity",
        type=float,
        default=0.3,
        help="DiffusionPurification intensity in (0, 1] (default: 0.3)",
    )
    p.add_argument(
        "--markdiffusion-model",
        type=str,
        default=None,
        help="Stable Diffusion model for purification (default: SD 2.1 base)",
    )
    p.add_argument(
        "--markdiffusion-size",
        type=int,
        default=512,
        help="Purification working size in px (default: 512)",
    )
    p.add_argument(
        "--markdiffusion-steps",
        type=int,
        default=50,
        help="Purification diffusion steps (default: 50)",
    )
    p.add_argument(
        "--markdiffusion-device",
        type=str,
        default=None,
        help="Purification device: auto|cpu|cuda|mps (default: auto)",
    )
    p.add_argument(
        "--markdiffusion-timeout",
        type=int,
        default=3600,
        help="Purification subprocess timeout in seconds (default: 3600)",
    )
    args = p.parse_args()

    if not args.path.is_file():
        eprint(f"not a file: {args.path}")
        return 2

    if args.in_place:
        bak, created = backup_path(args.path)
        if not created:
            eprint(f"backup {bak} already exists from an earlier run; keeping the original backup")
            src = args.path
        else:
            src = bak
        dest = args.path
    else:
        src = args.path
        dest = args.output or cleaned_path(args.path)

    try:
        result = clean_image(
            src,
            dest,
            strip_all_metadata=not args.keep_non_ai_metadata,
            synthid_dir=args.synthid_dir,
            remove_pixel=args.remove_pixel,
            ctrlregen_dir=args.ctrlregen_dir,
            ctrlregen_intensity=args.ctrlregen_intensity,
            ctrlregen_steps=args.ctrlregen_steps,
            ctrlregen_device=args.ctrlregen_device,
            ctrlregen_seed=args.ctrlregen_seed,
            ctrlregen_timeout=args.ctrlregen_timeout,
            markdiffusion_dir=args.markdiffusion_dir,
            markdiffusion_intensity=args.markdiffusion_intensity,
            markdiffusion_model=args.markdiffusion_model,
            markdiffusion_size=args.markdiffusion_size,
            markdiffusion_steps=args.markdiffusion_steps,
            markdiffusion_device=args.markdiffusion_device,
            markdiffusion_timeout=args.markdiffusion_timeout,
        )
    except Exception as e:
        eprint(f"error: {e}")
        return 1

    pr = result.get("pixel_removal")
    residual = result["still_has_c2pa"] or result["still_has_ai_metadata"]
    failed = residual or (pr is not None and not pr.get("available"))

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        eprint(f"wrote {result['output']} ({result['bytes_in']} -> {result['bytes_out']})")
        for a in result["actions"]:
            eprint(f"  - {a}")
        if result.get("synthid_before") and result["synthid_before"].get("available"):
            label = "yes" if result["synthid_before"].get("is_watermarked") else "no"
            eprint(
                "SynthID before: "
                f"confidence {result['synthid_before'].get('confidence', 0.0):.3f} "
                f"(watermarked: {label})"
            )
        if result.get("synthid_after") and result["synthid_after"].get("available"):
            label = "yes" if result["synthid_after"].get("is_watermarked") else "no"
            eprint(
                "SynthID after: "
                f"confidence {result['synthid_after'].get('confidence', 0.0):.3f} "
                f"(watermarked: {label})"
            )
        if pr is not None:
            if pr.get("available"):
                engine = (
                    "CtrlRegen" if args.remove_pixel == "ctrlregen" else "DiffusionPurification"
                )
                eprint(f"{engine}: removed on {pr.get('device', 'unknown device')}")
            else:
                engine = (
                    "CtrlRegen" if args.remove_pixel == "ctrlregen" else "DiffusionPurification"
                )
                eprint(f"{engine}: unavailable: {pr.get('error', 'unknown error')}")
        if residual:
            eprint("warning: residual C2PA/AI signals may remain")
            for f in result.get("post_findings") or []:
                eprint(f"  ! {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
