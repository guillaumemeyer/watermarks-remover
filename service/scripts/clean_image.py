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

ENGINE_AUTO = "auto"

ENGINE_NAMES = {
    "auto": "AutoChain",
    "ctrlregen": "CtrlRegen",
    "diffusion": "DiffusionPurification",
    "marknull": "MarkNull",
    "nfpa": "NFPA",
    "fmdiffwa": "FMDiffWA",
    "synthid_bypass": "SynthID Bypass",
    "treering": "TreeRing",
    "flux3": "FLUX 3",
    "chain": "AttackChain",
}


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
        choices=[
            "auto",
            "ctrlregen",
            "diffusion",
            "marknull",
            "nfpa",
            "fmdiffwa",
            "synthid_bypass",
            "treering",
            "chain",
            "flux3",
        ],
        default=None,
        help="Run optional pixel-watermark removal after metadata cleaning "
        "(auto = classify then chain; ctrlregen = CtrlRegen regeneration; "
        "diffusion = MarkDiffusion; marknull = MarkNull NLAS; nfpa = NFPA next-frame; "
        "fmdiffwa = frequency-domain Wiener; synthid_bypass = SynthID bypass; "
        "treering = Tree-Ring notch filter; chain = auto-classify and sequence; "
        "flux3 = FLUX 3 API regeneration, needs an OpenRouter key)",
    )
    p.add_argument(
        "--classify",
        action="store_true",
        help="Classify watermark type on the original image before removal "
        "and add result to JSON output "
        "(with chain: classify then run chain in-process; standalone: classify only)",
    )
    p.add_argument(
        "--stealth-check",
        action="store_true",
        help="Run forensic stealth check after removal",
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
    p.add_argument(
        "--marknull-mode",
        choices=["optimize", "fast"],
        default="optimize",
        help="MarkNull mode: optimize = NLAS gradient descent, fast = one VAE round-trip "
        "(default: optimize)",
    )
    p.add_argument(
        "--marknull-iterations",
        type=int,
        default=100,
        help="MarkNull optimisation iterations (default: 100)",
    )
    p.add_argument(
        "--marknull-device",
        type=str,
        default=None,
        help="MarkNull device: auto|cpu|cuda|mps (default: auto)",
    )
    p.add_argument(
        "--marknull-timeout",
        type=int,
        default=3600,
        help="MarkNull subprocess timeout in seconds (default: 3600)",
    )
    p.add_argument(
        "--nfpa-flow-strength",
        type=float,
        default=0.05,
        help="NFPA flow field strength in (0, 1] (default: 0.05)",
    )
    p.add_argument(
        "--nfpa-steps",
        type=int,
        default=50,
        help="NFPA DDIM steps per pass (default: 50)",
    )
    p.add_argument(
        "--nfpa-device",
        type=str,
        default=None,
        help="NFPA device: auto|cpu|cuda|mps (default: auto)",
    )
    p.add_argument(
        "--nfpa-timeout",
        type=int,
        default=3600,
        help="NFPA subprocess timeout in seconds (default: 3600)",
    )
    p.add_argument(
        "--flux3-api-key",
        type=str,
        default=None,
        help="OpenRouter API key for FLUX 3 (default: $OPENROUTER_API_KEY)",
    )
    p.add_argument(
        "--flux3-resolution",
        choices=["768", "1k", "1.5k", "2k", "4k"],
        default=None,
        help="FLUX 3 output resolution (default: auto from input size for chain, 1k for direct)",
    )
    p.add_argument(
        "--flux3-strength",
        choices=["minimal", "normal", "strong", "maximum"],
        default=None,
        help="FLUX 3 prompt strength (default: per-scheme for chain, normal for direct)",
    )
    p.add_argument(
        "--flux3-max-cost",
        type=float,
        default=0.50,
        help="Refuse FLUX 3 when the estimated cost in USD exceeds this (default: 0.50)",
    )
    p.add_argument(
        "--flux3-provider",
        choices=["openrouter", "bfl", "fal"],
        default="openrouter",
        help="FLUX 3 provider; only openrouter is implemented so far (default: openrouter)",
    )
    p.add_argument(
        "--flux3-timeout",
        type=int,
        default=600,
        help="FLUX 3 subprocess timeout in seconds (default: 600)",
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

    is_auto = args.remove_pixel == ENGINE_AUTO
    use_inprocess_chain = is_auto or (args.remove_pixel == "chain" and args.classify)
    effective_remove_pixel = None if use_inprocess_chain else args.remove_pixel

    pre_classify_result = None
    if args.classify and not use_inprocess_chain:
        from classify_watermark import WatermarkClassifier

        classifier = WatermarkClassifier()
        cr = classifier.classify(str(args.path))
        pre_classify_result = {"scheme": cr.scheme, "confidence": round(cr.confidence, 4)}

    try:
        result = clean_image(
            src,
            dest,
            strip_all_metadata=not args.keep_non_ai_metadata,
            synthid_dir=args.synthid_dir,
            remove_pixel=effective_remove_pixel,
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
            marknull_mode=args.marknull_mode,
            marknull_iterations=args.marknull_iterations,
            marknull_device=args.marknull_device,
            marknull_timeout=args.marknull_timeout,
            nfpa_flow_strength=args.nfpa_flow_strength,
            nfpa_steps=args.nfpa_steps,
            nfpa_device=args.nfpa_device,
            nfpa_timeout=args.nfpa_timeout,
            flux3_api_key=args.flux3_api_key,
            flux3_resolution=args.flux3_resolution,
            flux3_strength=args.flux3_strength,
            flux3_max_cost=args.flux3_max_cost,
            flux3_provider=args.flux3_provider,
            flux3_timeout=args.flux3_timeout,
        )
    except Exception as e:
        eprint(f"error: {e}")
        return 1

    if use_inprocess_chain:
        from attack_chain import run_chain
        from classify_watermark import WatermarkClassifier

        classifier = WatermarkClassifier()
        cr = classifier.classify(str(dest))
        scheme = cr.scheme
        result["classifier"] = {"scheme": scheme, "confidence": round(cr.confidence, 4)}
        skip_chain = scheme == "none" or (is_auto and scheme == "unknown")
        if not skip_chain:
            eprint(f"classified as {scheme} (confidence {cr.confidence:.2f}), running chain")
            chain_result = run_chain(
                dest, scheme, output_path=dest,
                api_key=args.flux3_api_key, max_cost=args.flux3_max_cost,
                flux3_resolution=args.flux3_resolution,
                flux3_strength=args.flux3_strength,
                flux3_timeout=args.flux3_timeout,
                flux3_provider=args.flux3_provider,
            )
            result["pixel_removal"] = {
                "available": True,
                "complete": chain_result.complete,
                "engine": "chain",
                "scheme": scheme,
                "steps_run": chain_result.steps_run,
                "stealth_score": chain_result.stealth_score,
                "elapsed_s": round(chain_result.elapsed_s, 2),
            }
            if not chain_result.complete:
                eprint(f"chain incomplete: {chain_result.steps_run}/{chain_result.steps_total} steps")
        else:
            eprint(f"no watermark detected (scheme={scheme}), skipping chain")
    elif pre_classify_result is not None:
        result["classifier"] = pre_classify_result

    if args.stealth_check and dest.is_file():
        import subprocess as _sp

        from common import subprocess_creationflags

        _fc = _sp.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "forensic_check.py"),
                str(dest),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            creationflags=subprocess_creationflags,
        )
        if _fc.returncode == 0:
            try:
                result["forensic"] = json.loads(_fc.stdout)
            except (json.JSONDecodeError, TypeError, ValueError):
                result["forensic"] = {"error": "parse failed"}
        else:
            result["forensic"] = {"error": f"exit {_fc.returncode}"}

    pr = result.get("pixel_removal")
    residual = result["still_has_c2pa"] or result["still_has_ai_metadata"]
    incomplete = pr is not None and not pr.get("complete", True)
    failed = residual or (pr is not None and not pr.get("available")) or incomplete

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
            engine = ENGINE_NAMES.get(
                args.remove_pixel or "", pr.get("engine", args.remove_pixel or "")
            )
            if pr.get("available") and pr.get("complete", True):
                where = pr.get("device") or pr.get("provider") or "unknown device"
                eprint(f"{engine}: removed on {where}")
            elif pr.get("available") and not pr.get("complete", True):
                steps = pr.get("steps_run", "?")
                eprint(f"{engine}: INCOMPLETE ({steps} steps ran, chain did not finish)")
            else:
                eprint(f"{engine}: unavailable: {pr.get('error', 'unknown error')}")
        if residual:
            eprint("warning: residual C2PA/AI signals may remain")
            for f in result.get("post_findings") or []:
                eprint(f"  ! {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
