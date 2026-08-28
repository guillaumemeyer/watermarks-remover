#!/usr/bin/env python3
"""Inspect raster images (PNG/JPEG/WebP/AVIF/HEIC/BMP/GIF/TIFF) for C2PA and AI-related metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import classify_finding_confidence, emit_json
from image_meta import inspect_image


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Image path (PNG/JPEG/WebP/AVIF/HEIC/BMP/GIF/TIFF)")
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--synthid-dir",
        type=str,
        default=None,
        help="reverse-SynthID checkout root for optional pixel SynthID scoring",
    )
    args = p.parse_args()

    if not args.path.is_file():
        print(f"not a file: {args.path}", file=sys.stderr)
        return 2

    report = inspect_image(args.path, synthid_dir=args.synthid_dir)
    if args.json:
        emit_json(report.to_dict())
    else:
        print(f"Path: {report.path}")
        print(f"Format: {report.format}")
        print(f"C2PA: {report.has_c2pa}")
        print(f"AI metadata: {report.has_ai_metadata}")
        if report.findings:
            print("Findings:")
            for f in report.findings:
                print(f"  - [{classify_finding_confidence(f)}] {f}")
        ct = report.tools.get("c2patool") or {}
        print(f"c2patool: {'yes' if ct.get('available') else 'no'}")
        et = report.tools.get("exiftool") or {}
        print(f"exiftool: {'yes' if et.get('available') else 'no'}")
        if et.get("interesting_lines"):
            print("exiftool highlights:")
            for line in et["interesting_lines"][:20]:
                print(f"  {line}")
        if report.synthid and report.synthid.get("available"):
            label = "yes" if report.synthid.get("is_watermarked") else "no"
            print(
                "SynthID score: "
                f"confidence {report.synthid.get('confidence', 0.0):.3f} "
                f"(watermarked: {label})"
            )
            if report.synthid.get("is_watermarked"):
                print(
                    "Hint: optional pixel removal is available via "
                    "clean_image.py IMG --remove-pixel ctrlregen "
                    "--ctrlregen-dir $NOAI_WATERMARK_DIR"
                )
        elif report.synthid and report.synthid.get("error"):
            print(f"SynthID score: error: {report.synthid['error']}")

    synthid_wm = bool(report.synthid and report.synthid.get("available") and report.synthid.get("is_watermarked"))
    return 0 if not (report.has_c2pa or report.has_ai_metadata or synthid_wm) else 1


if __name__ == "__main__":
    raise SystemExit(main())
