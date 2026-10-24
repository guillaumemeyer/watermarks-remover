#!/usr/bin/env python3
"""Apply a replacement watermark signal after removing the original.

Confuses watermark detectors by injecting a different signal so the original
cannot be verified.

Exit codes:
  0  replacement applied
  1  runtime error
  2  bad input
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import emit_json, eprint  # noqa: E402
from rewatermark_engine import REPLACEMENT_SCHEMES, ReWatermarkEngine  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Input image")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output path")
    p.add_argument(
        "--original-scheme", type=str, default="unknown", help="Original watermark scheme"
    )
    p.add_argument(
        "--replacement",
        choices=REPLACEMENT_SCHEMES,
        default="noise_ring",
        help="Replacement strategy (default: noise_ring)",
    )
    p.add_argument(
        "--strength", type=float, default=0.3, help="Injection strength 0.0-1.0 (default: 0.3)"
    )
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    if not args.path.is_file():
        eprint(f"not a file: {args.path}")
        return 2

    if not 0.0 <= args.strength <= 1.0:
        eprint(f"strength must be in [0, 1], got {args.strength}")
        return 2

    try:
        engine = ReWatermarkEngine()
        result = engine.attack(
            str(args.path),
            original_scheme=args.original_scheme,
            replacement_scheme=args.replacement,
            strength=args.strength,
            output_path=str(args.output) if args.output else None,
        )
    except ValueError as e:
        eprint(str(e))
        return 2
    except Exception as e:
        eprint(f"re-watermark error: {e}")
        return 1

    if args.json:
        emit_json(
            {
                "output": result.output_path,
                "original_scheme": result.original_scheme,
                "replacement_scheme": result.replacement_scheme,
                "injection_strength": result.injection_strength,
            }
        )
    else:
        eprint(
            f"re-watermarked: {args.path} -> {result.output_path} "
            f"({result.replacement_scheme}, strength={result.injection_strength})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
