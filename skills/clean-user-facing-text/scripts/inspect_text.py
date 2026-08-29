#!/usr/bin/env python3
"""Inspect text for invisible Unicode / space homoglyphs (Layer A) and optional stylometry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Allow running as script from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import emit_json, read_text_input
from score_stylometry import print_human_stylometry_report, score_text_stylometry
from text_unicode import human_report, inspect_text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="?", default="-", help="Text file path, or - for stdin")
    p.add_argument("--json", action="store_true", help="JSON report")
    p.add_argument(
        "--aggressive",
        action="store_true",
        help="Also flag Latin confusable / fullwidth lookalikes",
    )
    p.add_argument(
        "--strip-emoji-glue",
        action="store_true",
        help="Paranoid: flag all load-bearing invisibles too (emoji glue, script joiners, flag tags, same-script fillers/selectors, orthographic Cf)",
    )
    p.add_argument(
        "--stylometry",
        action="store_true",
        help="Also run zero-LLM statistical and stylometric AI cadence scoring",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Score threshold for --stylometry exit code (default: 0.65)",
    )
    p.add_argument(
        "--force-text",
        action="store_true",
        help="Scan even when the input looks like a binary container",
    )
    p.add_argument(
        "--audit",
        action="store_true",
        help="Detect-only audit: flag Unicode and stylometric spans without rewriting "
        "and emit a structured span list (implies --stylometry)",
    )
    args = p.parse_args()

    if args.audit:
        args.stylometry = True

    # read_text_input raises SystemExit(2) on unreadable/oversized/non-regular input.
    text = read_text_input(args.path, allow_binary=args.force_text)

    report = inspect_text(
        text,
        aggressive=args.aggressive,
        strip_emoji_glue=args.strip_emoji_glue,
    )

    stylometry_report = None
    if args.stylometry:
        input_label = "<stdin>" if args.path == "-" else args.path
        stylometry_report = score_text_stylometry(text, path=input_label)

    if args.audit:
        flagged: list[dict[str, Any]] = []
        for hit in report.to_dict()["hits"]:
            flagged.append(
                {
                    "detector": "unicode",
                    "kind": hit["kind"],
                    "label": hit["label"],
                    "count": hit["count"],
                    "sample_offsets": hit["sample_offsets"],
                    "severity": hit["confidence"],
                }
            )
        density_tier = "uncalibrated"
        if stylometry_report:
            density_tier = stylometry_report.density_tier
            for m in stylometry_report.matched_markers:
                flagged.append(
                    {
                        "detector": "stylometry",
                        "phrase": m["phrase"],
                        "count": m["count"],
                        "weight": m["weight"],
                        "samples": m.get("samples", []),
                        "spans": m.get("spans", []),
                        "severity": density_tier,
                    }
                )
        audit: dict[str, Any] = {
            "path": "<stdin>" if args.path == "-" else args.path,
            "density_tier": density_tier,
            "flagged_count": len(flagged),
            "flagged": flagged,
        }
        if stylometry_report:
            audit["stylometry"] = stylometry_report.to_dict()
        emit_json(audit)
    elif args.json:
        data = report.to_dict()
        if stylometry_report:
            data["stylometry"] = stylometry_report.to_dict()
        emit_json(data)
    else:
        print(human_report(report))
        if stylometry_report:
            print("\n" + "=" * 50 + "\n")
            print_human_stylometry_report(stylometry_report, explain=True)

    exit_code = 0
    if report.suspicious_total > 0:
        exit_code = 1
    if (
        stylometry_report
        and stylometry_report.status == "ok"
        and stylometry_report.score is not None
        and stylometry_report.score >= args.threshold
    ):
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
