#!/usr/bin/env python3
"""Forensic stealth evaluation: measure how detectable each engine's output is.

For each engine output from the WAVES evaluation, runs forensic_check.py and
computes per-engine forensic detectability rates. Ranks engines by the
composite score: removal_success * (1 - forensic_detectability) * quality.

Exit codes:
  0  evaluation completed
  1  runtime error
  2  bad input
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from common import emit_json, eprint, subprocess_creationflags  # noqa: E402


class ForensicEvalResult(NamedTuple):
    engine: str
    n_images: int
    mean_stealth: float
    detection_rate: float
    mean_psnr: float
    composite_score: float


def evaluate_engine_outputs(
    output_dir: Path,
    engine: str,
    stealth_threshold: float = 0.5,
) -> ForensicEvalResult:
    stealth_scores: list[float] = []
    detected = 0

    all_files = sorted(output_dir.glob("*.png")) + sorted(output_dir.glob("*.jpg"))
    image_files = [f for f in all_files if ".step" not in f.stem]
    if not image_files:
        return ForensicEvalResult(
            engine=engine,
            n_images=0,
            mean_stealth=1.0,
            detection_rate=1.0,
            mean_psnr=0.0,
            composite_score=0.0,
        )

    for img_path in image_files:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "forensic_check.py"),
                str(img_path),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            creationflags=subprocess_creationflags,
        )
        if result.returncode != 0:
            stealth_scores.append(1.0)
            detected += 1
            continue

        try:
            report = json.loads(result.stdout)
            score = float(report.get("stealth_score", 1.0))
            stealth_scores.append(score)
            if score >= stealth_threshold:
                detected += 1
        except (json.JSONDecodeError, TypeError, ValueError):
            stealth_scores.append(1.0)
            detected += 1

    n = len(stealth_scores)
    mean_stealth = float(np.mean(stealth_scores)) if stealth_scores else 1.0
    detection_rate = detected / n if n > 0 else 1.0
    composite = (1.0 - detection_rate) * (1.0 - mean_stealth)

    return ForensicEvalResult(
        engine=engine,
        n_images=n,
        mean_stealth=round(mean_stealth, 4),
        detection_rate=round(detection_rate, 4),
        mean_psnr=0.0,
        composite_score=round(composite, 4),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Directory with engine output subdirs (from run_full_eval.sh)",
    )
    p.add_argument(
        "--stealth-threshold",
        type=float,
        default=0.5,
        help="Stealth score above which the image is 'detected' (default: 0.5)",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    if not args.eval_dir.is_dir():
        eprint(f"not a directory: {args.eval_dir}")
        return 2

    results: list[dict] = []
    t0 = time.monotonic()

    for subdir in sorted(args.eval_dir.iterdir()):
        if not subdir.is_dir():
            continue
        parts = subdir.name.rsplit("_", 2)
        engine = parts[1] if len(parts) >= 3 else subdir.name

        result = evaluate_engine_outputs(subdir, engine, args.stealth_threshold)
        results.append(
            {
                "dir": subdir.name,
                "engine": result.engine,
                "n_images": result.n_images,
                "mean_stealth": result.mean_stealth,
                "detection_rate": result.detection_rate,
                "composite_score": result.composite_score,
            }
        )

    elapsed = time.monotonic() - t0

    if args.json:
        emit_json(
            {
                "results": results,
                "n_dirs": len(results),
                "elapsed_s": round(elapsed, 2),
            }
        )
    else:
        for r in sorted(results, key=lambda x: x["composite_score"], reverse=True):
            eprint(
                f"{r['dir']}: stealth={r['mean_stealth']:.3f} "
                f"detect_rate={r['detection_rate']:.3f} "
                f"composite={r['composite_score']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
