#!/usr/bin/env python3
"""Remove Tree-Ring watermarks via frequency notch filtering or VAE+PGD.

Exit codes:
  0  attack completed
  1  runtime error
  2  bad input
  3  engine unavailable (torch required for pgd mode)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import emit_json, eprint  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Input image")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output path")
    p.add_argument(
        "--mode", choices=("filter", "pgd"), default="filter", help="Attack mode (default: filter)"
    )
    p.add_argument("--iterations", type=int, default=100, help="PGD iterations (default: 100)")
    p.add_argument("--epsilon", type=float, default=0.03, help="PGD epsilon (default: 0.03)")
    p.add_argument("--device", type=str, default=None, help="Device for PGD (auto|cpu|cuda|mps)")
    p.add_argument("--timeout", type=int, default=600, help="Timeout in seconds (default: 600)")
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    if not args.path.is_file():
        eprint(f"not a file: {args.path}")
        return 2

    if args.mode == "pgd":
        try:
            import torch  # noqa: F401
        except ImportError:
            eprint("pgd mode requires torch")
            return 3

    try:
        from treering_attack import TreeRingAttacker

        attacker = TreeRingAttacker()
        result = attacker.attack(
            str(args.path),
            mode=args.mode,
            iterations=args.iterations,
            epsilon=args.epsilon,
            output_path=str(args.output) if args.output else None,
        )
    except ValueError as e:
        eprint(str(e))
        return 2
    except Exception as e:
        eprint(f"treering error: {e}")
        return 1

    if args.json:
        emit_json(
            {
                "output": result.output_path,
                "ring_energy_before": result.ring_energy_before,
                "ring_energy_after": result.ring_energy_after,
                "psnr": result.psnr,
                "iterations_used": result.iterations_used,
                "mode": args.mode,
            }
        )
    else:
        eprint(
            f"treering({args.mode}): ring_energy {result.ring_energy_before:.1f} -> "
            f"{result.ring_energy_after:.1f}, PSNR={result.psnr:.1f}dB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
