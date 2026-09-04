#!/usr/bin/env python3
"""Fail if any given file carries AI/C2PA provenance marks.

Designed to run as a pre-commit hook: the pre-commit framework invokes this
with the staged files as positional arguments, so it never touches git state
itself — it just scans whatever paths it's given. Reuses scan_file() /
is_actionable() from audit_lib.py (audit_dir.py's own per-file logic), so
the pre-commit gate and the CI gate (#101's SARIF export) agree on exactly
what counts as actionable.

Exit codes match audit_dir.py: 0 = clean, 1 = actionable findings found,
2 = usage error (a given path does not exist / is not a file).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_lib import is_actionable, scan_file
from common import MAX_INPUT_BYTES, eprint


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", type=Path, help="Files to check (e.g. staged files)")
    p.add_argument(
        "--check-stylometry",
        action="store_true",
        help="Also evaluate text files for AI statistical & stylometric signals",
    )
    args = p.parse_args()

    actionable: list[dict] = []
    for path in args.paths:
        if not path.is_file():
            eprint(f"not a file: {path}")
            return 2
        if path.stat().st_size > MAX_INPUT_BYTES:
            eprint(f"skipping {path}: larger than {MAX_INPUT_BYTES} bytes")
            continue
        item = scan_file(path, check_stylometry=args.check_stylometry)
        if item.get("kind") == "unknown":
            continue
        if is_actionable(item):
            actionable.append(item)

    if not actionable:
        return 0

    eprint(f"watermarks-remover: {len(actionable)} file(s) carry AI/C2PA provenance marks:")
    for item in actionable:
        eprint(f"  {item['path']}")
        for finding in item.get("findings", []):
            eprint(f"    - {finding}")
        if (
            item.get("status") == "service_unavailable"
            or item.get("error") == "service_unavailable"
        ):
            detail = item.get("detail") or item.get("error") or "service unavailable"
            eprint(f"    - service_unavailable: {detail}")
        if item.get("has_c2pa"):
            eprint("    - C2PA manifest present")
        if item.get("has_ai_metadata"):
            eprint("    - AI-generator metadata present")
    eprint(
        "Run `python3 service/scripts/clean_file.py <path> --in-place` "
        "(or the watermarks-remover-clean pre-commit hook) to strip these before committing."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
