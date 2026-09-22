#!/usr/bin/env python3
"""Aggregate AI-provenance audit over a directory tree.

Recursively inspects supported text/image/container files and emits one
summary plus a per-file finding list with confidence classifications.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_lib import aggregate, format_sarif, print_human_report, scan_file
from common import EXIT_PARTIAL, MAX_INPUT_BYTES, emit_json, eprint

DEFAULT_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".next",
    "target",
    ".cache",
}


def walk_files(root: Path, skip_dirs: set[str]):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs and not d.startswith("."))
        for fn in sorted(filenames):
            path = Path(dirpath) / fn
            if path.is_file():
                yield path


def _scan_worker(path: Path, check_stylometry: bool) -> tuple[dict | None, dict | None]:
    try:
        if path.stat().st_size > MAX_INPUT_BYTES:
            return None, {"path": str(path), "reason": "too large"}
        return scan_file(path, check_stylometry=check_stylometry), None
    except Exception as e:  # keep the audit going on one bad file
        return None, {"path": str(path), "reason": str(e)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Directory to audit recursively")
    p.add_argument(
        "--format",
        choices=["human", "json", "sarif"],
        default="human",
        help="Output format (default: human)",
    )
    p.add_argument(
        "--json", action="store_true", help="Emit a JSON report (alias for --format json)"
    )
    p.add_argument(
        "--sarif",
        action="store_true",
        help="Emit an OASIS SARIF 2.1.0 report (alias for --format sarif)",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=min(32, (os.cpu_count() or 1) + 4),
        help="Number of concurrent worker threads (default: CPU cores + 4)",
    )
    p.add_argument(
        "--check-stylometry",
        action="store_true",
        help="Also evaluate text files for AI statistical & stylometric signals",
    )
    p.add_argument(
        "--skip",
        default="",
        help="Comma-separated extra directory names to skip",
    )
    args = p.parse_args()

    root = args.path
    if not root.is_dir():
        eprint(f"not a directory: {root}")
        return 2

    skip_dirs = set(DEFAULT_SKIP_DIRS)
    for raw_part in args.skip.split(","):
        part = raw_part.strip()
        if part:
            skip_dirs.add(part)

    paths = list(walk_files(root, skip_dirs))
    files: list[dict] = []
    skipped: list[dict] = []

    if args.jobs <= 1:
        for path in paths:
            f, s = _scan_worker(path, args.check_stylometry)
            if f is not None:
                files.append(f)
            if s is not None:
                skipped.append(s)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(_scan_worker, p, args.check_stylometry) for p in paths]
            for fut in futures:
                f, s = fut.result()
                if f is not None:
                    files.append(f)
                if s is not None:
                    skipped.append(s)

    files.sort(key=lambda x: str(x.get("path", "")))
    skipped.sort(key=lambda x: str(x.get("path", "")))

    summary = aggregate(files)
    report = {
        "root": str(root),
        # Not len(files): "files" also holds the items no pipeline claimed, and
        # those were never opened. Their count is summary["unrecognized"].
        "files_scanned": summary["read"],
        "files_skipped": skipped,
        "summary": summary,
        "files": files,
    }

    out_format = args.format
    if args.json:
        out_format = "json"
    elif args.sarif:
        out_format = "sarif"

    if out_format == "json":
        emit_json(report)
    elif out_format == "sarif":
        sarif_doc = format_sarif(report)
        emit_json(sarif_doc)
    else:
        print_human_report(
            files,
            summary,
            extra_header={
                "Root": report["root"],
                "Files skipped": str(len(skipped)),
            },
        )

    # A partial scan (one or more files could not be scanned) is reported
    # with a distinct code in every output format: incomplete audits are
    # the more important CI signal and take precedence over actionable.
    return EXIT_PARTIAL if skipped else (1 if summary["actionable_files"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
