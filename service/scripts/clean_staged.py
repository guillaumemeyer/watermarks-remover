#!/usr/bin/env python3
"""Strip AI/C2PA provenance marks from given files in place.

Designed to run as a pre-commit hook: the pre-commit framework invokes this
with the staged files as positional arguments. It wraps clean_file.py
--in-place as a subprocess per file — no cleaning logic is duplicated here,
this is purely a batch-over-staged-files wrapper.

Rewriting a file a developer just staged is a real behavior change, not a
lint-and-continue: following the standard pre-commit auto-fixer convention
(same as e.g. black / ruff --fix hooks), this exits 1 whenever it modified
at least one file, so the hook framework stops the commit and the developer
reviews the diff and re-stages. Exit 0 means every file was already clean.

A file clean_file.py could not process at all exits 3 (EXIT_PARTIAL) — the
code audit_dir.py / audit_website.py already use for an incomplete run — and
it outranks the auto-fix 1: "this file was never cleaned" is a different
instruction than "your files were rewritten, re-stage them". Folding the two
into one "skipped" bucket is what let a cleaner that crashed, was killed, or
died before writing its report exit 0 and carry an uncleaned file into the
commit, with its traceback invisible outside pre-commit's verbose mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    EXIT_PARTIAL,
    MAX_INPUT_BYTES,
    eprint,
    result_has_changes,
    subprocess_creationflags,
)

CLEAN_FILE_PY = Path(__file__).resolve().parent / "clean_file.py"


def _file_digest(path: Path) -> bytes | None:
    """Compute the SHA-256 digest of *path* incrementally in 64 KiB chunks."""
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.digest()
    except OSError:
        return None


def _is_modifying_action(action: str) -> bool:
    """Return True if an action message indicates a real modification rather than filler/warning."""
    a = action.strip().lower()
    if a.startswith("no ") or a.startswith("warning:") or a.startswith("c2patool available"):
        return False
    if a.startswith("deep image pass not needed") or a.startswith("kept "):
        return False
    return not (" skipped" in a or " failed" in a)


def _changed(result: dict) -> bool:
    """Determine whether a clean_file.py JSON report indicates content was modified."""
    return result_has_changes(result)


def _failure_detail(proc: subprocess.CompletedProcess[str], summary: str) -> str:
    """Reduce clean_file.py's stderr to the one line worth putting in the report.

    An uncaught exception arrives as a multi-line traceback whose last line
    names the actual cause ("OSError: refusing to write through symlink: …");
    a killed child leaves no stderr at all, so fall back to naming the exit
    status. Echoing the whole traceback under a path prefix, as this used to,
    buries the reason in stack frames the developer has to read backwards.
    """
    said = [line.strip() for line in proc.stderr.splitlines() if line.strip()]
    if said:
        return f"{summary}: {said[-1]}"
    return f"{summary} (exit {proc.returncode})"


def _clean_one(path: Path) -> tuple[str, str]:
    """Clean a single file in place and report its outcome.

    Returns ('changed' | 'unchanged' | 'skipped' | 'failed', detail).
    *detail* explains a 'failed' status and is empty for every other status.
    """
    try:
        if path.stat().st_size > MAX_INPUT_BYTES:
            eprint(f"skipping {path}: larger than {MAX_INPUT_BYTES} bytes")
            return "skipped", ""
    except OSError:
        pass

    before_digest = _file_digest(path)

    proc = subprocess.run(
        [sys.executable, str(CLEAN_FILE_PY), str(path), "--in-place", "--json"],
        capture_output=True,
        text=True,
        check=False,
        creationflags=subprocess_creationflags,
    )
    if proc.returncode == 2:
        # Unrecognized format or oversized input; clean_file.py already
        # explained why on stderr, this is a non-fatal skip for the hook.
        return "skipped", ""
    # Every other outcome is judged on the JSON report, not on the exit code:
    # clean_file.py exits 1 for a *successful* clean that left residual
    # signals behind (pinned by tests/test_json_exit_code.py), so a non-zero
    # code is not by itself a failure. A run that produced no parsable report
    # never reached its write, though — that file is still marked.
    if not proc.stdout.strip():
        return "failed", _failure_detail(proc, "clean_file.py wrote no report")
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return "failed", _failure_detail(proc, "clean_file.py wrote an unparsable report")

    after_digest = _file_digest(path)

    if before_digest is not None and after_digest is not None:
        status = "changed" if before_digest != after_digest else "unchanged"
    else:
        status = "changed" if _changed(result) else "unchanged"
    return status, ""


def main() -> int:
    """Run in-place cleaning across all given staged file paths."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "paths", nargs="+", type=Path, help="Files to clean in place (e.g. staged files)"
    )
    args = p.parse_args()

    changed_paths: list[Path] = []
    failures: list[tuple[Path, str]] = []
    for path in args.paths:
        if not path.is_file():
            eprint(f"not a file: {path}")
            continue
        status, detail = _clean_one(path)
        if status == "changed":
            changed_paths.append(path)
        elif status == "failed":
            failures.append((path, detail))

    if changed_paths:
        eprint(f"watermarks-remover: cleaned {len(changed_paths)} file(s) in place:")
        for path in changed_paths:
            eprint(f"  {path}")
        eprint("Review the changes and re-stage before committing.")

    if failures:
        eprint(f"watermarks-remover: {len(failures)} file(s) could not be cleaned:")
        for path, detail in failures:
            eprint(f"  {path}")
            eprint(f"    - {detail}")
            # clean_file.py --in-place takes its backup before it writes, so a
            # run that died mid-write leaves the sidecar behind. Name it rather
            # than delete it: this wrapper cannot tell its own leftover from a
            # .bak the developer already had, and deleting the wrong file
            # inside an already-failing hook is the worse outcome. The original
            # is intact either way — safe_write_bytes() replaces atomically or
            # not at all — so the sidecar is redundant, not load-bearing.
            bak = path.with_suffix(path.suffix + ".bak")
            if bak.exists():
                eprint(f"    - backup left behind: {bak}")
        eprint("Those files still carry whatever marks they had. Re-run the cleaner by hand:")
        eprint("  python3 service/scripts/clean_file.py <path> --in-place")

    # A file that could not be processed outranks the auto-fix signal, the same
    # precedence audit_dir.py applies: an incomplete run is the more important
    # thing for the developer (and CI) to see, and exit 1 would send them
    # looking for a diff to re-stage that does not exist.
    return EXIT_PARTIAL if failures else (1 if changed_paths else 0)


if __name__ == "__main__":
    raise SystemExit(main())
