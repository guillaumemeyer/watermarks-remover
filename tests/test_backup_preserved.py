"""A second --in-place run preserves the original .bak (#172).

The pre-commit auto-fix flow (clean → commit blocked → re-stage → clean again)
used to overwrite the first run's backup with the first run's output,
destroying the only pristine copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import common


def test_first_run_creates_backup_second_preserves_it(tmp_path):
    original = tmp_path / "notes.md"
    original.write_text("Hello​World watermarked", encoding="utf-8")
    pristine = original.read_bytes()

    bak, created = common.backup_path(original)
    assert created is True
    assert bak.read_bytes() == pristine

    # Simulate run 1's output at the live path, then run 2's backup call.
    original.write_text("HelloWorld watermarked", encoding="utf-8")
    bak2, created2 = common.backup_path(original)
    assert bak2 == bak
    assert created2 is False
    # The .bak still holds the PRISTINE original, not run 1's output.
    assert bak.read_bytes() == pristine


def test_no_existing_bak_still_created(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"data")
    bak, created = common.backup_path(f)
    assert created is True
    assert bak.read_bytes() == b"data"
