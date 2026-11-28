"""Regression tests for ``--in-place`` dropping the executable bit (#436).

``safe_write_bytes`` wrote to a ``mkstemp`` file and applied the umask-default
mode before ``os.replace``, so a 0755 script came back 0644 - even when nothing
was removed. ``backup_path`` had the same problem on the ``.bak`` side: it
created the copy with the umask default, so the pristine copy lost the bits
too.

The POSIX assertions are skipped on Windows, which has no ``fchmod`` and no
POSIX mode bits; ``_replacement_mode`` and the ``fchmod`` wiring are tested on
every platform.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import common as common_module
from common import _default_file_mode, _replacement_mode, safe_write_bytes

posix_only = pytest.mark.skipif(not hasattr(os, "fchmod"), reason="POSIX mode bits only")


def test_replacement_mode_uses_the_destination_bits(tmp_path: Path):
    target = tmp_path / "kept.txt"
    target.write_bytes(b"x")
    assert _replacement_mode(target) == stat.S_IMODE(target.stat().st_mode)


def test_replacement_mode_falls_back_for_a_missing_destination(tmp_path: Path):
    assert _replacement_mode(tmp_path / "absent.txt") == _default_file_mode()


def test_safe_write_bytes_applies_the_replacement_mode(tmp_path: Path, monkeypatch):
    # Windows has no os.fchmod, so the POSIX branch is unreachable there; a
    # stub proves which mode the branch hands to the kernel.
    applied: list[int] = []
    monkeypatch.setattr(os, "fchmod", lambda fd, mode: applied.append(mode), raising=False)
    monkeypatch.setattr(common_module, "_replacement_mode", lambda dest: 0o640)

    target = tmp_path / "script.sh"
    target.write_bytes(b"old")
    safe_write_bytes(target, b"new")

    assert target.read_bytes() == b"new"
    assert applied == [0o640]


@posix_only
def test_replacement_mode_reads_a_real_mode(tmp_path: Path):
    target = tmp_path / "s.sh"
    target.write_bytes(b"#!/bin/sh\n")
    target.chmod(0o751)
    assert _replacement_mode(target) == 0o751


@posix_only
def test_safe_write_bytes_keeps_the_existing_mode(tmp_path: Path):
    target = tmp_path / "run.sh"
    target.write_bytes(b"old")
    target.chmod(0o755)
    safe_write_bytes(target, b"new")
    assert target.read_bytes() == b"new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


@posix_only
def test_safe_write_bytes_uses_the_umask_default_for_a_new_file(tmp_path: Path):
    target = tmp_path / "fresh.txt"
    safe_write_bytes(target, b"new")
    assert stat.S_IMODE(target.stat().st_mode) == _default_file_mode()


@posix_only
def test_in_place_keeps_the_executable_bit_on_both_files(tmp_path: Path):
    script = tmp_path / "s.sh"
    script.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    script.chmod(0o755)

    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPTS_DIR / "clean_file.py"), str(script), "--in-place"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr

    assert stat.S_IMODE(script.stat().st_mode) == 0o755
    backup = script.with_suffix(".sh.bak")
    assert backup.is_file()
    assert stat.S_IMODE(backup.stat().st_mode) == 0o755


@posix_only
def test_in_place_keeps_a_restrictive_mode(tmp_path: Path):
    # 0700 is the case the umask default can never reproduce: 0o666 & ~umask
    # only ever clears bits, so a private script always came back readable.
    script = tmp_path / "private.sh"
    script.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    script.chmod(0o700)

    subprocess.run(
        [sys.executable, "-I", str(SCRIPTS_DIR / "clean_file.py"), str(script), "--in-place"],
        check=True,
        capture_output=True,
    )
    assert stat.S_IMODE(script.stat().st_mode) == 0o700
