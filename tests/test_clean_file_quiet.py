"""Tests for clean_file.py --quiet / --only-changed flag."""

from __future__ import annotations

import struct
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
CLEAN_FILE_PY = SCRIPTS / "clean_file.py"


def _minimal_clean_png() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    ihdr_chunk = (
        struct.pack(">I", len(ihdr))
        + b"IHDR"
        + ihdr
        + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF)
    )
    iend_chunk = (
        struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    )
    return b"\x89PNG\r\n\x1a\n" + ihdr_chunk + iend_chunk


def test_clean_file_quiet_suppresses_untouched_text(tmp_path: Path):
    clean_txt = tmp_path / "clean.txt"
    clean_txt.write_text("Hello clean world!", encoding="utf-8")

    # Standard run prints wrote line
    r_normal = subprocess.run(
        [sys.executable, str(CLEAN_FILE_PY), str(clean_txt), "--in-place"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "wrote" in r_normal.stderr
    assert "removed=0" in r_normal.stderr

    # Quiet run suppresses emission
    r_quiet = subprocess.run(
        [sys.executable, str(CLEAN_FILE_PY), str(clean_txt), "--in-place", "-q"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert r_quiet.stderr.strip() == ""
    assert r_quiet.returncode == 0

    # Long flag --only-changed also suppresses emission
    r_only_changed = subprocess.run(
        [sys.executable, str(CLEAN_FILE_PY), str(clean_txt), "--in-place", "--only-changed"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert r_only_changed.stderr.strip() == ""
    assert r_only_changed.returncode == 0


def test_clean_file_quiet_prints_when_watermark_removed(tmp_path: Path):
    marked_txt = tmp_path / "marked.txt"
    marked_txt.write_text("Hello\u200bWorld!", encoding="utf-8")

    r_quiet = subprocess.run(
        [sys.executable, str(CLEAN_FILE_PY), str(marked_txt), "--in-place", "-q"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "wrote" in r_quiet.stderr
    assert "removed=1" in r_quiet.stderr
    assert r_quiet.returncode == 0


def test_clean_file_quiet_suppresses_untouched_image(tmp_path: Path):
    clean_png = tmp_path / "clean.png"
    clean_png.write_bytes(_minimal_clean_png())

    r_quiet = subprocess.run(
        [sys.executable, str(CLEAN_FILE_PY), str(clean_png), "--in-place", "-q"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert r_quiet.stderr.strip() == ""
    assert r_quiet.returncode == 0


def test_clean_file_quiet_json_suppresses_untouched_text(tmp_path: Path):
    """--quiet --json emits nothing for an unchanged text file."""
    clean_txt = tmp_path / "clean.txt"
    clean_txt.write_text("Hello clean world!", encoding="utf-8")

    r_quiet_json = subprocess.run(
        [sys.executable, str(CLEAN_FILE_PY), str(clean_txt), "--in-place", "-q", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert r_quiet_json.returncode == 0
    assert r_quiet_json.stdout.strip() == ""
    assert r_quiet_json.stderr.strip() == ""


def test_clean_file_quiet_json_suppresses_untouched_image(tmp_path: Path):
    """--quiet --json emits nothing for an unchanged image file."""
    clean_png = tmp_path / "clean.png"
    clean_png.write_bytes(_minimal_clean_png())

    r_quiet_json = subprocess.run(
        [sys.executable, str(CLEAN_FILE_PY), str(clean_png), "--in-place", "-q", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert r_quiet_json.returncode == 0
    assert r_quiet_json.stdout.strip() == ""
    assert r_quiet_json.stderr.strip() == ""
