import os
import stat
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import clean_staged


def _clean_png_bytes() -> bytes:
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


def _clean_jpeg_bytes() -> bytes:
    return (
        b"\xff\xd8\xff\xdb\x00\x43\x00"
        + b"\x00" * 64
        + b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xd9"
    )


def _clean_pdf_bytes() -> bytes:
    return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"


def _clean_mp4_bytes() -> bytes:
    return b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08free"


def test_bak_not_overwritten_on_second_in_place_run(tmp_path: Path):
    clean_file_py = SCRIPTS / "clean_file.py"
    target = tmp_path / "test.txt"
    original_content = "Hello" + chr(0x200B) + "World!"
    target.write_text(original_content, encoding="utf-8")

    # First in-place run: strips zero-width space, creates .bak with original
    subprocess.run(
        [sys.executable, str(clean_file_py), str(target), "--in-place"],
        check=True,
    )
    bak = tmp_path / "test.txt.bak"
    assert bak.exists()
    assert bak.read_text(encoding="utf-8") == original_content
    assert target.read_text(encoding="utf-8") == "HelloWorld!"

    # Second in-place run: should NOT overwrite .bak with "HelloWorld!"
    subprocess.run(
        [sys.executable, str(clean_file_py), str(target), "--in-place"],
        check=True,
    )
    assert bak.read_text(encoding="utf-8") == original_content


@pytest.mark.skipif(not hasattr(os, "fchmod"), reason="requires POSIX file mode support")
def test_clean_file_in_place_noop_preserves_executable_mode_and_skips_backup(tmp_path: Path):
    clean_file_py = SCRIPTS / "clean_file.py"
    target = tmp_path / "s.sh"
    content = b"#!/bin/sh\necho hi\n"
    target.write_bytes(content)
    target.chmod(0o755)
    original_inode = target.stat().st_ino

    proc = subprocess.run(
        [sys.executable, str(clean_file_py), str(target), "--in-place"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert stat.S_IMODE(target.stat().st_mode) == 0o755
    assert target.stat().st_ino == original_inode
    assert target.read_bytes() == content
    assert not target.with_suffix(".sh.bak").exists()
    assert "already clean" in proc.stderr.lower()


@pytest.mark.skipif(not hasattr(os, "fchmod"), reason="requires POSIX file mode support")
def test_clean_file_in_place_preserves_executable_mode_on_changes(tmp_path: Path):
    clean_file_py = SCRIPTS / "clean_file.py"
    target = tmp_path / "s.sh"
    target.write_text("#!/bin/sh\necho hi\u200b\n", encoding="utf-8")
    target.chmod(0o755)

    subprocess.run(
        [sys.executable, str(clean_file_py), str(target), "--in-place"],
        capture_output=True,
        text=True,
        check=True,
    )

    bak = target.with_suffix(".sh.bak")
    assert target.read_text(encoding="utf-8") == "#!/bin/sh\necho hi\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o755
    assert stat.S_IMODE(bak.stat().st_mode) == 0o755


def test_clean_file_in_place_noop_refuses_symlink(tmp_path: Path):
    clean_file_py = SCRIPTS / "clean_file.py"
    target = tmp_path / "target.sh"
    target.write_bytes(b"#!/bin/sh\necho hi\n")
    target.chmod(0o755)
    link = tmp_path / "link.sh"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    proc = subprocess.run(
        [sys.executable, str(clean_file_py), str(link), "--in-place"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "refusing to write through symlink" in proc.stderr.lower()
    assert target.read_bytes() == b"#!/bin/sh\necho hi\n"
    assert not link.with_suffix(".sh.bak").exists()


@pytest.mark.parametrize(
    ("filename", "data"),
    [
        ("clean.png", _clean_png_bytes()),
        ("clean.jpg", _clean_jpeg_bytes()),
        ("non_marker.jpg", b"\xff\xd8abc"),
        ("clean.pdf", _clean_pdf_bytes()),
        ("clean.mp4", _clean_mp4_bytes()),
    ],
)
def test_clean_file_binary_already_clean_reports_clean(tmp_path: Path, filename: str, data: bytes):
    clean_file_py = SCRIPTS / "clean_file.py"
    target = tmp_path / filename
    target.write_bytes(data)

    # CLI standard output
    proc = subprocess.run(
        [sys.executable, str(clean_file_py), str(target), "--in-place"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "already clean" in proc.stderr.lower() or "already clean" in proc.stdout.lower()

    # Pre-commit clean_staged hook
    status, _ = clean_staged._clean_one(target)
    assert status == "unchanged"
