import struct
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _clean_png_bytes() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    ihdr_chunk = (
        struct.pack(">I", len(ihdr))
        + b"IHDR"
        + ihdr
        + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF)
    )
    # Include an AI text chunk so clean_image has something to strip
    tEXt_data = b"Description\x00Created with Midjourney"
    text_chunk = (
        struct.pack(">I", len(tEXt_data))
        + b"tEXt"
        + tEXt_data
        + struct.pack(">I", zlib.crc32(b"tEXt" + tEXt_data) & 0xFFFFFFFF)
    )
    iend_chunk = (
        struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    )
    return b"\x89PNG\r\n\x1a\n" + ihdr_chunk + text_chunk + iend_chunk


def test_clean_file_in_place_no_backup(tmp_path: Path):
    clean_file_py = SCRIPTS / "clean_file.py"
    target = tmp_path / "doc.txt"
    original_content = "Hello" + chr(0x200B) + "World!"
    target.write_text(original_content, encoding="utf-8")

    res = subprocess.run(
        [sys.executable, str(clean_file_py), str(target), "--in-place", "--no-backup"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    bak = tmp_path / "doc.txt.bak"
    assert not bak.exists()
    assert target.read_text(encoding="utf-8") == "HelloWorld!"


def test_clean_file_no_backup_without_in_place_refused(tmp_path: Path):
    clean_file_py = SCRIPTS / "clean_file.py"
    target = tmp_path / "doc.txt"
    target.write_text("Hello", encoding="utf-8")

    res = subprocess.run(
        [sys.executable, str(clean_file_py), str(target), "--no-backup"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 2
    assert "refusing --no-backup without --in-place" in res.stderr


def test_clean_text_in_place_no_backup(tmp_path: Path):
    clean_text_py = SCRIPTS / "clean_text.py"
    target = tmp_path / "text.txt"
    original_content = "Invisible" + chr(0x200C) + "Mark"
    target.write_text(original_content, encoding="utf-8")

    res = subprocess.run(
        [sys.executable, str(clean_text_py), str(target), "--in-place", "--no-backup"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    bak = tmp_path / "text.txt.bak"
    assert not bak.exists()
    assert target.read_text(encoding="utf-8") == "InvisibleMark"


def test_clean_image_in_place_no_backup(tmp_path: Path):
    clean_image_py = SCRIPTS / "clean_image.py"
    target = tmp_path / "sample.png"
    target.write_bytes(_clean_png_bytes())

    res = subprocess.run(
        [sys.executable, str(clean_image_py), str(target), "--in-place", "--no-backup"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    bak = tmp_path / "sample.png.bak"
    assert not bak.exists()
    assert target.exists()
    assert b"Midjourney" not in target.read_bytes()


def test_clean_file_container_in_place_no_backup(tmp_path: Path):
    clean_file_py = SCRIPTS / "clean_file.py"
    target = tmp_path / "page.html"
    content = '<!DOCTYPE html><html><head><meta name="generator" content="ChatGPT"></head><body>' + chr(0x200B) + 'test</body></html>'
    target.write_text(content, encoding="utf-8")

    res = subprocess.run(
        [sys.executable, str(clean_file_py), str(target), "--in-place", "--no-backup"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    bak = tmp_path / "page.html.bak"
    assert not bak.exists()
    cleaned = target.read_text(encoding="utf-8")
    assert "ChatGPT" not in cleaned
    assert chr(0x200B) not in cleaned
