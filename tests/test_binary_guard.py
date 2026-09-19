"""Tests for the binary-input guard on the text-only tools."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from common import guard_binary, looks_binary

DOCX_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    "<w:body><w:p><w:r><w:t>Table 1 holds the results.</w:t></w:r></w:p></w:body>"
    "</w:document>"
)


def make_docx(path: Path) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", DOCX_XML)
    return path


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


# --- looks_binary ----------------------------------------------------------


@pytest.mark.parametrize(
    "data,expected_fragment",
    [
        (b"PK\x03\x04rest", "ZIP"),
        (b"%PDF-1.7\n", "PDF"),
        (b"\x89PNG\r\n\x1a\nrest", "PNG"),
        (b"\xff\xd8\xff\xe0rest", "JPEG"),
        (b"\x7fELF\x02\x01", "ELF"),
        (b"SQLite format 3\x00", "SQLite"),
        (b"plain text\x00with a nul", "NUL"),
    ],
)
def test_flags_binary(data, expected_fragment):
    kind = looks_binary(data)
    assert kind is not None
    assert expected_fragment.lower() in kind.lower()


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"Just some prose.\n",
        b"# Markdown\n\n- bullet\n",
        "Accented prose: naïve café résumé\n".encode(),
        "Zero width​ and nbsp here\n".encode(),
        b"Latin-1 bytes: caf\xe9 na\xefve\n",  # not UTF-8, still text
        b"a\tb\r\nc\x0cd\x1b[0m\n",  # tabs, CRLF, form feed, ANSI escape
    ],
)
def test_allows_text(data):
    assert looks_binary(data) is None


def test_compressed_bytes_are_flagged(tmp_path):
    data = make_docx(tmp_path / "x.docx").read_bytes()
    assert looks_binary(data) is not None


def test_guard_binary_can_be_overridden():
    guard_binary(b"PK\x03\x04", "x.docx", allow_binary=True)  # must not raise
    with pytest.raises(SystemExit) as exc:
        guard_binary(b"PK\x03\x04", "x.docx")
    assert exc.value.code == 2


# --- CLI behaviour ---------------------------------------------------------


def test_inspect_text_refuses_docx(tmp_path):
    docx = make_docx(tmp_path / "doc.docx")
    r = run("inspect_text.py", str(docx))
    assert r.returncode == 2
    assert "looks like" in r.stderr
    assert "inspect_file.py" in r.stderr
    assert "Suspicious:" not in r.stdout


def test_inspect_text_force_text_still_works(tmp_path):
    docx = make_docx(tmp_path / "doc.docx")
    r = run("inspect_text.py", str(docx), "--force-text")
    assert r.returncode in (0, 1)
    assert "Length:" in r.stdout


def test_clean_text_refuses_docx_and_writes_nothing(tmp_path):
    docx = make_docx(tmp_path / "doc.docx")
    before = docx.read_bytes()
    out = tmp_path / "doc.cleaned.docx"
    r = run("clean_text.py", str(docx), "-o", str(out))
    assert r.returncode == 2
    assert not out.exists()
    assert docx.read_bytes() == before


def test_clean_text_in_place_leaves_docx_intact(tmp_path):
    docx = make_docx(tmp_path / "doc.docx")
    before = docx.read_bytes()
    r = run("clean_text.py", str(docx), "--in-place")
    assert r.returncode == 2
    assert docx.read_bytes() == before
    assert not (tmp_path / "doc.docx.bak").exists()


def test_clean_file_still_routes_docx_to_container(tmp_path):
    docx = make_docx(tmp_path / "doc.docx")
    out = tmp_path / "out.docx"
    r = run("clean_file.py", str(docx), "-o", str(out), "--json")
    assert r.returncode == 0, r.stderr
    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        assert zf.testzip() is None
        assert "word/document.xml" in zf.namelist()


def test_clean_file_refuses_unknown_binary(tmp_path):
    blob = tmp_path / "mystery.bin"
    blob.write_bytes(b"\x00\x01\x02\x03" * 64)
    out = tmp_path / "out.bin"
    r = run("clean_file.py", str(blob), "-o", str(out))
    assert r.returncode == 2
    assert not out.exists()


def test_clean_file_in_place_refuses_before_taking_a_backup(tmp_path):
    """The refusal must land before backup_path(), not after.

    Sniffing after the backup left a .bak sidecar for a file the run never
    touches, which is exactly what clean_text.py avoids.
    """
    blob = tmp_path / "mystery.bin"
    blob.write_bytes(b"\x00\x01\x02\x03" * 64)
    before = blob.read_bytes()
    r = run("clean_file.py", str(blob), "--in-place")
    assert r.returncode == 2
    assert blob.read_bytes() == before
    assert not (tmp_path / "mystery.bin.bak").exists()
    assert list(tmp_path.iterdir()) == [blob]


def test_clean_file_in_place_as_text_on_docx_leaves_no_backup(tmp_path):
    """--as text bypasses classify(), so the guard is the only thing left."""
    docx = make_docx(tmp_path / "doc.docx")
    before = docx.read_bytes()
    r = run("clean_file.py", str(docx), "--in-place", "--as", "text")
    assert r.returncode == 2
    assert docx.read_bytes() == before
    assert not (tmp_path / "doc.docx.bak").exists()


def test_clean_file_auto_refuses_unknown_text_like_bytes(tmp_path):
    """Extension-less bytes with no magic classify as "unknown" and are
    refused in auto mode — even when they look like plain text, because
    "text" is no longer the catch-all for unrecognized files."""
    blob = tmp_path / "no_extension"
    blob.write_text("just plain text, no extension, no magic\n", encoding="utf-8")
    before = blob.read_bytes()
    r = run("clean_file.py", str(blob), "--in-place")
    assert r.returncode == 2
    assert blob.read_bytes() == before
    assert not (tmp_path / "no_extension.bak").exists()


def test_clean_file_as_text_opt_in_cleans_unknown(tmp_path):
    """--as text is the explicit opt-in that turns an unknown file into a
    text clean."""
    blob = tmp_path / "no_extension"
    blob.write_text("Hidden\u200bmark here.\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    r = run("clean_file.py", str(blob), "-o", str(out), "--as", "text")
    assert r.returncode == 0, r.stderr
    assert out.read_text(encoding="utf-8") == "Hiddenmark here.\n"


def test_clean_file_force_text_opt_in_on_unknown_binary(tmp_path):
    """--force-text also opts in, bypassing both the unknown refusal and
    the binary guard (explicit override)."""
    blob = tmp_path / "mystery.bin"
    blob.write_bytes(b"\x00\x01\x02\x03" * 64)
    out = tmp_path / "out.bin"
    r = run("clean_file.py", str(blob), "-o", str(out), "--force-text")
    assert r.returncode == 0, r.stderr
    assert out.exists()


def test_inspect_file_json_reports_unknown_kind(tmp_path):
    blob = tmp_path / "no_extension"
    blob.write_text("no magic, no extension\n", encoding="utf-8")
    r = run("inspect_file.py", str(blob), "--json")
    assert r.returncode == 0
    import json

    payload = json.loads(r.stdout)
    assert payload["kind"] == "unknown"
    assert "note" in payload


def test_router_advice_is_not_circular(tmp_path):
    """clean_file.py refuses unknown bytes with router advice (never a pointer
    back at itself); inspect_file.py reports the file as unknown instead."""
    blob = tmp_path / "mystery.bin"
    blob.write_bytes(b"\x00\x01\x02\x03" * 64)
    r = run("clean_file.py", str(blob))
    assert r.returncode == 2
    assert "no supported text, image, audio, video or container format" in r.stderr
    assert "Use inspect_file.py / clean_file.py" not in r.stderr
    assert "--force-text" in r.stderr
    r = run("inspect_file.py", str(blob))
    assert r.returncode == 0
    assert "Kind: unknown" in r.stdout
    assert "--as text|image|container" in r.stdout


def test_text_only_scripts_keep_the_pointer_to_the_routers(tmp_path):
    docx = make_docx(tmp_path / "doc.docx")
    r = run("clean_text.py", str(docx))
    assert r.returncode == 2
    assert "Use inspect_file.py / clean_file.py" in r.stderr


def test_text_files_are_unaffected(tmp_path):
    src = tmp_path / "note.txt"
    src.write_text("Hidden​mark here.\n", encoding="utf-8")
    out = tmp_path / "note.cleaned.txt"
    r = run("clean_text.py", str(src), "-o", str(out))
    assert r.returncode == 0, r.stderr
    assert out.read_text(encoding="utf-8") == "Hiddenmark here.\n"


def test_stdin_binary_is_refused():
    docx = io.BytesIO()
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("word/document.xml", DOCX_XML)
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "inspect_text.py")],
        input=docx.getvalue(),
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert r.returncode == 2
    assert b"looks like" in r.stderr


PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 32
JPEG_HEADER = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"A" * 64


@pytest.mark.parametrize("data,label", [(PNG_HEADER, "PNG"), (JPEG_HEADER, "JPEG")])
@pytest.mark.parametrize("io_encoding", [None, "cp1252", "latin-1"])
def test_stdin_non_ascii_magic_is_refused_whatever_the_codec(data, label, io_encoding):
    """The ZIP test alone could not catch this: 'PK' is ASCII.

    PNG's 0x89 and JPEG's 0xff only survive to the sniff if stdin is read as
    bytes. Decoding first makes detection depend on the console codec.
    """
    env = dict(os.environ)
    if io_encoding is None:
        env.pop("PYTHONIOENCODING", None)
    else:
        env["PYTHONIOENCODING"] = io_encoding
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "inspect_text.py")],
        input=data,
        capture_output=True,
        timeout=60,
        env=env,
        check=False,
    )
    assert r.returncode == 2, (label, io_encoding, r.stderr)
    assert label.encode() in r.stderr, (label, io_encoding, r.stderr)


def test_stdin_text_still_flows_through():
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "clean_text.py")],
        input="plain​text\n".encode(),
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.replace(b"\r\n", b"\n") == b"plaintext\n"
