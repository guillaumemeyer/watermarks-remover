"""Tests for the pre-commit hook wrappers (check_staged.py / clean_staged.py)."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest


def _png_chunk(typ: bytes, data: bytes) -> bytes:
    """Build a PNG chunk with length, type, payload, and CRC32."""
    body = typ + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _minimal_png() -> bytes:
    """A real, clean 1x1 PNG (stdlib only, deterministic)."""
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def _minimal_pdf_with_xmp() -> bytes:
    """A minimal PDF with an XMP packet that gets blanked with spaces (same file length)."""
    xmp = (
        b"<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>"
        b"<x:xmpmeta xmlns:x='adobe:ns:meta/'>"
        b"<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>"
        b"<rdf:Description>"
        b"<digitalSourceType>trainedAlgorithmicMedia</digitalSourceType>"
        b"</rdf:Description></rdf:RDF></x:xmpmeta>"
        b"<?xpacket end='w'?>"
    )
    return b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n" + xmp + b"\n%%EOF\n"


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import check_staged
import clean_staged


def _watermarked_text() -> str:
    """Return a string containing zero-width space Layer A watermark."""
    return "Hello" + chr(0x200B) + "World!"


def test_check_staged_clean_file_exits_0(tmp_path, monkeypatch, capsys):
    """Clean text file must pass the check hook with exit 0."""
    f = tmp_path / "clean.txt"
    f.write_text("Nothing to see here.", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(f)])
    assert check_staged.main() == 0


def test_check_staged_marked_file_exits_1(tmp_path, monkeypatch, capsys):
    """Marked text file must fail the check hook with exit 1."""
    f = tmp_path / "marked.txt"
    f.write_text(_watermarked_text(), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(f)])
    assert check_staged.main() == 1
    err = capsys.readouterr().err
    assert str(f) in err
    assert "layer-a" in err


def test_check_staged_multiple_files_one_marked(tmp_path, monkeypatch, capsys):
    """A batch containing clean and marked files must report the marked file and exit 1."""
    clean = tmp_path / "clean.txt"
    clean.write_text("plain text", encoding="utf-8")
    marked = tmp_path / "marked.txt"
    marked.write_text(_watermarked_text(), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(clean), str(marked)])
    assert check_staged.main() == 1
    err = capsys.readouterr().err
    assert str(marked) in err
    assert str(clean) not in err


def test_check_staged_unknown_format_skipped(tmp_path, monkeypatch):
    """Unrecognized binary format is skipped by check_staged."""
    f = tmp_path / "data.bin"
    f.write_bytes(b"\x00\x01\x02\xff\xfe no known magic bytes here")
    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(f)])
    assert check_staged.main() == 0


def test_check_staged_missing_path_exits_2(tmp_path, monkeypatch):
    """Missing path produces an argument error exit code 2."""
    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(tmp_path / "nope.txt")])
    assert check_staged.main() == 2


def test_clean_staged_marked_file_cleans_and_exits_1(tmp_path, monkeypatch, capsys):
    """Marked file is cleaned in place and clean_staged exits 1 requesting re-stage."""
    f = tmp_path / "marked.txt"
    f.write_text(_watermarked_text(), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 1
    assert f.read_text(encoding="utf-8") == "HelloWorld!"
    err = capsys.readouterr().err
    assert str(f) in err


def test_clean_staged_already_clean_file_exits_0_unchanged(tmp_path, monkeypatch):
    """Already clean text file exits 0 without modifying file content."""
    f = tmp_path / "clean.txt"
    original = "Nothing to see here."
    f.write_text(original, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 0
    assert f.read_text(encoding="utf-8") == original


def test_clean_staged_clean_image_report_exits_0(tmp_path, monkeypatch, capsys):
    """A clean image report with filler actions exits 0 without demanding a re-stage."""
    # Issue #173: every strip_* appends a "nothing was removed" filler action,
    # so a non-empty actions list used to read as "changed" even when the file
    # on disk was byte-identical. A clean image must not ask for a re-stage.
    f = _staged_file(tmp_path)
    report = {
        "kind": "image",
        "actions": ["no PNG metadata chunks removed (already clean or none matched)"],
        "bytes_in": 69,
        "bytes_out": 69,
        "still_has_c2pa": False,
        "still_has_ai_metadata": False,
    }
    _fake_clean_file(monkeypatch, 0, stdout=json.dumps(report))
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 0
    assert "cleaned 1 file(s)" not in capsys.readouterr().err


def test_clean_staged_byte_identical_with_non_filler_action_exits_0(tmp_path, monkeypatch, capsys):
    """A byte-identical file on disk exits 0 even if the cleaner report carries a non-filler action."""
    # When before_digest == after_digest on disk, digest comparison is
    # authoritative, preventing any action heuristic misread from forcing exit 1.
    f = _staged_file(tmp_path)
    report = {
        "kind": "image",
        "format": "jpeg",
        "actions": ["preserved entropy-coded scan"],
        "bytes_in": 1024,
        "bytes_out": 1024,
        "still_has_c2pa": False,
        "still_has_ai_metadata": False,
    }
    _fake_clean_file(monkeypatch, 0, stdout=json.dumps(report))
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 0
    assert "cleaned 1 file(s)" not in capsys.readouterr().err


def test_clean_staged_modified_image_report_exits_1(tmp_path, monkeypatch, capsys):
    """A report indicating removed image metadata exits 1 requesting re-stage."""
    # The other side of the byte comparison: a strip that really changed the
    # file still asks the developer to re-stage it.
    f = _staged_file(tmp_path)
    report = {
        "kind": "image",
        "actions": ["strip PNG c2pa chunk (jumb, 1234 bytes)"],
        "bytes_in": 1400,
        "bytes_out": 181,
        "still_has_c2pa": False,
        "still_has_ai_metadata": False,
    }
    _fake_clean_file(monkeypatch, 0, stdout=json.dumps(report), mutate_file=f)
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 1
    assert "cleaned 1 file(s)" in capsys.readouterr().err


def test_clean_staged_clean_image_end_to_end_exits_0(tmp_path, monkeypatch):
    """Subprocess clean on a real clean PNG exits 0 without demanding a re-stage."""
    # No mocking: drive the real clean_file.py subprocess on a byte-identical
    # clean PNG, mirroring the exact repro in the issue.
    f = tmp_path / "clean.png"
    f.write_bytes(_minimal_png())
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 0
    assert f.read_bytes() == _minimal_png()


def test_clean_staged_same_length_pdf_report_exits_1(tmp_path, monkeypatch, capsys):
    """PDF whose XMP is blanked with spaces retains length but changes content and exits 1."""
    f = _staged_file(tmp_path)
    report = {
        "kind": "container",
        "format": "pdf",
        "actions": [
            "blanked XMP xpacket x1 (degraded; byte offsets preserved)",
            "warning: pure-stdlib PDF strip is best-effort; prefer exiftool",
        ],
        "bytes_in": 500,
        "bytes_out": 500,
        "still_has_c2pa": False,
        "still_has_ai_metadata": False,
    }
    _fake_clean_file(monkeypatch, 0, stdout=json.dumps(report), mutate_file=f)
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 1
    assert "cleaned 1 file(s)" in capsys.readouterr().err


def test_clean_staged_same_length_pdf_end_to_end_exits_1(tmp_path, monkeypatch, capsys):
    """Real subprocess clean on a PDF with XMP overwrites XMP with spaces, exits 1, same length."""
    f = tmp_path / "marked.pdf"
    pdf_bytes = _minimal_pdf_with_xmp()
    f.write_bytes(pdf_bytes)
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 1
    cleaned = f.read_bytes()
    assert len(cleaned) == len(pdf_bytes)
    assert cleaned != pdf_bytes
    assert b"trainedAlgorithmicMedia" not in cleaned
    assert "cleaned 1 file(s)" in capsys.readouterr().err


def test_clean_staged_unknown_format_skipped(tmp_path, monkeypatch):
    """Unrecognized binary format is skipped by clean_staged."""
    f = tmp_path / "data.bin"
    original = b"\x00\x01\x02\xff\xfe no known magic bytes here"
    f.write_bytes(original)
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 0
    assert f.read_bytes() == original


def test_clean_staged_oversized_file_skipped_without_reading(tmp_path, monkeypatch, capsys):
    """An oversized staged file is skipped without loading its entire contents into memory."""
    f = tmp_path / "huge.bin"
    f.touch()
    real_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        st = real_stat(self, *args, **kwargs)
        if self == f:
            import os

            return os.stat_result(
                (
                    st.st_mode,
                    st.st_ino,
                    st.st_dev,
                    st.st_nlink,
                    st.st_uid,
                    st.st_gid,
                    clean_staged.MAX_INPUT_BYTES + 1,
                    st.st_atime,
                    st.st_mtime,
                    st.st_ctime,
                )
            )
        return st

    def forbidden_open(*args, **kwargs):
        raise AssertionError("open() must not be called on oversized files")

    monkeypatch.setattr(Path, "stat", fake_stat)
    monkeypatch.setattr(Path, "open", forbidden_open)
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 0
    err = capsys.readouterr().err
    assert f"skipping {f}: larger than {clean_staged.MAX_INPUT_BYTES} bytes" in err


# ---------------------------------------------------------------------------
# A cleaner that could not run must not pass as "already clean" (issue #159).
# These pin exit 3 == common.EXIT_PARTIAL: an incomplete run outranks the
# auto-fix 1, which would send the developer looking for a diff to re-stage.
# ---------------------------------------------------------------------------


CRASH_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "clean_file.py", line 115, in main\n'
    "    safe_write_text(dest, cleaned)\n"
    "OSError: refusing to write through symlink: /repo/staged.txt\n"
)


def _staged_file(tmp_path: Path) -> Path:
    """Create a temporary staged file fixture."""
    f = tmp_path / "staged.txt"
    f.write_text("body", encoding="utf-8")
    return f


def _fake_clean_file(
    monkeypatch,
    returncode: int,
    stdout: str = "",
    stderr: str = "",
    mutate_file: Path | None = None,
) -> None:
    """Pin the clean_file.py subprocess to one outcome without running it."""

    def fake_run(cmd, *a, **k):
        if mutate_file is not None:
            mutate_file.write_bytes(b"modified by cleaner subprocess")
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(clean_staged.subprocess, "run", fake_run)


def test_clean_staged_crashed_cleaner_exits_partial(tmp_path, monkeypatch, capsys):
    """An uncaught exception in clean_file.py reports the error and exits EXIT_PARTIAL (3)."""
    # The bug: an uncaught exception in clean_file.py left stdout empty, which
    # counted as "skipped", so the hook exited 0 and the file went in uncleaned.
    f = _staged_file(tmp_path)
    _fake_clean_file(monkeypatch, 1, stderr=CRASH_TRACEBACK)
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 3
    err = capsys.readouterr().err
    assert "could not be cleaned" in err
    assert str(f) in err
    # The cause is reported, not the stack of frames that produced it.
    assert "OSError: refusing to write through symlink" in err
    assert "Traceback (most recent call last)" not in err


def test_clean_staged_killed_cleaner_reports_exit_status(tmp_path, monkeypatch, capsys):
    """A cleaner killed by a signal reports its exit code and exits EXIT_PARTIAL (3)."""
    # A child killed by a signal (negative returncode on POSIX) leaves no
    # stderr to quote, so the exit status itself has to carry the report.
    f = _staged_file(tmp_path)
    _fake_clean_file(monkeypatch, -9)
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 3
    assert "wrote no report (exit -9)" in capsys.readouterr().err


def test_clean_staged_empty_output_exits_partial(tmp_path, monkeypatch, capsys):
    """A cleaner producing empty stdout exits EXIT_PARTIAL (3)."""
    f = _staged_file(tmp_path)
    _fake_clean_file(monkeypatch, 0, stdout="   \n")
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 3
    assert "wrote no report" in capsys.readouterr().err


def test_clean_staged_malformed_json_exits_partial(tmp_path, monkeypatch, capsys):
    """A cleaner producing malformed JSON stdout exits EXIT_PARTIAL (3)."""
    f = _staged_file(tmp_path)
    _fake_clean_file(monkeypatch, 0, stdout="{not json")
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 3
    assert "unparsable report" in capsys.readouterr().err


def test_clean_staged_skip_code_still_exits_0(tmp_path, monkeypatch):
    """A cleaner exiting 2 (unrecognized format / oversize) is treated as a skipped file."""
    # Regression guard on the deliberate skip: exit 2 from clean_file.py means
    # an unrecognized format or an oversized input, not a failure to clean.
    f = _staged_file(tmp_path)
    _fake_clean_file(monkeypatch, 2, stderr="refusing to classify: unrecognized format")
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 0


def test_clean_staged_residual_signals_are_not_a_failure(tmp_path, monkeypatch, capsys):
    """A clean that left residual signals is treated as a changed file (exit 1), not a failure."""
    # clean_file.py exits 1 for a *successful* clean that left residual signals
    # (tests/test_json_exit_code.py). Judging the run by its exit code instead
    # of its report would turn every one of those into a hook failure.
    f = _staged_file(tmp_path)
    report = {"kind": "image", "actions": ["strip xmp"], "still_has_c2pa": True}
    _fake_clean_file(monkeypatch, 1, stdout=json.dumps(report), mutate_file=f)
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 1
    err = capsys.readouterr().err
    assert "cleaned 1 file(s) in place" in err
    assert "could not be cleaned" not in err


def test_clean_staged_changed_fallback_helper():
    """_changed() evaluates reports directly when before/after digests cannot be computed."""
    assert clean_staged._changed({"stats": {"removed_count": 2, "replaced_count": 0}})
    assert clean_staged._changed({"stats": {"removed_count": 0, "replaced_count": 1}})
    assert not clean_staged._changed({"stats": {"removed_count": 0, "replaced_count": 0}})
    assert clean_staged._changed({"bytes_in": 1200, "bytes_out": 200})
    assert clean_staged._changed({"actions": ["strip PNG c2pa chunk (jumb)"]})
    assert clean_staged._changed({"actions": ["blanked XMP xpacket x1"]})
    assert not clean_staged._changed(
        {"actions": ["no PNG metadata chunks removed (already clean or none matched)"]}
    )
    assert not clean_staged._changed({"actions": ["warning: exiftool failed"]})
    assert not clean_staged._changed({"actions": ["deep image pass not needed"]})
    assert not clean_staged._changed({"actions": ["kept 4 bytes of truncated tail"]})


def test_clean_staged_failure_outranks_a_successful_clean(tmp_path, monkeypatch, capsys):
    """In a batch with both cleaned and failed files, EXIT_PARTIAL (3) outranks exit 1."""
    # Mixed batch: both outcomes are reported, and the incomplete-run code wins.
    marked = tmp_path / "marked.txt"
    marked.write_text(_watermarked_text(), encoding="utf-8")
    broken = tmp_path / "broken.txt"
    broken.write_text(_watermarked_text(), encoding="utf-8")

    real_run = clean_staged.subprocess.run

    def fake_run(cmd, *a, **k):
        if str(broken) in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=CRASH_TRACEBACK)
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(clean_staged.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(marked), str(broken)])
    assert clean_staged.main() == 3
    err = capsys.readouterr().err
    assert "cleaned 1 file(s) in place" in err
    assert "could not be cleaned" in err
    assert marked.read_text(encoding="utf-8") == "HelloWorld!"


def _make_symlink(dest: Path, target: Path) -> None:
    """Create a symlink, skipping where the platform denies the privilege."""
    try:
        dest.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


def test_clean_staged_symlinked_path_exits_partial_end_to_end(tmp_path, monkeypatch, capsys):
    """End-to-end symlink write refusal leaves backup and exits EXIT_PARTIAL (3)."""
    # No mocking: common.py refuses to write through a symlink, clean_file.py
    # does not handle that OSError, and the hook used to swallow it as a skip.
    target = tmp_path / "real" / "target.txt"
    target.parent.mkdir()
    target.write_text(_watermarked_text(), encoding="utf-8")
    link = tmp_path / "link.txt"
    _make_symlink(link, target)

    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(link)])
    assert clean_staged.main() == 3
    # The file really is still marked; the hook must not have implied otherwise.
    assert target.read_text(encoding="utf-8") == _watermarked_text()
    err = capsys.readouterr().err
    assert "could not be cleaned" in err
    assert "refusing to write through symlink" in err
    # clean_file.py backs up before it writes, so a failed run leaves a sidecar.
    # The report names it rather than deleting a file this wrapper did not make.
    assert (tmp_path / "link.txt.bak").exists()
    assert "backup left behind" in err


def test_pre_commit_hooks_manifest_defines_both_hooks():
    """Verify that .pre-commit-hooks.yaml declares watermarks-remover-check and clean."""
    # No PyYAML in this project's stdlib-only test deps (requirements-dev.txt) —
    # check the manifest's shape textually rather than adding a parser dependency.
    text = (ROOT / ".pre-commit-hooks.yaml").read_text(encoding="utf-8")
    assert "id: watermarks-remover-check" in text
    assert "id: watermarks-remover-clean" in text
    assert text.count("entry: python3 service/scripts/") == 2
    assert text.count("language: system") == 2
