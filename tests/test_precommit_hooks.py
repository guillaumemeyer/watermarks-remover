"""Tests for the pre-commit hook wrappers (check_staged.py / clean_staged.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import check_staged
import clean_staged


def _watermarked_text() -> str:
    return "Hello" + chr(0x200B) + "World!"


def test_check_staged_clean_file_exits_0(tmp_path, monkeypatch, capsys):
    f = tmp_path / "clean.txt"
    f.write_text("Nothing to see here.", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(f)])
    assert check_staged.main() == 0


def test_check_staged_marked_file_exits_1(tmp_path, monkeypatch, capsys):
    f = tmp_path / "marked.txt"
    f.write_text(_watermarked_text(), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(f)])
    assert check_staged.main() == 1
    err = capsys.readouterr().err
    assert str(f) in err
    assert "layer-a" in err


def test_check_staged_multiple_files_one_marked(tmp_path, monkeypatch, capsys):
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
    f = tmp_path / "data.bin"
    f.write_bytes(b"\x00\x01\x02\xff\xfe no known magic bytes here")
    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(f)])
    assert check_staged.main() == 0


def test_check_staged_missing_path_exits_2(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(tmp_path / "nope.txt")])
    assert check_staged.main() == 2


def test_clean_staged_marked_file_cleans_and_exits_1(tmp_path, monkeypatch, capsys):
    f = tmp_path / "marked.txt"
    f.write_text(_watermarked_text(), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 1
    assert f.read_text(encoding="utf-8") == "HelloWorld!"
    err = capsys.readouterr().err
    assert str(f) in err


def test_clean_staged_already_clean_file_exits_0_unchanged(tmp_path, monkeypatch):
    f = tmp_path / "clean.txt"
    original = "Nothing to see here."
    f.write_text(original, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 0
    assert f.read_text(encoding="utf-8") == original


def test_clean_staged_unknown_format_skipped(tmp_path, monkeypatch):
    f = tmp_path / "data.bin"
    original = b"\x00\x01\x02\xff\xfe no known magic bytes here"
    f.write_bytes(original)
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 0
    assert f.read_bytes() == original


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
    f = tmp_path / "staged.txt"
    f.write_text("body", encoding="utf-8")
    return f


def _fake_clean_file(monkeypatch, returncode: int, stdout: str = "", stderr: str = "") -> None:
    """Pin the clean_file.py subprocess to one outcome without running it."""
    monkeypatch.setattr(
        clean_staged.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], returncode, stdout=stdout, stderr=stderr),
    )


def test_clean_staged_crashed_cleaner_exits_partial(tmp_path, monkeypatch, capsys):
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
    # A child killed by a signal (negative returncode on POSIX) leaves no
    # stderr to quote, so the exit status itself has to carry the report.
    f = _staged_file(tmp_path)
    _fake_clean_file(monkeypatch, -9)
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 3
    assert "wrote no report (exit -9)" in capsys.readouterr().err


def test_clean_staged_empty_output_exits_partial(tmp_path, monkeypatch, capsys):
    f = _staged_file(tmp_path)
    _fake_clean_file(monkeypatch, 0, stdout="   \n")
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 3
    assert "wrote no report" in capsys.readouterr().err


def test_clean_staged_malformed_json_exits_partial(tmp_path, monkeypatch, capsys):
    f = _staged_file(tmp_path)
    _fake_clean_file(monkeypatch, 0, stdout="{not json")
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 3
    assert "unparsable report" in capsys.readouterr().err


def test_clean_staged_skip_code_still_exits_0(tmp_path, monkeypatch):
    # Regression guard on the deliberate skip: exit 2 from clean_file.py means
    # an unrecognized format or an oversized input, not a failure to clean.
    f = _staged_file(tmp_path)
    _fake_clean_file(monkeypatch, 2, stderr="refusing to classify: unrecognized format")
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 0


def test_clean_staged_residual_signals_are_not_a_failure(tmp_path, monkeypatch, capsys):
    # clean_file.py exits 1 for a *successful* clean that left residual signals
    # (tests/test_json_exit_code.py). Judging the run by its exit code instead
    # of its report would turn every one of those into a hook failure.
    f = _staged_file(tmp_path)
    report = {"kind": "image", "actions": ["strip xmp"], "still_has_c2pa": True}
    _fake_clean_file(monkeypatch, 1, stdout=json.dumps(report))
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 1
    err = capsys.readouterr().err
    assert "cleaned 1 file(s) in place" in err
    assert "could not be cleaned" not in err


def test_clean_staged_failure_outranks_a_successful_clean(tmp_path, monkeypatch, capsys):
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
    # No PyYAML in this project's stdlib-only test deps (requirements-dev.txt) —
    # check the manifest's shape textually rather than adding a parser dependency.
    text = (ROOT / ".pre-commit-hooks.yaml").read_text(encoding="utf-8")
    assert "id: watermarks-remover-check" in text
    assert "id: watermarks-remover-clean" in text
    assert text.count("entry: python3 service/scripts/") == 2
    assert text.count("language: system") == 2


def test_changed_recognizes_noop_actions_as_unchanged():
    """The exact no-op action strings every strip/clean emitter produces must
    read as unchanged — before, any non-empty actions list was 'changed' and
    the clean hook failed forever on non-text files (#173)."""
    noop_actions = [
        "no PNG metadata chunks removed (already clean or none matched)",
        "no GIF metadata blocks removed (already clean or none matched)",
        "no TIFF metadata tags removed (already clean or none matched)",
        "no WebP metadata chunks removed (already clean or none matched)",
        "no AVIF metadata boxes removed (already clean or none matched)",
        "no MP4 metadata boxes removed (already clean or none matched)",
        "no ID3v2 tag removed (no AI/C2PA markers found)",
        "no ID3v2 frames removed (already clean or none matched)",
        "no WAV metadata chunks removed (already clean or none matched)",
        "no AI frontmatter keys or embedded data URIs removed",
        "no HTML AI meta removed",
        "no SVG metadata removed",
        "no DOCX metadata parts removed",
        "no ODT metadata removed",
        "no OPF metadata removed",
    ]
    for action in noop_actions:
        assert clean_staged._changed({"actions": [action]}) is False, action


def test_changed_still_fires_on_real_actions():
    assert clean_staged._changed({"actions": ["removed 3 tEXt chunks"]}) is True
    assert clean_staged._changed({"actions": ["drop ID3v2.3 tag (210 bytes)"]}) is True
    # A mix of noop + real is a change.
    assert (
        clean_staged._changed(
            {"actions": ["no SVG metadata removed", "removed embedded data URI"]}
        )
        is True
    )


def test_changed_stats_path_unchanged():
    assert clean_staged._changed({"stats": {"removed_count": 0, "replaced_count": 0}}) is False
    assert clean_staged._changed({"stats": {"removed_count": 2, "replaced_count": 0}}) is True


def test_clean_hook_passes_on_unchanged_png(tmp_path, monkeypatch):
    """End to end: cleaning an already-clean PNG twice exits 0 both times."""
    import struct
    import zlib

    def chunk(ctype, payload):
        return (
            struct.pack(">I", len(payload))
            + ctype
            + payload
            + struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")
    f = tmp_path / "clean.png"
    f.write_bytes(png)
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(f)])
    assert clean_staged.main() == 0
    assert f.read_bytes() == png
