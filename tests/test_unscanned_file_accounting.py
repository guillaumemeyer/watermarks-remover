"""A file nothing read must not be counted as a file that was scanned.

``classify()`` answers ``unknown`` for every suffix outside the four routing
tables, and both consumers read that as nothing to say: ``aggregate()`` counted
the item under ``Files scanned`` while ``Files skipped`` only ever tracked read
failures, and ``check_staged.py`` dropped it with a bare ``continue``. The
result is a summary that reports ten files read when it read three, and a gate
that returns 0 on a set of files it never opened.

Widening an extension table moves individual suffixes across that line; it does
not change what the numbers mean. These tests pin the accounting itself: what
was read is counted as read, what was not read is counted and named.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import check_staged
from audit_lib import aggregate, print_human_report, scan_file

ZWSP = chr(0x200B)
CARRIER = f"Intro{ZWSP} paragraph with a hidden carrier.\n"

# Suffixes no table claims, so ``classify()`` answers "unknown" and the byte
# sniffer finds no image/container/av signature either.
UNREAD_NAMES = ("settings.env", "matrix.m", "deps.lock", "notes.xyz")


def _mixed_tree(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    """Two files the scanners read, four they walk past."""
    read = []
    for name in ("a.txt", "b.md"):
        path = tmp_path / name
        path.write_text(CARRIER, encoding="utf-8")
        read.append(path)
    unread = []
    for name in UNREAD_NAMES:
        path = tmp_path / name
        path.write_text(CARRIER, encoding="utf-8")
        unread.append(path)
    return read, unread


def test_aggregate_separates_read_from_unrecognized(tmp_path):
    read, unread = _mixed_tree(tmp_path)
    summary = aggregate([scan_file(p) for p in read + unread])

    assert summary["read"] == len(read)
    assert summary["unrecognized"] == len(unread)
    assert summary["total"] == len(read) + len(unread)


def test_aggregate_on_read_files_only_reports_nothing_unrecognized(tmp_path):
    read, _ = _mixed_tree(tmp_path)
    summary = aggregate([scan_file(p) for p in read])

    assert summary["unrecognized"] == 0
    assert summary["read"] == summary["total"] == len(read)


def test_scanned_line_counts_only_the_files_that_were_read(tmp_path, capsys):
    read, unread = _mixed_tree(tmp_path)
    files = [scan_file(p) for p in read + unread]
    print_human_report(files, aggregate(files))
    out = capsys.readouterr().out

    assert f"Files scanned: {len(read)}" in out
    assert f"Files unrecognized: {len(unread)}" in out


def test_precommit_gate_names_the_files_it_did_not_read(tmp_path, monkeypatch, capsys):
    """Silence is the defect: an unread file is not a file that passed."""
    _, unread = _mixed_tree(tmp_path)
    paths = [str(p) for p in unread]

    monkeypatch.setattr(sys, "argv", ["check_staged.py", *paths])
    # The exit code stays 0 on purpose: nobody's gate turns red overnight
    # because this hook learned to speak.
    assert check_staged.main() == 0

    err = capsys.readouterr().err
    for path in paths:
        assert path in err
    assert "not scanned" in err


def test_precommit_gate_reports_unread_files_alongside_findings(tmp_path, monkeypatch, capsys):
    """A carrier in one file must not hide the fact that another went unread."""
    read, unread = _mixed_tree(tmp_path)
    paths = [str(p) for p in read + unread]

    monkeypatch.setattr(sys, "argv", ["check_staged.py", *paths])
    assert check_staged.main() == 1

    err = capsys.readouterr().err
    assert "layer-a" in err
    for path in unread:
        assert str(path) in err


def test_precommit_gate_stays_quiet_when_every_file_was_read(tmp_path, monkeypatch, capsys):
    """No new noise on a repository the scanners fully understand."""
    path = tmp_path / "clean.txt"
    path.write_text("Ordinary prose, nothing hidden.\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(path)])
    assert check_staged.main() == 0
    assert capsys.readouterr().err == ""
