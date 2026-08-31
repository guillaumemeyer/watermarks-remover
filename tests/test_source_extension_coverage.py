"""An invisible carrier must not change verdict with the file extension.

``TEXT_EXTS`` in ``format_dispatch.py`` grew one contributor at a time: it
carried ``.gd``/``.gdshader`` for Godot and the whole JS/TS family, but none of
C, C++, Java, Kotlin, C#, Ruby, PHP, Swift, shell, SQL, reStructuredText, or any
localization resource format. ``classify()`` answered ``unknown`` for those, and
both consumers read ``unknown`` as nothing to say: ``audit_lib`` records
"unrecognized format; not scanned", and ``check_staged.py`` skips the file
without printing a line, so the pre-commit gate returned 0.

Witness before the fix — ten files, each carrying exactly one U+200B::

    Files skipped: 0
    Files scanned: 10
    By kind: {'unknown': 7, 'text': 2, 'markdown': 1}
    Actionable files: 3

Only ``.txt``, ``.md`` and ``.ts`` were read; the summary still counted all ten
as scanned. On a Java, C++, Ruby, PHP or shell repository the
``watermarks-remover-check`` hook was a gate that always returned 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import check_staged
from audit_lib import is_actionable, scan_file
from av_meta import AV_EXTS
from format_dispatch import CONTAINER_EXTS, IMAGE_EXTS, TEXT_EXTS, classify

ZWSP = chr(0x200B)
CARRIER = f"Intro{ZWSP} paragraph with a hidden carrier.\n"

# One representative name per family the audit used to walk past.
SOURCE_NAMES = (
    "main.c",
    "engine.h",
    "node.cpp",
    "App.java",
    "Main.kt",
    "Program.cs",
    "lib.rb",
    "index.php",
    "View.swift",
    "job.scala",
    "app.dart",
    "init.lua",
    "run.sh",
    "deploy.ps1",
    "report.sql",
    "Widget.vue",
    "Panel.svelte",
    "guide.rst",
    "manual.adoc",
    "paper.tex",
    "settings.ini",
    "rows.tsv",
)

# Localization resources are where user-facing strings actually live, which is
# what the sibling skill (clean-user-facing-text) exists to clean.
L10N_NAMES = (
    "messages.po",
    "template.pot",
    "Localizable.strings",
    "app_en.arb",
    "Resources.resx",
    "app.properties",
)


@pytest.mark.parametrize("name", SOURCE_NAMES + L10N_NAMES)
def test_extension_classifies_as_text(tmp_path, name):
    path = tmp_path / name
    path.write_text(CARRIER, encoding="utf-8")
    assert classify(path) == "text", f"{name} still routes nowhere"


@pytest.mark.parametrize("name", SOURCE_NAMES + L10N_NAMES)
def test_same_carrier_same_verdict_as_txt(tmp_path, name):
    """Identical bytes must not be actionable as .txt and silent as .java."""
    control = tmp_path / "control.txt"
    control.write_text(CARRIER, encoding="utf-8")
    subject = tmp_path / name
    subject.write_text(CARRIER, encoding="utf-8")

    control_item, subject_item = scan_file(control), scan_file(subject)
    assert subject_item["confidence"] == control_item["confidence"]
    assert is_actionable(subject_item) == is_actionable(control_item)
    assert is_actionable(subject_item)


def test_precommit_gate_catches_source_files(tmp_path, monkeypatch, capsys):
    """The check hook must fail the commit, not pass it without a word."""
    paths = []
    for name in ("main.c", "App.java", "run.sh", "report.sql", "messages.po"):
        path = tmp_path / name
        path.write_text(CARRIER, encoding="utf-8")
        paths.append(str(path))

    monkeypatch.setattr(sys, "argv", ["check_staged.py", *paths])
    assert check_staged.main() == 1
    err = capsys.readouterr().err
    for path in paths:
        assert path in err
    assert "layer-a" in err


def test_clean_source_file_still_passes(tmp_path, monkeypatch):
    """Widening the table must not turn ordinary source into a finding."""
    path = tmp_path / "main.c"
    path.write_text('#include <stdio.h>\nint main(void) { puts("hi"); }\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(path)])
    assert check_staged.main() == 0


def test_utf16_localization_file_yields_no_findings(tmp_path, monkeypatch):
    """macOS .strings files are often UTF-16; decoding them must not invent hits."""
    path = tmp_path / "Localizable.strings"
    path.write_text('"greeting" = "Hello";\n', encoding="utf-16")
    item = scan_file(path)
    assert item["suspicious_total"] == 0
    assert not is_actionable(item)

    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(path)])
    assert check_staged.main() == 0


def test_extension_tables_stay_disjoint():
    """A suffix must resolve to exactly one pipeline."""
    tables = {
        "image": IMAGE_EXTS,
        "container": CONTAINER_EXTS,
        "av": AV_EXTS,
        "text": TEXT_EXTS,
    }
    names = sorted(tables)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = tables[left] & tables[right]
            assert not overlap, f"{left}/{right} both claim {sorted(overlap)}"
