"""The same non-breaking space must not change confidence with the extension.

``text_hit_confidence()`` calls a Layer A space hit "informational" because a
space homoglyph is weaker context than an invisible carrier. Container formats
(markdown, HTML) are classified by ``classify_finding_confidence()`` instead,
which had no such rule, so identical bytes came back "informational" as a
``.txt`` and "probable" as a ``.md`` — and only the ``.md`` made the file
actionable, which is what a pre-commit gate reads.

Measured on a French manuscript: 430 correct non-breaking spaces across four
chapters, every one of them reported as a probable AI mark because the files
are ``.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_lib import is_actionable, scan_file
from common import classify_finding_confidence

NBSP = " "
NNBSP = " "
ZWSP = "​"

FRENCH = f"Le doute{NBSP}: il reste entier.{NNBSP}Elle demanda{NNBSP}?\n"


def test_container_space_finding_is_informational():
    finding = "layer-a: U+00A0 U+00A0 NO-BREAK SPACE (Zs) x150 (space)"
    assert classify_finding_confidence(finding) == "informational"


def test_container_invisible_finding_stays_probable():
    finding = "layer-a: U+200B U+200B ZERO WIDTH SPACE (Cf) x1 (zwj_family)"
    assert classify_finding_confidence(finding) == "probable"


def test_same_bytes_same_confidence(tmp_path):
    md = tmp_path / "chapter.md"
    txt = tmp_path / "chapter.txt"
    md.write_text(FRENCH, encoding="utf-8")
    txt.write_text(FRENCH, encoding="utf-8")

    md_item, txt_item = scan_file(md), scan_file(txt)
    assert md_item["confidence"] == txt_item["confidence"]
    assert is_actionable(md_item) == is_actionable(txt_item)
    assert not is_actionable(md_item), "French typography alone must not gate a commit"


def test_real_carrier_in_markdown_is_still_actionable(tmp_path):
    """The exemption must not hide an actual invisible carrier."""
    md = tmp_path / "carrier.md"
    md.write_text(f"un texte{ZWSP} pique\n", encoding="utf-8")
    item = scan_file(md)
    assert is_actionable(item)
