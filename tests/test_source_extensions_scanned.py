"""A zero-width carrier in TypeScript or GDScript must not pass as unknown.

TEXT_EXTS carried ``.js`` but none of the extensions a current front end or a
Godot project is actually written in, so ``audit_dir.py`` classified them
"unknown; not scanned" while still counting them in "Files scanned". A repo of
``.ts``/``.tsx`` audited clean because nothing had read it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_lib import is_actionable, scan_file
from format_dispatch import classify

ZWSP = "​"

SOURCE_EXTS = [".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".gd", ".gdshader"]


def test_source_extensions_classify_as_text(tmp_path):
    for ext in SOURCE_EXTS:
        f = tmp_path / ("sample" + ext)
        f.write_text("const label = 'x'\n", encoding="utf-8")
        assert classify(f) == "text", f"{ext} still routes as {classify(f)}"


def test_zero_width_in_source_is_found(tmp_path):
    for ext in SOURCE_EXTS:
        f = tmp_path / ("carrier" + ext)
        f.write_text(f"const label = 'title{ZWSP}here'\n", encoding="utf-8")
        item = scan_file(f)
        assert item["kind"] != "unknown", f"{ext} was not scanned"
        assert item["suspicious_total"] >= 1, f"{ext} hid a zero-width carrier"
        assert is_actionable(item), f"{ext} reported a carrier as non-actionable"


def test_unknown_binary_still_unknown(tmp_path):
    """The widened set must not turn every unrecognized file into text."""
    f = tmp_path / "blob.bin"
    f.write_bytes(b"\x00\x01\x02\x03not text at all")
    assert classify(f) == "unknown"
