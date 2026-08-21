"""Truncated ID3v2 handling (#163).

A tag whose declared size overruns the file used to inspect as clean (empty
findings AND empty notes — byte-for-byte a genuinely clean MP3) and the
cleaner wrote the file back verbatim, certifying it. Both must now say
"unread"/"dropped".
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import av_meta


def _syncsafe(size: int) -> bytes:
    return bytes([(size >> 21) & 0x7F, (size >> 14) & 0x7F, (size >> 7) & 0x7F, size & 0x7F])


def _id3_header(declared_size: int) -> bytes:
    # ID3v2.3 header: "ID3", major=3, minor=0, flags=0, 4-byte syncsafe size.
    return b"ID3" + bytes([3, 0, 0]) + _syncsafe(declared_size)


def test_truncated_tag_reports_unread_not_clean():
    header = _id3_header(160)  # declares 170 bytes total
    data = header + b"contentcredentials" + b"\x00" * 40
    assert len(data) < 170, "fixture must be shorter than the declared tag"

    has_c2pa, has_ai, findings = av_meta._inspect_id3v2(data)
    # The failure surfaces as a finding (audits aggregate findings), naming
    # the truncation rather than mimicking a clean result.
    assert has_c2pa is False and has_ai is False
    assert any("truncated" in f.lower() for f in findings), findings


def test_full_tag_still_parses_clean():
    data = _id3_header(4) + b"abcd" + b"\x00" * 4
    # A well-formed small tag parses without the truncation note.
    _, _, findings = av_meta._inspect_id3v2(data)
    assert not any("truncated" in f.lower() for f in findings)


def test_truncated_tag_is_dropped_not_certified():
    data = _id3_header(160) + b"contentcredentials" + b"\x00" * 40

    out, actions = av_meta._strip_id3v2(data, strip_all_metadata=False)
    assert any("truncated" in a.lower() for a in actions), actions
    # The tag bytes (and any markers in the readable prefix) are gone.
    assert b"contentcredentials" not in out
    assert not out.startswith(b"ID3")
