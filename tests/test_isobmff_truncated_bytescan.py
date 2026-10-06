"""Truncated ISOBMFF containers still run the whole-file C2PA byte scan (#167).

A first box whose size overruns the file (a truncated AVIF/HEIC/MP4 download)
used to return early with a not-a-valid finding, skipping the byte-scan
fallback every sibling inspector reaches — literal b"c2pa" could be present
and the report still said clean.

The inverse also holds: a container that parsed completely but has a 1-7 byte
tail must NOT re-enable the byte scan (#371).
"""

from __future__ import annotations

import base64
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import av_meta
import container_meta
import image_meta


def _avif(marker: bytes, first_box_overruns: bool) -> bytes:
    ftyp = b"\x00\x00\x00\x14ftypavif" + b"\x00\x00\x00\x00"  # 20-byte ftyp
    body = marker + b"\x00" * 16
    if first_box_overruns:
        # First box (ftyp) declares a size larger than the whole file.
        size = len(ftyp) + len(body) + 64
        head = struct.pack(">I", size) + b"ftypavif" + b"\x00" * 8
        return head + body
    return struct.pack(">I", len(ftyp)) + b"ftypavif" + b"\x00" * 8 + body


def test_truncated_avif_still_bytescans_for_c2pa():
    data = _avif(b"c2pa contentcredentials", first_box_overruns=True)
    assert b"c2pa" in data

    has_c2pa, _, findings = image_meta.inspect_isobmff(data, fmt="avif")
    # The byte-scan fallback fires instead of the early clean-looking return.
    assert has_c2pa is True, findings
    assert any("byte-scan C2PA markers" in f for f in findings), findings
    # The parse failure is still reported.
    assert any("no ISOBMFF boxes found" in f for f in findings), findings


def test_intact_avif_box_parse_path_unchanged():
    data = _avif(b"c2pa contentcredentials", first_box_overruns=False)
    has_c2pa, _, findings = image_meta.inspect_isobmff(data, fmt="avif")
    assert has_c2pa is True
    # A walkable container finds the C2PA box directly (stronger evidence than
    # the byte scan); the parse-failure note must not appear.
    assert any("top-level box" in f and "c2pa" in f.lower() for f in findings), findings
    assert not any("no ISOBMFF boxes found" in f for f in findings)


def test_truncated_markerless_avif_reports_parse_failure():
    data = _avif(b"nothing interesting", first_box_overruns=True)
    has_c2pa, _, findings = image_meta.inspect_isobmff(data, fmt="avif")
    assert has_c2pa is False
    assert findings == ["not a valid AVIF (no ISOBMFF boxes found)"]


# ---------------------------------------------------------------------------
# A short trailing tail must not re-enable the byte scan (#371 regression)
# ---------------------------------------------------------------------------


def _box(fourcc: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + fourcc + payload


def _parsed_container(brand: bytes, mdat_payload: bytes, tail: bytes = b"") -> bytes:
    """ftyp + a fully parseable mdat, plus `tail` bytes after the last box.

    The only C2PA-looking bytes live inside the media payload, so a complete box
    walk must stay silent (#371). `tail` is padding a real muxer can leave
    behind: `_parse_isobmff_boxes` walks `while pos + 8 <= end`, so 1-7 trailing
    bytes stop the walk short of EOF while the container is fully parsed and not
    truncated at all. Gating the byte scan on `scanned_end < len(data)` reads
    that as "parsing stopped early" and re-enables the scan; the gate the rest
    of this tree uses is a tail of >= 8 bytes (#170, and av_meta's
    `len(data) - scanned_end >= 8`).
    """
    ftyp = _box(b"ftyp", brand + b"\x00\x00\x00\x00" + brand)
    return ftyp + _box(b"mdat", mdat_payload) + tail


# Chance ASCII in a compressed stream -- the C2PA_MARKERS byte scan.
MDAT_ASCII_MARKER = b"\x00" * 16 + b"c2pa" + b"\x00" * 16
# Chance `uuid` + content-provenance user type -- the _contains_c2pa_prov_box scan.
MDAT_PROV_UUID = b"\x00" * 16 + b"uuid" + image_meta.C2PA_BMFF_UUID + b"\x00" * 16
# 0, 1, 4 and 7 trailing bytes. The 7-byte tail is non-zero to show the gate is
# about the tail's length, not its contents.
SHORT_TAILS = [b"", b"\x00", b"\x00\x00\x00\x00", b"\x01\x02\x03\x04\x05\x06\x07"]


@pytest.mark.parametrize("tail", SHORT_TAILS)
def test_short_tail_keeps_parsed_mdat_markers_ignored(tail):
    data = _parsed_container(b"isom", MDAT_ASCII_MARKER, tail)
    assert b"c2pa" in data  # the marker really is there, inside the media payload
    has_c2pa, has_ai, findings = image_meta.inspect_isobmff(data, fmt="mp4")
    assert has_c2pa is False, findings
    assert has_ai is False, findings
    assert not any("byte-scan" in f for f in findings), findings


@pytest.mark.parametrize("tail", SHORT_TAILS)
def test_short_tail_keeps_parsed_prov_uuid_bytes_ignored(tail):
    # The second gate: a UUID-shaped byte run in mdat must not be read as a
    # content-provenance box either.
    data = _parsed_container(b"isom", MDAT_PROV_UUID, tail)
    has_c2pa, has_ai, findings = image_meta.inspect_isobmff(data, fmt="mp4")
    assert has_c2pa is False, findings
    assert has_ai is False, findings
    assert not any("byte-scan" in f for f in findings), findings


def test_eight_byte_tail_still_byte_scans():
    # An 8-byte header whose box overruns the file is a truncated download, and
    # the byte scan must still run for it (#167/#170/#176).
    data = _parsed_container(b"isom", MDAT_ASCII_MARKER, b"\xff" * 8)
    has_c2pa, _has_ai, findings = image_meta.inspect_isobmff(data, fmt="mp4")
    assert has_c2pa is True, findings
    assert any("byte-scan C2PA markers" in f for f in findings), findings


def test_truncated_c2pa_manifest_box_still_detected():
    # A real content-provenance box cut off mid-manifest carries no ASCII
    # marker, so only the user-type scan can catch it -- and it must, because
    # the 112-byte tail it leaves is a genuine truncation.
    manifest = _box(b"uuid", image_meta.C2PA_BMFF_UUID + b"manifest\x00" + bytes(range(1, 100)))
    data = _parsed_container(b"isom", b"\x00" * 16) + manifest[:112]
    has_c2pa, has_ai, findings = image_meta.inspect_isobmff(data, fmt="mp4")
    assert has_c2pa is True, findings
    assert has_ai is True, findings
    assert any("content-provenance user type" in f for f in findings), findings


def test_mp4_short_tail_report_is_clean(tmp_path):
    # The user-visible symptom, via av_meta's caller: a padded MP4 reported as
    # C2PA-marked makes `wr inspect` exit 1 and `wr clean` warn about residual
    # signals while writing back byte-identical output.
    src = tmp_path / "clip.mp4"
    src.write_bytes(_parsed_container(b"isom", MDAT_ASCII_MARKER, b"\x00\x00\x00\x00"))

    report = av_meta.inspect_av(src)
    assert report.format == "mp4"
    assert report.has_c2pa is False, report.findings
    assert report.has_ai_metadata is False, report.findings
    assert "likely_false_positive" not in report.to_dict()["findings_confidence"]


def test_avif_short_tail_through_embedded_data_uri():
    # The gate lives in the shared inspect_isobmff, so a non-MP4 caller must be
    # fixed by the same change: container_meta routes embedded AVIF/HEIC images
    # through it too.
    avif = _parsed_container(b"avif", MDAT_ASCII_MARKER, b"\x00\x00\x00\x00")
    assert image_meta.detect_format(avif) == "avif"
    html = '<img src="data:image/avif;base64,' + base64.b64encode(avif).decode("ascii") + '">'
    has_c2pa, has_ai, findings, _details = container_meta.inspect_html(html)
    assert has_c2pa is False, findings
    assert has_ai is False, findings
    assert not any("byte-scan" in f for f in findings), findings
