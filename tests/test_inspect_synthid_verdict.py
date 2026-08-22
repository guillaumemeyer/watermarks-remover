import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import image_meta
import server


def _clean_png_bytes() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    ihdr_chunk = (
        struct.pack(">I", len(ihdr))
        + b"IHDR"
        + ihdr
        + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF)
    )
    iend_chunk = (
        struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    )
    return b"\x89PNG\r\n\x1a\n" + ihdr_chunk + iend_chunk


def test_inspect_image_synthid_watermark_marked_suspicious(monkeypatch):
    data = _clean_png_bytes()

    # Mock SynthID scorer returning watermarked
    monkeypatch.setattr(
        image_meta,
        "run_synthid_score",
        lambda path, synthid_dir=None: {
            "available": True,
            "is_watermarked": True,
            "confidence": 0.92,
            "phase_match": 0.88,
        },
    )

    res = server._inspect_payload(data, "test.png", run_detect=False)
    assert res["ok"] is True
    assert res["suspicious"] is True
    assert any("synthid" in f.lower() for f in res["report"]["findings"])


def test_inspect_image_synthid_failure_marks_inconclusive(monkeypatch):
    data = _clean_png_bytes()

    # Mock SynthID scorer error
    monkeypatch.setattr(
        image_meta,
        "run_synthid_score",
        lambda path, synthid_dir=None: {
            "available": False,
            "error": "connection refused to synthid backend",
        },
    )

    res = server._inspect_payload(data, "test.png", run_detect=False)
    assert res["ok"] is True
    assert any("inconclusive" in f.lower() for f in res["report"]["findings"])


def test_inspect_image_synthid_clean_verdict_not_suspicious(monkeypatch):
    data = _clean_png_bytes()

    monkeypatch.setattr(
        image_meta,
        "run_synthid_score",
        lambda path, synthid_dir=None: {
            "available": True,
            "is_watermarked": False,
            "confidence": 0.05,
            "phase_match": 0.01,
        },
    )

    res = server._inspect_payload(data, "test.png", run_detect=False)
    assert res["ok"] is True
    assert res["suspicious"] is False
    assert not any("synthid" in f.lower() for f in res["report"]["findings"])
