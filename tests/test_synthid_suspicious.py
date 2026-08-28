"""SynthID image score feeds the /inspect verdict (#165).

A scorer saying "watermarked" used to read as clean (detected_wm covered text
detectors only), and a failed scorer was indistinguishable from one that ran
and found nothing.
"""

from __future__ import annotations

import base64
import http.client
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import server  # noqa: E402


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _png() -> bytes:
    import struct
    import zlib

    def chunk(ctype: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + ctype
            + payload
            + struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


def _payload_with_synthid(monkeypatch, available: bool, watermarked: bool, error: str | None = None):
    def fake_inspect_image(path):
        import image_meta

        report = image_meta.inspect_image(path)
        report.synthid = (
            {"available": available, "is_watermarked": watermarked, "score": 0.97 if watermarked else 0.03}
            if error is None
            else {"available": False, "error": error}
        )
        return report

    monkeypatch.setattr(server, "inspect_image", fake_inspect_image)


def test_synthid_watermarked_is_suspicious(monkeypatch):
    _payload_with_synthid(monkeypatch, available=True, watermarked=True)
    out = server._inspect_payload(_png(), "img.png", run_detect=False)
    assert out["suspicious"] is True
    assert out["report"]["synthid"]["is_watermarked"] is True


def test_synthid_clean_low_score_stays_not_suspicious(monkeypatch):
    _payload_with_synthid(monkeypatch, available=True, watermarked=False)
    out = server._inspect_payload(_png(), "img.png", run_detect=False)
    assert out["suspicious"] is False


def test_synthid_scorer_failure_is_surfaced_not_silent(monkeypatch):
    _payload_with_synthid(monkeypatch, available=False, watermarked=False, error="sidecar unreachable")
    out = server._inspect_payload(_png(), "img.png", run_detect=False)
    assert out["suspicious"] is False
    assert out.get("synthid_probe_failed") is True
    assert out["report"]["synthid"]["error"] == "sidecar unreachable"


def test_no_synthid_at_all_is_plain_not_suspicious(monkeypatch):
    def fake_inspect_image(path):
        import image_meta

        report = image_meta.inspect_image(path)
        report.synthid = None
        return report

    monkeypatch.setattr(server, "inspect_image", fake_inspect_image)
    out = server._inspect_payload(_png(), "img.png", run_detect=False)
    assert out["suspicious"] is False
    assert "synthid_probe_failed" not in out
