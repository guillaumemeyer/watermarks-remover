"""Tests for the FLUX 3 CLI wrapper (no network: FLUX3_TEST_MOCK or a patched client)."""

from __future__ import annotations

import base64
import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import clean_flux3
import image_meta
from flux3_client import (
    PROMPT_STRATEGIES,
    Flux3Client,
    Flux3Error,
    Flux3Result,
    Flux3Unavailable,
)

JPEG_1X1 = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
    "Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwh"
    "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAAR"
    "CAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAA"
    "AAAA/8QAFAEBAAAAAAAAAAAAAAAAAAAABP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhED"
    "EQA/AI6AaG//2Q=="
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """No real API key, mock switch or SynthID scorer leaks in from the developer's shell."""
    for name in (
        "OPENROUTER_API_KEY",
        "FLUX3_TEST_MOCK",
        "FLUX3_TEST_MOCK_BYTES",
        "REVERSE_SYNTHID_DIR",
        "WATERMARKS_SYNTHID_SCORER_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _make_png_with_c2pa_jumbf(w: int, h: int) -> bytes:
    """A decodable gray PNG carrying C2PA/JUMBF markers the way a FLUX 3 response might."""
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    rows = b"".join(b"\x00" + b"\x80\x80\x80" * w for _ in range(h))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"tEXt", b"Comment\x00jumb c2pa manifest by Black Forest Labs")
        + _png_chunk(b"caBX", b"\x00\x00\x00\x20jumb\x00\x00\x00\x18jumdc2pa")
        + _png_chunk(b"IDAT", zlib.compress(rows))
        + _png_chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------------------
# Subprocess: exit codes and the mocked round trip
# ---------------------------------------------------------------------------


def test_missing_api_key_exits_3(tmp_path):
    img = tmp_path / "test.png"
    img.write_bytes(JPEG_1X1)
    rc = _run("clean_flux3.py", str(img)).returncode
    assert rc == 3


def test_nonexistent_file_exits_2():
    rc = _run("clean_flux3.py", "/nonexistent.png", "--api-key", "k").returncode
    assert rc == 2


def test_non_image_input_exits_2(tmp_path):
    src = tmp_path / "notes.png"
    src.write_bytes(b"not an image at all")
    rc = _run("clean_flux3.py", str(src), "--api-key", "k").returncode
    assert rc == 2


def test_flux3_strips_output_metadata(tmp_path, monkeypatch):
    """FLUX 3 output must not carry C2PA or BFL watermarks."""
    img = tmp_path / "wm.png"
    img.write_bytes(JPEG_1X1)
    out = tmp_path / "clean.png"
    c2pa_png = _make_png_with_c2pa_jumbf(256, 256)
    monkeypatch.setenv("FLUX3_TEST_MOCK_BYTES", base64.b64encode(c2pa_png).decode())
    monkeypatch.setenv("FLUX3_TEST_MOCK", "1")
    cp = _run("clean_flux3.py", str(img), "-o", str(out), "--api-key", "test-key")
    assert cp.returncode == 0, cp.stderr
    out_bytes = out.read_bytes()
    assert out_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"jumb" not in out_bytes
    assert b"c2pa" not in out_bytes.lower()
    assert b"black forest" not in out_bytes.lower()


def test_clean_flux3_produces_output(tmp_path, monkeypatch):
    img = tmp_path / "wm.png"
    img.write_bytes(JPEG_1X1)
    out = tmp_path / "clean.png"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("FLUX3_TEST_MOCK", "1")
    cp = _run("clean_flux3.py", str(img), "-o", str(out), "--api-key", "test-key", "--json")
    assert cp.returncode == 0, cp.stderr
    assert out.is_file()
    report = json.loads(cp.stdout)
    assert "cost_usd" in report
    assert report["cost_usd"] == 0.0
    assert report["mock"] is True
    # Without FLUX3_TEST_MOCK_BYTES the "generated" image is the input itself.
    assert out.read_bytes().startswith(b"\xff\xd8\xff")
    assert report["synthid_check"] == "skipped"
    assert report["flux3_watermark_detected"] is None


def test_cost_guard_refuses_over_budget(tmp_path, monkeypatch):
    img = tmp_path / "wm.jpg"
    img.write_bytes(JPEG_1X1)
    out = tmp_path / "clean.jpg"
    monkeypatch.setenv("FLUX3_TEST_MOCK", "1")
    # 4k is estimated at $0.61, over the $0.50 default.
    cp = _run("clean_flux3.py", str(img), "-o", str(out), "--api-key", "k", "--resolution", "4k")
    assert cp.returncode == 2
    assert "max-cost" in cp.stderr
    assert not out.exists()
    cp = _run(
        "clean_flux3.py",
        *(str(img), "-o", str(out), "--api-key", "k", "--resolution", "4k", "--max-cost", "1"),
    )
    assert cp.returncode == 0, cp.stderr
    assert out.is_file()


@pytest.mark.parametrize("provider", ["bfl", "fal"])
def test_unimplemented_provider_exits_3(tmp_path, monkeypatch, provider):
    img = tmp_path / "wm.jpg"
    img.write_bytes(JPEG_1X1)
    monkeypatch.setenv("FLUX3_TEST_MOCK", "1")
    cp = _run("clean_flux3.py", str(img), "--api-key", "k", "--provider", provider)
    assert cp.returncode == 3
    assert not (tmp_path / "wm.flux3.jpg").exists()


# ---------------------------------------------------------------------------
# In-process: the real (non-mock) client path and the SynthID post-check
# ---------------------------------------------------------------------------


def test_real_path_sends_strategy_prompt_and_reports_cost(tmp_path, monkeypatch, capsys):
    img = tmp_path / "wm.jpg"
    img.write_bytes(JPEG_1X1)
    calls = []

    def fake_generate(self, image_bytes, prompt, resolution="1k", aspect_ratio="auto"):
        calls.append((self.api_key, image_bytes, prompt, resolution))
        return Flux3Result(_make_png_with_c2pa_jumbf(8, 8), 0.1, "image/png")

    monkeypatch.setattr(Flux3Client, "generate", fake_generate)
    rc = clean_flux3.main(
        [str(img), "--api-key", "k", "--strength", "strong", "--resolution", "2k", "--json"]
    )
    assert rc == 0
    assert calls == [("k", JPEG_1X1, PROMPT_STRATEGIES["strong"], "2k")]
    report = json.loads(capsys.readouterr().out)
    assert report["cost_usd"] == 0.1
    assert report["estimated_cost_usd"] == 0.1
    assert report["mock"] is False
    assert report["format"] == "png"
    # JPEG in, PNG back: the default name follows the format FLUX 3 actually returned.
    out = tmp_path / "wm.flux3.png"
    assert report["output"] == str(out)
    assert b"c2pa" not in out.read_bytes().lower()


def test_custom_prompt_overrides_strength(tmp_path, monkeypatch, capsys):
    img = tmp_path / "wm.jpg"
    img.write_bytes(JPEG_1X1)
    prompts = []

    def fake_generate(self, image_bytes, prompt, resolution="1k", aspect_ratio="auto"):
        prompts.append(prompt)
        return Flux3Result(JPEG_1X1, 0.05, "image/jpeg")

    monkeypatch.setattr(Flux3Client, "generate", fake_generate)
    rc = clean_flux3.main([str(img), "--api-key", "k", "--prompt", "redraw it", "--json"])
    assert rc == 0
    assert prompts == ["redraw it"]
    report = json.loads(capsys.readouterr().out)
    assert report["prompt"] == "redraw it"
    assert report["strength"] is None


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (Flux3Error("HTTP 500"), 1),
        (ValueError("not a PNG, JPEG or WebP"), 2),
        (Flux3Unavailable("httpx is not installed"), 3),
    ],
)
def test_client_errors_map_to_exit_codes(tmp_path, monkeypatch, exc, expected):
    img = tmp_path / "wm.jpg"
    img.write_bytes(JPEG_1X1)

    def fake_generate(self, image_bytes, prompt, resolution="1k", aspect_ratio="auto"):
        raise exc

    monkeypatch.setattr(Flux3Client, "generate", fake_generate)
    assert clean_flux3.main([str(img), "--api-key", "k"]) == expected
    assert not (tmp_path / "wm.flux3.jpg").exists()


@pytest.mark.parametrize(("confidence", "detected"), [(0.9, True), (0.6, False)])
def test_synthid_threshold_flags_flux3_watermark(
    tmp_path, monkeypatch, capsys, confidence, detected
):
    img = tmp_path / "wm.jpg"
    img.write_bytes(JPEG_1X1)
    monkeypatch.setenv("FLUX3_TEST_MOCK", "1")
    scored = []

    def fake_score(path, upstream_dir=None, *, data=None):
        scored.append(Path(path))
        return {"available": True, "confidence": confidence, "is_watermarked": True}

    monkeypatch.setattr(image_meta, "run_synthid_score", fake_score)
    rc = clean_flux3.main([str(img), "--api-key", "k", "--json"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["synthid_check"] == "scored"
    assert report["synthid"]["confidence"] == confidence
    assert report["flux3_watermark_detected"] is detected
    # A detection triggers a second metadata strip of the written output.
    assert ("restrip_changed" in report) is detected
    out = tmp_path / "wm.flux3.jpg"
    assert (out in scored) is detected


def test_unavailable_scorer_is_not_reported_as_clean(tmp_path, monkeypatch, capsys):
    img = tmp_path / "wm.jpg"
    img.write_bytes(JPEG_1X1)
    monkeypatch.setenv("FLUX3_TEST_MOCK", "1")
    monkeypatch.setattr(
        image_meta,
        "run_synthid_score",
        lambda path, upstream_dir=None, *, data=None: {"available": False, "error": "no deps"},
    )
    assert clean_flux3.main([str(img), "--api-key", "k", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["synthid_check"] == "unavailable"
    assert report["flux3_watermark_detected"] is None
