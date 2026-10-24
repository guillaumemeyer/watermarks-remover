"""Tests for the FLUX 3 OpenRouter client and its cost guard (no network: httpx.post is mocked)."""

from __future__ import annotations

import base64
import os
import struct
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

JPEG_1X1 = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFAEBAAAAAAAAAAAAAAAAAAAABP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AI6AaG//2Q=="
)

BASE64_1X1_PNG = base64.b64encode(
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
).decode()


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(payload, zlib.crc32(ctype)) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _make_png(width: int, height: int) -> bytes:
    """RGB PNG of random noise: incompressible, so 4096x4096 stays ~50 MB on disk."""
    raw = b"".join(b"\x00" + os.urandom(width * 3) for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw, 0))
        + _png_chunk(b"IEND", b"")
    )


@pytest.fixture
def _unload_new_pil():
    """Drop PIL modules this test loaded: test_markdiffusion_harness imports a stub PIL
    in-process and would otherwise get the real Pillow left in sys.modules."""

    def pil_modules() -> set[str]:
        return {name for name in sys.modules if name == "PIL" or name.startswith("PIL.")}

    before = pil_modules()
    yield
    for name in pil_modules() - before:
        del sys.modules[name]


def test_estimate_cost_known_resolutions():
    from flux3_client import estimate_cost

    assert estimate_cost("768") == pytest.approx(0.04)
    assert estimate_cost("1k") == pytest.approx(0.05)
    assert estimate_cost("4k") == pytest.approx(0.61)


def test_estimate_cost_unknown_defaults():
    from flux3_client import estimate_cost

    assert estimate_cost("unknown") == pytest.approx(0.05)


def test_cost_guard_blocks_over_budget():
    from flux3_client import Flux3CostGuard

    guard = Flux3CostGuard(max_per_image=0.10, max_per_batch=0.20)
    assert guard.check("1k") is True
    guard.record(0.05)
    guard.record(0.05)
    guard.record(0.05)
    assert guard.check("4k") is False  # 0.15 + 0.61 > 0.20


def test_cost_guard_per_image_limit():
    from flux3_client import Flux3CostGuard

    guard = Flux3CostGuard(max_per_image=0.10, max_per_batch=100.0)
    assert guard.check("4k") is False  # 0.61 > 0.10


def test_cost_guard_batch_limit_alone():
    from flux3_client import Flux3CostGuard

    guard = Flux3CostGuard(max_per_image=0.50, max_per_batch=0.30)
    guard.record(0.10)
    guard.record(0.10)
    assert guard.check("2k") is True  # 0.20 + 0.10 within budget
    guard.record(0.10)
    assert guard.check("1k") is False  # 0.30 + 0.05 > 0.30


def test_generate_sends_correct_payload(monkeypatch):
    pytest.importorskip("httpx")
    import types

    from flux3_client import Flux3Client

    captured = {}

    def mock_post(url, *, json, headers, timeout):
        captured.update(json=json, url=url, headers=headers, timeout=timeout)
        resp = types.SimpleNamespace(
            status_code=200,
            json=lambda: {
                "data": [{"b64_json": BASE64_1X1_PNG, "media_type": "image/png"}],
                "usage": {"cost": 0.04},
            },
            raise_for_status=lambda: None,
        )
        return resp

    monkeypatch.setattr("flux3_client.httpx.post", mock_post)
    client = Flux3Client(api_key="test-key")
    result = client.generate(JPEG_1X1, "reproduce exactly", resolution="2k")
    assert captured["url"] == "https://openrouter.ai/api/v1/images"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["timeout"] == 120.0
    assert captured["json"]["model"] == "black-forest-labs/flux-3-image"
    assert captured["json"]["resolution"] == "2K"
    assert captured["json"]["prompt"] == "reproduce exactly"
    assert captured["json"]["aspect_ratio"] == "auto"
    assert captured["json"]["n"] == 1
    refs = captured["json"]["input_references"]
    assert len(refs) == 1
    assert refs[0]["type"] == "image_url"
    assert refs[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert result.cost_usd == pytest.approx(0.04)
    assert result.media_type == "image/png"
    assert len(result.image_bytes) > 0


@pytest.mark.usefixtures("_unload_new_pil")
def test_flux3_resizes_large_input(monkeypatch):
    pytest.importorskip("httpx")
    pytest.importorskip("PIL")
    import json as json_mod
    import types

    from flux3_client import Flux3Client

    big_image = _make_png(4096, 4096)  # ~50MB raw
    # Guard against a vacuous pass: inlined without a resize, this input is over the limit.
    assert len(base64.b64encode(big_image)) > 20_000_000
    captured = {}

    def mock_post(url, *, json, headers, timeout):
        captured["payload_size"] = len(json_mod.dumps(json))
        return types.SimpleNamespace(
            status_code=200,
            json=lambda: {
                "data": [{"b64_json": BASE64_1X1_PNG, "media_type": "image/png"}],
                "usage": {"cost": 0.10},
            },
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr("flux3_client.httpx.post", mock_post)
    client = Flux3Client(api_key="test-key")
    client.generate(big_image, "reproduce", resolution="2k")
    assert captured["payload_size"] < 20_000_000  # under 20MB


def test_prompt_strategies_exact_text():
    from flux3_client import PROMPT_STRATEGIES

    assert PROMPT_STRATEGIES == {
        "minimal": "Enhance this image slightly for improved clarity and sharpness",
        "normal": "Faithfully reproduce this photograph maintaining all visual elements exactly",
        "strong": (
            "Create a fresh rendering of this scene with identical subject, composition, "
            "and lighting"
        ),
        "maximum": (
            "Reimagine this image from scratch preserving the exact same scene and composition"
        ),
    }


def test_generate_without_httpx_raises_unavailable(monkeypatch):
    import flux3_client

    monkeypatch.setattr(flux3_client, "httpx", None)
    client = flux3_client.Flux3Client(api_key="test-key")
    with pytest.raises(flux3_client.Flux3Unavailable, match="httpx"):
        client.generate(JPEG_1X1, "reproduce")


def test_generate_malformed_response_raises_flux3_error(monkeypatch):
    pytest.importorskip("httpx")
    import types

    import flux3_client

    def mock_post(url, *, json, headers, timeout):
        return types.SimpleNamespace(
            status_code=200, json=lambda: {"data": []}, raise_for_status=lambda: None
        )

    monkeypatch.setattr("flux3_client.httpx.post", mock_post)
    client = flux3_client.Flux3Client(api_key="test-key")
    with pytest.raises(flux3_client.Flux3Error):
        client.generate(JPEG_1X1, "reproduce")
