"""Tests for the HTTP detection surface: /detect, /inspect detect flag,
/clean detect_before/detect_after, capabilities, and the SynthID sidecar."""

from __future__ import annotations

import base64
import http.client
import json
import struct
import subprocess
import sys
import threading
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import image_meta
import server
import text_detectors


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _watermarked_png() -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00")
    text = b"Comment\x00c2pa test contentcredentials"
    return (
        sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"tEXt", text)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _post(conn, path: str, payload: dict) -> tuple[int, dict]:
    conn.request(
        "POST",
        path,
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    data = resp.read()
    return resp.status, json.loads(data) if data else {}


def _get(conn, path: str) -> tuple[int, dict]:
    conn.request("GET", path)
    resp = conn.getresponse()
    data = resp.read()
    return resp.status, json.loads(data) if data else {}


class _FakeResp:
    def __init__(self, data: dict):
        self._data = json.dumps(data).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._data


@pytest.fixture(scope="module")
def conn() -> http.client.HTTPConnection:
    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
    yield c
    c.close()
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "WATERMARKS_SYNTHID_SCORER_URL",
        "WATERMARKS_SYNTHID_SCORER_API_KEY",
        "MARKLLM_DIR",
    ):
        monkeypatch.delenv(key, raising=False)


def test_capabilities_exposes_detectors(conn):
    status, body = _get(conn, "/capabilities")
    assert status == 200
    assert set(body["text_detectors"]) == {"markllm", "gumbel", "claude-text"}
    assert "synthid_http" in body["scorers"]


def test_openapi_includes_detect(conn):
    status, body = _get(conn, "/openapi.json")
    assert status == 200
    assert "/detect" in body["paths"]
    assert (
        "detect_before"
        in body["paths"]["/clean"]["post"]["requestBody"]["content"]["application/json"]["schema"][
            "properties"
        ]["options"]["properties"]
    )


def test_detect_text_without_detectors(conn):
    payload = {"file": _b64(b"some plain text"), "name": "notes.txt"}
    status, body = _post(conn, "/detect", payload)
    assert status == 200
    assert body["kind"] == "text"
    names = {d["detector"] for d in body["detections"]}
    assert "stylometry" in names
    assert "claude-text" in names  # placeholder always reports unavailable


def _markllm_watermarked(monkeypatch, tmp_path):
    """Configure the MarkLLM detector with a stubbed subprocess result."""
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    monkeypatch.setenv("MARKLLM_DIR", str(upstream))

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps({"is_watermarked": True, "score": 1.0})
        )

    monkeypatch.setattr(text_detectors.subprocess, "run", fake_run)


def test_detect_text_with_markllm(conn, monkeypatch, tmp_path):
    _markllm_watermarked(monkeypatch, tmp_path)
    payload = {"file": _b64(b"watermarked prose here"), "name": "notes.txt"}
    status, body = _post(conn, "/detect", payload)
    assert status == 200
    markllm = next(d for d in body["detections"] if d["detector"] == "markllm")
    assert markllm["available"] is True
    assert markllm["is_watermarked"] is True


def test_inspect_detect_is_opt_in(conn, monkeypatch, tmp_path):
    _markllm_watermarked(monkeypatch, tmp_path)
    txt = b"watermarked prose here"
    # without the flag: no detector calls, no text_detectors key
    status, body = _post(conn, "/inspect", {"file": _b64(txt), "name": "notes.txt"})
    assert status == 200
    assert "text_detectors" not in body["report"]
    # with the flag: detector results appear and can flip suspicious
    status, body = _post(conn, "/inspect", {"file": _b64(txt), "name": "notes.txt", "detect": True})
    assert status == 200
    assert "text_detectors" in body["report"]
    assert body["suspicious"]["verdict"] is True
    wm = body["suspicious"]["classes"]["watermark_detector"]
    assert wm["present"] is True
    assert wm["strength"] == "scheme_specific"
    assert wm["description"]
    assert body["suspicious"]["description"]


def test_clean_text_detect_before_after(conn, monkeypatch, tmp_path):
    _markllm_watermarked(monkeypatch, tmp_path)
    txt = ("watermarked prose here. " * 5).encode("utf-8")
    status, body = _post(
        conn,
        "/clean",
        {
            "file": _b64(txt),
            "name": "notes.txt",
            "options": {"detect_before": True, "detect_after": True},
        },
    )
    assert status == 200
    det = body["report"]["text_detectors"]
    assert set(det) == {"before", "after"}
    assert det["before"][0]["is_watermarked"] is True
    assert det["after"][0]["is_watermarked"] is True


def test_clean_image_detect_before_after_sidecar(conn, monkeypatch):
    monkeypatch.setenv("WATERMARKS_SYNTHID_SCORER_URL", "http://scorer:8766")
    monkeypatch.setattr(
        image_meta.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResp(
            {
                "available": True,
                "is_watermarked": True,
                "confidence": 0.91,
                "phase_match": 0.8,
            }
        ),
    )
    status, body = _post(
        conn,
        "/clean",
        {
            "file": _b64(_watermarked_png()),
            "name": "shot.png",
            "options": {"detect_before": True, "detect_after": True},
        },
    )
    assert status == 200
    report = body["report"]
    assert report["synthid_before"]["is_watermarked"] is True
    assert report["synthid_after"]["is_watermarked"] is True


def test_run_synthid_score_http_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("WATERMARKS_SYNTHID_SCORER_URL", "http://scorer:8766")
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        return _FakeResp({"available": True, "is_watermarked": False, "confidence": 0.1})

    monkeypatch.setattr(image_meta.urllib.request, "urlopen", fake_urlopen)
    img = tmp_path / "x.png"
    img.write_bytes(_watermarked_png())
    payload = image_meta.run_synthid_score(img)
    assert payload["available"] is True
    assert payload["is_watermarked"] is False
    assert seen["url"] == "http://scorer:8766/score"
    assert seen["timeout"] == 60.0


def test_detect_image_no_scorer(conn):
    status, body = _post(conn, "/detect", {"file": _b64(_watermarked_png()), "name": "shot.png"})
    assert status == 200
    assert body["kind"] == "image"
    assert body["detections"][0]["available"] is False


def test_detect_batch_text_files(conn):
    status, body = _post(
        conn,
        "/detect/batch",
        {
            "files": [
                {"file": _b64(b"Hello world from simple clean text."), "name": "doc1.txt"},
                {"file": _b64(b"Second document test."), "name": "doc2.txt"},
            ]
        },
    )
    assert status == 200
    assert body["ok"] is True
    assert len(body["results"]) == 2
    res1, res2 = body["results"]
    assert res1["name"] == "doc1.txt"
    assert res1["ok"] is True
    assert res1["kind"] == "text"
    assert any(d["detector"] == "stylometry" for d in res1["detections"])
    assert res2["name"] == "doc2.txt"
    assert res2["ok"] is True


def test_detect_batch_mixed_formats(conn):
    status, body = _post(
        conn,
        "/detect/batch",
        {
            "files": [
                {"file": _b64(b"Plain text note"), "name": "a.txt"},
                {"file": _b64(_watermarked_png()), "name": "b.png"},
            ]
        },
    )
    assert status == 200
    assert body["ok"] is True
    results = {r["name"]: r for r in body["results"]}
    assert results["a.txt"]["kind"] == "text"
    assert results["b.png"]["kind"] == "image"


def test_detect_batch_bad_entry_does_not_abort_others(conn):
    status, body = _post(
        conn,
        "/detect/batch",
        {
            "files": [
                {"file": "!!!not_base64!!!", "name": "bad.txt"},
                {"file": _b64(b"Valid text"), "name": "good.txt"},
            ]
        },
    )
    assert status == 200
    assert body["ok"] is True
    results = {r["name"]: r for r in body["results"]}
    assert results["bad.txt"]["ok"] is False
    assert "base64" in results["bad.txt"]["error"]
    assert results["good.txt"]["ok"] is True


def test_detect_batch_empty_rejected(conn):
    status, body = _post(conn, "/detect/batch", {"files": []})
    assert status == 400
    assert "must not be empty" in body["error"]


def test_detect_batch_openapi_spec_registered(conn):
    status, body = _get(conn, "/openapi.json")
    assert status == 200
    assert "/detect/batch" in body["paths"]
    assert "post" in body["paths"]["/detect/batch"]
