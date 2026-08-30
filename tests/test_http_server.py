"""Tests for the HTTP service entrypoint (server.py)."""

from __future__ import annotations

import base64
import http.client
import json
import re
import struct
import sys
import threading
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import container_meta
import server


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _watermarked_png() -> bytes:
    """1x1 PNG whose tEXt chunk carries C2PA markers (structure only)."""
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


def _isobmff_box(fourcc: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + fourcc + payload


def _minimal_mp4() -> bytes:
    """A minimal structurally-valid MP4: ftyp + mdat (classifies as AV)."""
    ftyp = _isobmff_box(b"ftyp", b"isom" + struct.pack(">I", 0) + b"isomiso2mp41")
    mdat = _isobmff_box(b"mdat", b"\x00" * 24)
    return ftyp + mdat


def _riff_chunk(cid: bytes, payload: bytes) -> bytes:
    pad = b"\x00" if len(payload) & 1 else b""
    return cid + struct.pack("<I", len(payload)) + payload + pad


def _minimal_wav() -> bytes:
    """A minimal mono 8-bit PCM WAV (classifies as AV/audio)."""
    samples = bytes([128] * 200)
    fmt = struct.pack("<HHIIHH", 1, 1, 8000, 8000, 1, 8)
    body = b"WAVE" + _riff_chunk(b"fmt ", fmt) + _riff_chunk(b"data", samples)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _post(conn: http.client.HTTPConnection, path: str, payload: dict) -> tuple[int, dict]:
    conn.request(
        "POST",
        path,
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    data = resp.read()
    return resp.status, json.loads(data) if data else {}


def _get(conn: http.client.HTTPConnection, path: str) -> tuple[int, dict]:
    conn.request("GET", path)
    resp = conn.getresponse()
    data = resp.read()
    return resp.status, json.loads(data) if data else {}


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


def test_health(conn):
    status, body = _get(conn, "/health")
    assert status == 200
    assert body["ok"] is True
    assert "version" in body


def test_capabilities(conn):
    status, body = _get(conn, "/capabilities")
    assert status == 200
    assert set(body["tools"]) == {"c2patool", "exiftool", "qpdf", "ghostscript", "ffmpeg"}
    assert "pixel_backends" in body
    assert "scorers" in body
    assert "harnesses" in body


def test_openapi_spec_covers_all_endpoints(conn):
    status, body = _get(conn, "/openapi.json")
    assert status == 200
    assert body["openapi"] == "3.0.3"
    assert body["info"]["title"] == "watermarks-remover service"
    expected = {
        "/health": {"get"},
        "/capabilities": {"get"},
        "/openapi.json": {"get"},
        "/inspect": {"post"},
        "/clean": {"post"},
    }
    for path, methods in expected.items():
        assert path in body["paths"]
        for method in methods:
            assert method in body["paths"][path]
            assert "responses" in body["paths"][path][method]


def test_openapi_spec_describes_request_bodies(conn):
    status, body = _get(conn, "/openapi.json")
    assert status == 200
    clean = body["paths"]["/clean"]["post"]
    assert clean["requestBody"]["required"] is True
    schema = clean["requestBody"]["content"]["application/json"]["schema"]
    assert "file" in schema["properties"]
    assert "options" in schema["properties"]
    inspect = body["paths"]["/inspect"]["post"]
    assert "file" in inspect["requestBody"]["content"]["application/json"]["schema"]["properties"]


def test_openapi_spec_reflects_auth(conn, monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "sekret")
    conn.request("GET", "/openapi.json", headers={"Authorization": "Bearer sekret"})
    resp = conn.getresponse()
    body = json.loads(resp.read())
    assert resp.status == 200
    assert "bearerAuth" in body["components"]["securitySchemes"]
    assert body["security"] == [{"bearerAuth": []}]


def test_inspect_text_finds_watermark(conn):
    data = "Hello\u200bWorld\u00ad!".encode("utf-8")
    status, body = _post(conn, "/inspect", {"file": _b64(data), "name": "note.txt"})
    assert status == 200
    assert body["kind"] == "text"
    assert body["suspicious"]["verdict"] is True
    assert body["report"]["suspicious_total"] == 2
    layer_a = body["suspicious"]["classes"]["layer_a_unicode"]
    assert layer_a["present"] is True
    assert layer_a["strength"] == "deterministic"
    assert layer_a["description"]


def test_clean_text_roundtrip(conn):
    data = "Hello\u200bWorld\u00ad!".encode("utf-8")
    status, body = _post(conn, "/clean", {"file": _b64(data), "name": "note.txt"})
    assert status == 200
    assert body["kind"] == "text"
    cleaned = base64.b64decode(body["cleaned"]).decode("utf-8")
    assert cleaned == "HelloWorld!"
    assert body["report"]["stats"]["removed_count"] == 2


def test_clean_png_strips_metadata(conn):
    data = _watermarked_png()
    status, body = _post(conn, "/clean", {"file": _b64(data), "name": "shot.png"})
    assert status == 200
    assert body["kind"] == "image"
    cleaned = base64.b64decode(body["cleaned"])
    assert b"c2pa" not in cleaned.lower()
    assert body["report"]["format"] == "png"
    assert any("tEXt" in a for a in body["report"]["actions"]) or "actions" in body["report"]


def test_clean_markdown_container(conn):
    data = (ROOT / "tests" / "fixtures" / "sample_ai.md").read_bytes()
    status, body = _post(conn, "/clean", {"file": _b64(data), "name": "note.md"})
    assert status == 200
    assert body["kind"] == "container"
    cleaned = base64.b64decode(body["cleaned"]).decode("utf-8")
    assert "generator: Claude" not in cleaned
    assert body["report"]["format"] == "markdown"


def test_clean_av_honors_remove_pixel(conn, monkeypatch):
    # No purification backend is configured in the test env, so remove_pixel on
    # the AV path must be accepted and reported as unavailable (not silently
    # ignored), while the metadata strip still runs.
    for var in ("NOAI_WATERMARK_DIR", "MARKDIFFUSION_DIR"):
        monkeypatch.delenv(var, raising=False)
    data = _minimal_mp4()
    status, body = _post(
        conn,
        "/clean",
        {"file": _b64(data), "name": "clip.mp4", "options": {"remove_pixel": "ctrlregen"}},
    )
    assert status == 200
    assert body["kind"] == "av"
    assert body["report"]["pixel_removal"]["available"] is False
    assert any("video purification" in a for a in body["report"]["actions"])
    # The cleaned payload length must match the reported bytes_out.
    assert len(base64.b64decode(body["cleaned"])) == body["report"]["bytes_out"]


def test_clean_av_honors_remove_audio_watermark(conn):
    data = _minimal_wav()
    status, body = _post(
        conn,
        "/clean",
        {"file": _b64(data), "name": "clip.wav", "options": {"remove_audio_watermark": True}},
    )
    assert status == 200
    assert body["kind"] == "av"
    assert "audio_mark_removal" in body["report"]
    assert "available" in body["report"]["audio_mark_removal"]
    assert len(base64.b64decode(body["cleaned"])) == body["report"]["bytes_out"]


def test_clean_av_m4a_reencode_dest_distinct_from_container_clean(conn, monkeypatch):
    """An .m4a input must not reuse the container-clean dest for the re-encode.

    The AV branch derives the container-clean dest as ``out<ext>`` (so ``.m4a``
    gives ``out.m4a``). If the audio chain re-used the same name, ffmpeg would
    refuse to edit its input in place and silently skip the chain. This stubs
    ``audio_purify`` so the regression is independent of whether ffmpeg is on
    the runner.
    """
    calls: list[tuple[Path, Path]] = []

    def fake_purify(src: Path, dest: Path) -> dict:
        calls.append((src, dest))
        return {"available": False, "error": "spy: ffmpeg unavailable"}

    monkeypatch.setattr(server, "audio_purify", fake_purify)
    monkeypatch.setattr(server, "media_has_video", lambda path: False)

    status, _ = _post(
        conn,
        "/clean",
        {
            "file": _b64(_minimal_mp4()),
            "name": "audio.m4a",
            "options": {"remove_audio_watermark": True},
        },
    )
    assert status == 200
    assert len(calls) == 1, "audio_purify should run once for an .m4a audio item"
    src, dest = calls[0]
    assert src != dest, "re-encode dest must be distinct from the container-clean dest"


def test_clean_av_m4a_success_uses_reencode_dest_output(conn, monkeypatch):
    """A successful purify run must become the returned report and cleaned bytes.

    The success path re-points the handler at the re-encode dest: after the
    chain writes its output there, `bytes_out`/`cleaned` must reflect that file,
    not the earlier container-clean output.
    """
    out_bytes = _minimal_mp4()

    def fake_purify(src: Path, dest: Path) -> dict:
        dest.write_bytes(out_bytes)
        return {"available": True, "tempo": 1.08, "pitch_semitones": 2.0, "codec": "aac"}

    after = SimpleNamespace(has_c2pa=False, has_ai_metadata=False, findings=[], format="mp4")
    monkeypatch.setattr(server, "audio_purify", fake_purify)
    monkeypatch.setattr(server, "media_has_video", lambda path: False)
    monkeypatch.setattr(server, "inspect_av", lambda path: after)

    status, body = _post(
        conn,
        "/clean",
        {
            "file": _b64(_minimal_mp4()),
            "name": "audio.m4a",
            "options": {"remove_audio_watermark": True},
        },
    )
    assert status == 200
    assert body["report"]["audio_mark_removal"]["available"] is True
    assert any("destructive audio watermark chain" in a for a in body["report"]["actions"])
    assert body["report"]["bytes_out"] == len(out_bytes)
    assert base64.b64decode(body["cleaned"]) == out_bytes


def test_clean_av_rejects_invalid_remove_pixel(conn):
    data = _minimal_mp4()
    status, body = _post(
        conn,
        "/clean",
        {"file": _b64(data), "name": "clip.mp4", "options": {"remove_pixel": "bogus"}},
    )
    assert status == 400
    assert "remove_pixel" in body["error"]


def test_unknown_option_rejected(conn):
    status, body = _post(
        conn, "/clean", {"file": _b64(b"x"), "name": "x.txt", "options": {"nope": 1}}
    )
    assert status == 400
    assert "unknown option" in body["error"]


def test_unknown_deep_images_mode_rejected(conn):
    """A typo must answer 400, not fall through to a mode that may recompress."""
    status, body = _post(
        conn,
        "/clean",
        {"file": _b64(b"x"), "name": "x.txt", "options": {"deep_images": "lossles"}},
    )
    assert status == 400
    assert "deep_images" in body["error"]


def test_known_deep_images_modes_accepted(conn):
    # Driven from the frozenset itself: a mode added there without a test here
    # would otherwise look covered.
    for mode in sorted(container_meta.DEEP_IMAGE_MODES):
        status, _body = _post(
            conn,
            "/clean",
            {
                "file": _b64(b"plain text\n"),
                "name": "x.txt",
                "options": {"deep_images": mode},
            },
        )
        assert status == 200, mode


@pytest.mark.parametrize(
    ("key", "value", "type_name"),
    [
        ("nfkc", "false", "boolean"),
        ("aggressive_homoglyphs", 1, "boolean"),
        ("keep_non_ai_metadata", None, "boolean"),
        ("also_layer_a_text", {}, "boolean"),
        ("strip_all_metadata", [], "boolean"),
        ("remove_pixel", False, "string"),
        ("remove_audio_watermark", "false", "boolean"),
    ],
)
def test_option_wrong_type_rejected(conn, key, value, type_name):
    status, body = _post(
        conn, "/clean", {"file": _b64(b"x"), "name": "x.txt", "options": {key: value}}
    )
    assert status == 400
    assert body["error"] == f"option '{key}' must be a {type_name}"


def test_bad_base64_rejected(conn):
    status, _body = _post(conn, "/inspect", {"file": "!!!not-base64!!!"})
    assert status == 400


def test_binary_named_as_text_rejected(conn):
    status, _body = _post(conn, "/clean", {"file": _b64(_watermarked_png()), "name": "x.txt"})
    assert status == 400


def test_clean_unknown_format_rejected(conn):
    data = b"no magic, no extension"
    status, body = _post(conn, "/clean", {"file": _b64(data), "name": "input"})
    assert status == 400
    assert "unrecognized file format" in body["error"]


def test_inspect_unknown_format_reports_kind(conn):
    data = b"no magic, no extension"
    status, body = _post(conn, "/inspect", {"file": _b64(data), "name": "input"})
    assert status == 200
    assert body["kind"] == "unknown"
    assert body["suspicious"]["verdict"] is False
    # No stylometry is computed for a non-text asset, so its signals are null
    # rather than omitted; the OpenAPI schema must permit that.
    styl = body["suspicious"]["classes"]["stylometry"]
    assert styl["signals"]["score"] is None
    assert styl["signals"]["density_tier"] is None
    assert "note" in body["report"]


def test_missing_file_field_rejected(conn):
    status, _body = _post(conn, "/inspect", {"name": "x.txt"})
    assert status == 400


def test_safe_name_sanitizes_traversal():
    assert server._safe_name("../../etc/passwd") == "passwd"
    assert server._safe_name("a/b/c/notes.md") == "notes.md"
    assert server._safe_name("..\\..\\win.txt") == "win.txt"
    assert server._safe_name("..") == "input"
    assert server._safe_name(".") == "input"
    assert server._safe_name("") == "input"
    assert server._safe_name("report.docx") == "report.docx"


def test_traversal_name_does_not_escape(conn, tmp_path):
    data = "Hello\u200bWorld!".encode("utf-8")
    status, body = _post(conn, "/clean", {"file": _b64(data), "name": "../../escape.txt"})
    assert status == 200
    assert body["kind"] == "text"
    cleaned = base64.b64decode(body["cleaned"]).decode("utf-8")
    assert cleaned == "HelloWorld!"
    assert not (tmp_path / "escape.txt").exists()


def test_oversized_body_413(conn, monkeypatch):
    monkeypatch.setattr(server, "MAX_BODY_BYTES", 64)
    data = "x" * 200
    status, _body = _post(conn, "/inspect", {"file": _b64(data.encode())})
    assert status == 413


def test_auth_required(conn, monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "sekret")
    status, _ = _post(conn, "/health", {"anything": 1})
    assert status == 401
    conn.request("GET", "/health", headers={"Authorization": "Bearer sekret"})
    resp = conn.getresponse()
    assert resp.status == 200
    resp.read()


def test_404(conn):
    status, _body = _get(conn, "/nope")
    assert status == 404
    status, _body = _post(conn, "/nope", {"file": _b64(b"x")})
    assert status == 404


def test_openapi_spec_covers_batch_endpoints(conn):
    status, body = _get(conn, "/openapi.json")
    assert status == 200
    assert "/inspect/batch" in body["paths"]
    assert "post" in body["paths"]["/inspect/batch"]
    assert "/clean/batch" in body["paths"]
    assert "post" in body["paths"]["/clean/batch"]


def test_inspect_batch_mixed_results(conn):
    watermarked = ("Hello" + chr(0x200B) + "World!").encode("utf-8")
    clean = b"nothing to see here"
    status, body = _post(
        conn,
        "/inspect/batch",
        {
            "files": [
                {"file": _b64(watermarked), "name": "a.txt"},
                {"file": _b64(clean), "name": "b.txt"},
            ]
        },
    )
    assert status == 200
    assert body["ok"] is True
    results = {r["name"]: r for r in body["results"]}
    assert results["a.txt"]["ok"] is True
    assert results["a.txt"]["suspicious"]["verdict"] is True
    assert results["b.txt"]["ok"] is True
    assert results["b.txt"]["suspicious"]["verdict"] is False


def test_clean_batch_mixed_results(conn):
    watermarked = ("Hello" + chr(0x200B) + "World!").encode("utf-8")
    status, body = _post(
        conn,
        "/clean/batch",
        {
            "files": [
                {"file": _b64(watermarked), "name": "a.txt"},
                {"file": _b64(b"x"), "name": "b.unknownext"},
            ]
        },
    )
    assert status == 200
    assert body["ok"] is True
    results = {r["name"]: r for r in body["results"]}
    assert results["a.txt"]["ok"] is True
    cleaned = base64.b64decode(results["a.txt"]["cleaned"]).decode("utf-8")
    assert cleaned == "HelloWorld!"
    assert results["b.unknownext"]["ok"] is False
    assert "unrecognized file format" in results["b.unknownext"]["error"]


def test_batch_one_bad_option_does_not_abort_others(conn):
    status, body = _post(
        conn,
        "/clean/batch",
        {
            "files": [
                {"file": _b64(b"hello"), "name": "a.txt", "options": {"nope": 1}},
                {"file": _b64(b"hello"), "name": "b.txt"},
            ]
        },
    )
    assert status == 200
    results = {r["name"]: r for r in body["results"]}
    assert results["a.txt"]["ok"] is False
    assert "unknown option" in results["a.txt"]["error"]
    assert results["b.txt"]["ok"] is True


def test_batch_empty_files_rejected(conn):
    status, body = _post(conn, "/inspect/batch", {"files": []})
    assert status == 400
    assert "must not be empty" in body["error"]


def test_batch_missing_files_field_rejected(conn):
    status, body = _post(conn, "/clean/batch", {})
    assert status == 400
    assert "files" in body["error"]


def test_batch_over_limit_rejected(conn, monkeypatch):
    monkeypatch.setattr(server, "MAX_BATCH_FILES", 2)
    status, body = _post(
        conn,
        "/inspect/batch",
        {"files": [{"file": _b64(b"x"), "name": "a.txt"}] * 3},
    )
    assert status == 400
    assert "batch limit" in body["error"]


def test_clean_jpeg_preserves_format_extension(conn):
    # Minimal valid JPEG bytes (SOI + APP0 + EOI)
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
    status, body = _post(
        conn,
        "/clean",
        {"file": _b64(jpeg_bytes), "name": "photo.jpg"},
    )
    assert status == 200
    assert body["ok"] is True
    assert body["kind"] == "image"
    assert body["report"]["format"] == "jpeg"
    cleaned = base64.b64decode(body["cleaned"])
    assert cleaned[:2] == b"\xff\xd8"


def test_clean_extensionless_svg_container_post_inspection(conn):
    svg_bytes = (
        b'<svg xmlns="http://www.w3.org/2000/svg"><metadata>c2pa test</metadata><rect/></svg>'
    )
    status, body = _post(
        conn,
        "/clean",
        {"file": _b64(svg_bytes)},
    )
    assert status == 200
    assert body["ok"] is True
    assert body["kind"] == "container"
    assert body["report"]["format"] == "svg"
    assert body["report"]["still_has_c2pa"] is False


def test_log_message_includes_readable_timestamp(monkeypatch):
    """Request logs are prefixed with local time plus UTC offset."""
    lines: list[str] = []
    monkeypatch.setattr(server, "eprint", lines.append)
    monkeypatch.setattr(server.Handler, "address_string", lambda self: "127.0.0.1")

    handler = server.Handler.__new__(server.Handler)
    handler.log_message('"%s" %s -', "GET /health HTTP/1.1", 200)

    assert len(lines) == 1
    assert re.match(
        r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [+-]\d{4}\] 127\.0\.0\.1 - ",
        lines[0],
    )
    assert lines[0].endswith('"GET /health HTTP/1.1" 200 -')
