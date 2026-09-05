"""Tests for the /watermark and /watermark/batch endpoints in server.py."""

from __future__ import annotations

import base64
import http.client
import json
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import server  # type: ignore
import text_watermark  # type: ignore


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


class _FakeSidecarResp:
    def __init__(self, data: dict, status: int = 200):
        self._data = json.dumps(data).encode("utf-8")
        self.status = status

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
        "WATERMARKS_SYNTHID_TEXT_URL",
        "WATERMARKS_SYNTHID_TEXT_API_KEY",
        "WATERMARKS_SYNTHID_TEXT_TIMEOUT",
        "MARKLLM_DIR",
    ):
        monkeypatch.delenv(key, raising=False)


class _FakeOpener:
    def __init__(self, handler):
        self._handler = handler

    def open(self, req, timeout=None):
        return self._handler(req, timeout=timeout)


def test_capabilities_includes_text_generators(conn):
    status, body = _get(conn, "/capabilities")
    assert status == 200
    assert "text_generators" in body
    assert body["text_generators"]["synthid_http"] is False
    assert body["text_generators"]["markllm"] is False


def test_openapi_spec_includes_watermark_endpoints(conn):
    status, body = _get(conn, "/openapi.json")
    assert status == 200
    assert "/watermark" in body["paths"]
    assert "post" in body["paths"]["/watermark"]
    assert "/watermark/batch" in body["paths"]
    assert "post" in body["paths"]["/watermark/batch"]


def test_watermark_unconfigured_returns_explanatory_error(conn):
    status, body = _post(conn, "/watermark", {"text": "A prompt to watermark"})
    assert status == 503
    assert body["ok"] is False
    assert "WATERMARKS_SYNTHID_TEXT_URL" in body["error"]


def test_watermark_unknown_options_rejected(conn):
    status, body = _post(
        conn, "/watermark", {"text": "Hello", "options": {"bogus_option": "value"}}
    )
    assert status == 400
    assert body["ok"] is False
    assert "unsupported option" in body["error"]


def test_watermark_options_validation_bounds(conn):
    status, body = _post(conn, "/watermark", {"text": "Hello", "options": {"max_new_tokens": 0}})
    assert status == 400
    assert body["ok"] is False
    assert "max_new_tokens" in body["error"]

    status, body = _post(conn, "/watermark", {"text": "Hello", "options": {"temperature": -1.0}})
    assert status == 400
    assert body["ok"] is False
    assert "temperature" in body["error"]

    status, body = _post(conn, "/watermark", {"text": "Hello", "options": {"top_p": 1.5}})
    assert status == 400
    assert body["ok"] is False
    assert "top_p" in body["error"]


def test_watermark_missing_text_and_file_returns_bad_request(conn):
    status, body = _post(conn, "/watermark", {"options": {"scheme": "synthid"}})
    assert status == 400
    assert body["ok"] is False
    assert "must include 'text' or base64 'file'" in body["error"]


def test_watermark_binary_file_rejected(conn):
    bad_base64 = base64.b64encode(b"\x00\x01\x02\x03\x04\x00\x00\x00").decode("ascii")
    status, body = _post(conn, "/watermark", {"file": bad_base64})
    assert status == 400
    assert body["ok"] is False
    assert "binary" in body["error"]


def test_watermark_sidecar_success(conn, monkeypatch):
    captured_req = {}

    def fake_urlopen(req, timeout=None):
        captured_req["url"] = req.full_url
        captured_req["headers"] = dict(req.headers)
        captured_req["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeSidecarResp(
            {
                "ok": True,
                "kind": "text",
                "watermarked_text": "Watermarked output sample",
                "report": {
                    "scheme_used": "synthid",
                    "model": "facebook/opt-1.3b",
                    "keys_used": [118, 504, 421, 521],
                },
            }
        )

    monkeypatch.setattr(text_watermark, "OPENER", _FakeOpener(fake_urlopen))
    monkeypatch.setenv("WATERMARKS_SYNTHID_TEXT_URL", "http://wr-synthid-text:8767")
    monkeypatch.setenv("WATERMARKS_SYNTHID_TEXT_API_KEY", "secret-token")

    status, body = _post(
        conn,
        "/watermark",
        {
            "text": "Hello world prompt",
            "keys": [118, 504, 421, 521],
            "options": {"scheme": "synthid", "seed": 42},
        },
    )

    assert status == 200
    assert body["ok"] is True
    assert body["watermarked_text"] == "Watermarked output sample"
    assert body["report"]["keys_used"] == [118, 504, 421, 521]

    # Verify sidecar request details
    assert captured_req["url"] == "http://wr-synthid-text:8767/watermark"
    assert captured_req["headers"].get("Authorization") == "Bearer secret-token"
    assert captured_req["body"]["text"] == "Hello world prompt"
    assert captured_req["body"]["keys"] == [118, 504, 421, 521]
    assert captured_req["body"]["options"]["seed"] == 42


def test_watermark_sidecar_via_base64_file(conn, monkeypatch):
    raw_prompt = "Prompt from encoded file"
    b64_prompt = base64.b64encode(raw_prompt.encode("utf-8")).decode("ascii")

    def fake_urlopen(req, timeout=None):
        return _FakeSidecarResp(
            {
                "ok": True,
                "kind": "text",
                "watermarked_text": "Watermarked from file",
                "report": {"scheme_used": "synthid"},
            }
        )

    monkeypatch.setattr(text_watermark, "OPENER", _FakeOpener(fake_urlopen))
    monkeypatch.setenv("WATERMARKS_SYNTHID_TEXT_URL", "http://127.0.0.1:8767")

    status, body = _post(conn, "/watermark", {"file": b64_prompt, "name": "prompt.txt"})
    assert status == 200
    assert body["ok"] is True
    assert body["watermarked_text"] == "Watermarked from file"


def test_watermark_sidecar_unreachable_fail_soft(conn, monkeypatch):
    import urllib.error

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(text_watermark, "OPENER", _FakeOpener(fake_urlopen))
    monkeypatch.setenv("WATERMARKS_SYNTHID_TEXT_URL", "http://127.0.0.1:8767")

    status, body = _post(conn, "/watermark", {"text": "Testing error handling"})
    assert status == 502
    assert body["ok"] is False
    assert "unreachable" in body["error"].lower()


def test_watermark_batch_success(conn, monkeypatch):
    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        prompt = body.get("text", "")
        return _FakeSidecarResp(
            {
                "ok": True,
                "kind": "text",
                "watermarked_text": f"WM: {prompt}",
                "report": {"scheme_used": "synthid"},
            }
        )

    monkeypatch.setattr(text_watermark, "OPENER", _FakeOpener(fake_urlopen))
    monkeypatch.setenv("WATERMARKS_SYNTHID_TEXT_URL", "http://127.0.0.1:8767")

    batch_payload = {
        "files": [
            {"name": "p1.txt", "text": "Prompt one", "keys": [1, 2, 3]},
            {"name": "p2.txt", "text": "Prompt two", "keys": [4, 5, 6]},
        ]
    }

    status, body = _post(conn, "/watermark/batch", batch_payload)
    assert status == 200
    assert body["ok"] is True
    assert len(body["results"]) == 2
    assert body["results"][0]["name"] == "p1.txt"
    assert body["results"][0]["ok"] is True
    assert body["results"][0]["watermarked_text"] == "WM: Prompt one"
    assert body["results"][1]["name"] == "p2.txt"
    assert body["results"][1]["ok"] is True
    assert body["results"][1]["watermarked_text"] == "WM: Prompt two"


def test_watermark_batch_partial_failure_does_not_abort_batch(conn, monkeypatch):
    def fake_urlopen(req, timeout=None):
        return _FakeSidecarResp(
            {
                "ok": True,
                "kind": "text",
                "watermarked_text": "WM result",
                "report": {"scheme_used": "synthid"},
            }
        )

    monkeypatch.setattr(text_watermark, "OPENER", _FakeOpener(fake_urlopen))
    monkeypatch.setenv("WATERMARKS_SYNTHID_TEXT_URL", "http://127.0.0.1:8767")

    batch_payload = {
        "files": [
            {"name": "valid.txt", "text": "Good prompt"},
            {"name": "missing_text.txt"},  # Invalid: no text or file
            {"name": "valid2.txt", "text": "Another good prompt"},
        ]
    }

    status, body = _post(conn, "/watermark/batch", batch_payload)
    assert status == 200
    assert body["ok"] is True
    assert len(body["results"]) == 3
    assert body["results"][0]["ok"] is True
    assert body["results"][1]["ok"] is False
    assert "error" in body["results"][1]
    assert body["results"][2]["ok"] is True


def test_watermark_batch_empty_files_rejected(conn):
    status, body = _post(conn, "/watermark/batch", {"files": []})
    assert status == 400
    assert body["ok"] is False
    assert "empty" in body["error"]


def test_watermark_batch_unconfigured_returns_503(conn):
    status, body = _post(conn, "/watermark/batch", {"files": [{"text": "Hello prompt"}]})
    assert status == 503
    assert body["ok"] is False
    assert "WATERMARKS_SYNTHID_TEXT_URL" in body["error"]


def test_watermark_batch_wall_clock_timeout(conn, monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(text_watermark, "OPENER", _FakeOpener(fake_urlopen))
    monkeypatch.setenv("WATERMARKS_SYNTHID_TEXT_URL", "http://127.0.0.1:8767")
    monkeypatch.setenv("WATERMARKS_SYNTHID_TEXT_TIMEOUT", "0.01")

    batch_payload = {
        "files": [
            {"name": "p1.txt", "text": "Prompt one"},
            {"name": "p2.txt", "text": "Prompt two"},
        ]
    }
    status, body = _post(conn, "/watermark/batch", batch_payload)
    assert status == 200
    assert body["ok"] is True
    assert len(body["results"]) == 2
    assert body["results"][0]["ok"] is False
    assert body["results"][1]["ok"] is False
    err = (body["results"][0]["error"] + " " + body["results"][1]["error"]).lower()
    assert "deadline" in err or "timed out" in err


def test_watermark_batch_caps_enforced(conn, monkeypatch):
    monkeypatch.setattr(server, "MAX_BATCH_FILES", 3)
    payload = {"files": [{"text": f"Prompt {i}"} for i in range(4)]}
    status, body = _post(conn, "/watermark/batch", payload)
    assert status == 400
    assert body["ok"] is False
    assert "limit" in body["error"]


def test_watermark_http_blocks_redirect():
    """Ensure _watermark_http refuses 302 redirects to prevent SSRF and key leakage."""
    import http.server

    state: dict = {"collector_port": None}
    captured: dict = {}

    class Redirector(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(302)
            self.send_header(
                "Location",
                f"http://127.0.0.1:{state['collector_port']}/leak",
            )
            self.end_headers()

        def log_message(self, format, *args):
            pass

    class Collector(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            captured["hit"] = True
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            captured["hit"] = True
            self.send_response(200)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    collector = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    redirector = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Redirector)
    state["collector_port"] = collector.server_address[1]
    t1 = threading.Thread(target=collector.serve_forever, daemon=True)
    t2 = threading.Thread(target=redirector.serve_forever, daemon=True)
    t1.start()
    t2.start()

    try:
        res = text_watermark._watermark_http(
            "test prompt",
            f"http://127.0.0.1:{redirector.server_address[1]}",
            api_key="secret-key",
            timeout=2.0,
        )
        assert res["ok"] is False
        assert not captured.get("hit"), "Redirect was followed, leaking request/credentials!"
    finally:
        collector.shutdown()
        redirector.shutdown()
        collector.server_close()
        redirector.server_close()
        t1.join(timeout=2)
        t2.join(timeout=2)


def test_resolve_timeout(monkeypatch):
    from text_watermark import DEFAULT_SYNTHID_TEXT_TIMEOUT, resolve_timeout

    # Defaults when no explicit value or env
    monkeypatch.delenv("WATERMARKS_SYNTHID_TEXT_TIMEOUT", raising=False)
    assert resolve_timeout(None) == DEFAULT_SYNTHID_TEXT_TIMEOUT

    # Explicit clamped between floor (1.0) and ceiling (600.0)
    assert resolve_timeout(0.1) == 1.0
    assert resolve_timeout(-10.0) == 1.0
    assert resolve_timeout(9999.0) == 600.0
    assert resolve_timeout(45.5) == 45.5

    # Env variable
    monkeypatch.setenv("WATERMARKS_SYNTHID_TEXT_TIMEOUT", "120.0")
    assert resolve_timeout(None) == 120.0

    # Env variable clamped
    monkeypatch.setenv("WATERMARKS_SYNTHID_TEXT_TIMEOUT", "0.05")
    assert resolve_timeout(None) == 1.0


def test_watermark_error_status_maps_on_error_code():
    """_watermark_error_status maps on stable error_code, not error text."""
    from http import HTTPStatus

    # client_error -> 400
    assert server._watermark_error_status({"error_code": "client_error"}) == HTTPStatus.BAD_REQUEST

    # auth -> 502
    assert server._watermark_error_status({"error_code": "auth"}) == HTTPStatus.BAD_GATEWAY

    # unconfigured -> 503
    assert (
        server._watermark_error_status({"error_code": "unconfigured"})
        == HTTPStatus.SERVICE_UNAVAILABLE
    )

    # backend_error, unreachable, timeout, ssrf -> 502
    for code in ("backend_error", "unreachable", "timeout", "ssrf"):
        assert server._watermark_error_status({"error_code": code}) == HTTPStatus.BAD_GATEWAY

    # Missing/unknown error_code -> 400 fallback
    assert server._watermark_error_status({}) == HTTPStatus.BAD_REQUEST
    assert server._watermark_error_status({"error_code": "unknown"}) == HTTPStatus.BAD_REQUEST


def test_parse_watermark_options_device_config_offline():
    """Verify device, config, and offline are accepted and validated."""
    opts = text_watermark.parse_watermark_options(
        {"device": "cuda:0", "config": "my_config.json", "offline": True}
    )
    assert opts == {"device": "cuda:0", "config": "my_config.json", "offline": True}

    # device must be a non-empty string
    with pytest.raises(ValueError, match="device"):
        text_watermark.parse_watermark_options({"device": ""})
    with pytest.raises(ValueError, match="device"):
        text_watermark.parse_watermark_options({"device": 123})

    # config must be a non-empty string
    with pytest.raises(ValueError, match="config"):
        text_watermark.parse_watermark_options({"config": ""})

    # offline must be a boolean
    with pytest.raises(ValueError, match="offline"):
        text_watermark.parse_watermark_options({"offline": "yes"})
    with pytest.raises(ValueError, match="offline"):
        text_watermark.parse_watermark_options({"offline": 1})


def test_openapi_spec_preserves_502_503_descriptors(conn):
    spec = server.openapi_spec()
    resp = spec["paths"]["/watermark"]["post"]["responses"]
    # 502/503 must be preserved as full response descriptors, not wrapped
    # schemas, and not labeled "Success".
    for code in ("502", "503"):
        entry = resp[code]
        assert "description" in entry and "content" in entry
        assert entry["description"] != "Success"
        assert entry["content"]["application/json"]["schema"] is server._ERROR_SCHEMA
    # The 200 success response is still a wrapped schema labeled "Success".
    assert resp["200"]["description"] == "Success"


def test_watermark_request_schema_exclusive_minimum_is_boolean():
    schema = server._watermark_request_schema()
    opts = schema["properties"]["options"]["properties"]
    # OpenAPI 3.0.3 requires boolean exclusiveMinimum, not a numeric form.
    assert opts["temperature"] == {
        "type": "number",
        "minimum": 0,
        "exclusiveMinimum": True,
    }
    assert opts["top_p"] == {
        "type": "number",
        "minimum": 0,
        "exclusiveMinimum": True,
        "maximum": 1,
    }


def test_local_cache_key_distinguishes_default_and_empty_keys(monkeypatch, tmp_path):
    """keys=None (default) and keys=[] (empty) must use distinct local cached models."""
    import detect_text_watermark as dtw

    upstream = tmp_path / "upstream"
    upstream.mkdir()

    construction_keys: list[list[int] | None] = []

    class _FakeConfig:
        def __init__(self, keys) -> None:
            self.keys = keys
            self.gen_kwargs: dict[str, Any] = {}

    class _FakeWM:
        def __init__(self, keys) -> None:
            self.config = _FakeConfig(keys)
            self.config.gen_kwargs["max_new_tokens"] = 200
            self.config.gen_kwargs["min_length"] = 0

        def generate_watermarked_text(self, text: str) -> str:
            return f"{text}[{self.config.keys}]"

    def _fake_load(upstream, alg, config, model, device, **kwargs):
        keys = kwargs.get("watermark_keys")
        construction_keys.append(keys)
        return _FakeWM(keys)

    monkeypatch.setattr(dtw, "_load_algorithm", _fake_load)
    monkeypatch.setattr(dtw, "_resolve_config", lambda *a, **k: upstream / "config.json")
    monkeypatch.setattr(dtw, "resolve_device", lambda *a, **k: "cpu")
    monkeypatch.setattr(dtw, "resolve_upstream", lambda *a, **k: upstream)
    monkeypatch.setattr(dtw, "SCHEMES", {"synthid": "SynthID"})
    from collections import OrderedDict

    monkeypatch.setattr(text_watermark, "_LOCAL_MODEL_CACHE", OrderedDict())
    monkeypatch.setattr(text_watermark, "MAX_LOCAL_CACHED_MODELS", 4)

    default_res = text_watermark._watermark_local_markllm(
        "hello", keys=None, options={}, timeout=None, upstream_dir=str(upstream)
    )
    empty_res = text_watermark._watermark_local_markllm(
        "hello", keys=[], options={}, timeout=None, upstream_dir=str(upstream)
    )
    default_res2 = text_watermark._watermark_local_markllm(
        "hello", keys=None, options={}, timeout=None, upstream_dir=str(upstream)
    )

    assert default_res["ok"] is True
    assert default_res["watermarked_text"] == "hello[None]"
    assert empty_res["ok"] is True
    assert empty_res["watermarked_text"] == "hello[[]]"
    # Keys were applied at construction, keeping None (default) and [] distinct.
    assert construction_keys == [None, []]
    # Two distinct models were loaded; default reuse hits the cache.
    assert len(construction_keys) == 2
    assert default_res2["watermarked_text"] == "hello[None]"
