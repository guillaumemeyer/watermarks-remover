"""Unit tests for the SynthID text sidecar server (synthid_text_server.py)."""

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

import synthid_text_server  # type: ignore


def _post(
    conn: http.client.HTTPConnection, path: str, payload: dict, headers: dict | None = None
) -> tuple[int, dict]:
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    conn.request(
        "POST",
        path,
        body=json.dumps(payload).encode("utf-8"),
        headers=h,
    )
    resp = conn.getresponse()
    data = resp.read()
    return resp.status, json.loads(data) if data else {}


def _get(
    conn: http.client.HTTPConnection, path: str, headers: dict | None = None
) -> tuple[int, dict]:
    h = {}
    if headers:
        h.update(headers)
    conn.request("GET", path, headers=h)
    resp = conn.getresponse()
    data = resp.read()
    return resp.status, json.loads(data) if data else {}


def _fake_generate(
    text: str,
    keys: list[int] | None,
    options: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    opts = options or {}
    scheme = opts.get("scheme", "synthid")
    model_name = opts.get("model", synthid_text_server.DEFAULT_MODEL)
    key_repr = f" [keys:{','.join(map(str, keys))}]" if keys else ""
    watermarked = f"{text}{key_repr}"
    report = {
        "scheme_used": scheme,
        "model": model_name,
        "watermarked_chars": len(watermarked),
        "keys_used": keys,
    }
    return watermarked, report


@pytest.fixture
def sidecar_server(monkeypatch):
    monkeypatch.setattr(synthid_text_server, "API_KEY", "")
    monkeypatch.setattr(synthid_text_server, "_generate_watermarked_sample", _fake_generate)
    srv = synthid_text_server.ThreadingHTTPServer(("127.0.0.1", 0), synthid_text_server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
    yield c, srv
    c.close()
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def test_sidecar_health(sidecar_server):
    conn, _ = sidecar_server
    status, body = _get(conn, "/health")
    assert status == 200
    assert body["ok"] is True
    assert "version" in body


def test_sidecar_auth_enforced(monkeypatch):
    monkeypatch.setattr(synthid_text_server, "API_KEY", "sidecar-secret")
    srv = synthid_text_server.ThreadingHTTPServer(("127.0.0.1", 0), synthid_text_server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
    try:
        # Without auth -> 401
        status, body = _get(conn, "/health")
        assert status == 401
        assert body["error"] == "unauthorized"

        # With wrong token -> 401
        status, body = _get(conn, "/health", headers={"Authorization": "Bearer wrong"})
        assert status == 401

        # With correct token -> 200
        status, body = _get(conn, "/health", headers={"Authorization": "Bearer sidecar-secret"})
        assert status == 200
        assert body["ok"] is True
    finally:
        conn.close()
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def test_sidecar_watermark_single(sidecar_server):
    conn, _ = sidecar_server
    status, body = _post(
        conn,
        "/watermark",
        {
            "text": "Testing single prompt",
            "keys": [118, 504, 421, 521],
            "options": {"scheme": "synthid", "seed": 42},
        },
    )
    assert status == 200
    assert body["ok"] is True
    assert body["kind"] == "text"
    assert "watermarked_text" in body
    assert body["report"]["keys_used"] == [118, 504, 421, 521]


def test_sidecar_watermark_batch(sidecar_server):
    conn, _ = sidecar_server
    status, body = _post(
        conn,
        "/watermark/batch",
        {
            "files": [
                {"name": "f1.txt", "text": "Prompt 1", "keys": [10, 20]},
                {"name": "f2.txt", "text": "Prompt 2"},
            ]
        },
    )
    assert status == 200
    assert body["ok"] is True
    assert len(body["results"]) == 2
    assert body["results"][0]["name"] == "f1.txt"
    assert body["results"][0]["ok"] is True
    assert body["results"][1]["name"] == "f2.txt"
    assert body["results"][1]["ok"] is True


def test_sidecar_watermark_base64_file(sidecar_server):
    conn, _ = sidecar_server
    raw_content = "Prompt from encoded file payload"
    b64_file = base64.b64encode(raw_content.encode("utf-8")).decode("ascii")
    status, body = _post(
        conn,
        "/watermark",
        {"file": b64_file, "keys": [10, 20]},
    )
    assert status == 200
    assert body["ok"] is True
    assert raw_content in body["watermarked_text"]
    assert body["report"]["keys_used"] == [10, 20]


def test_sidecar_watermark_non_string_file(sidecar_server):
    conn, _ = sidecar_server
    status, body = _post(
        conn,
        "/watermark",
        {"file": 12345},
    )
    assert status == 400
    assert body["ok"] is False
    assert "'file' must be a base64-encoded string" in body["error"]


def test_sidecar_watermark_batch_limit_exceeded(sidecar_server):
    conn, _ = sidecar_server
    status, body = _post(
        conn,
        "/watermark/batch",
        {
            "files": [
                {"name": f"f{i}.txt", "text": f"Prompt {i}"}
                for i in range(synthid_text_server.MAX_BATCH_FILES + 1)
            ]
        },
    )
    assert status == 400
    assert body["ok"] is False
    assert f"exceeds the {synthid_text_server.MAX_BATCH_FILES}-file batch limit" in body["error"]


def test_sidecar_watermark_input_cap_text(sidecar_server, monkeypatch):
    conn, _ = sidecar_server
    monkeypatch.setattr(synthid_text_server, "MAX_INPUT_BYTES", 20)
    status, body = _post(
        conn,
        "/watermark",
        {"text": "A" * 25},
    )
    assert status == 400
    assert body["ok"] is False
    assert "input size cap" in body["error"]


def test_sidecar_watermark_input_cap_decoded_file(sidecar_server, monkeypatch):
    conn, _ = sidecar_server
    monkeypatch.setattr(synthid_text_server, "MAX_INPUT_BYTES", 20)
    b64_file = base64.b64encode(b"B" * 25).decode("ascii")
    status, body = _post(
        conn,
        "/watermark",
        {"file": b64_file},
    )
    assert status == 400
    assert body["ok"] is False
    assert "input size cap" in body["error"]


def test_sidecar_watermark_invalid_scheme(sidecar_server):
    conn, _ = sidecar_server
    status, body = _post(
        conn,
        "/watermark",
        {
            "text": "Prompt text",
            "options": {"scheme": True},
        },
    )
    assert status == 400
    assert body["ok"] is False
    assert "'scheme' must be a non-empty string" in body["error"]


def test_sidecar_keys_rejects_nested_json(sidecar_server):
    conn, _ = sidecar_server
    status, body = _post(
        conn,
        "/watermark",
        {
            "text": "Prompt text",
            "keys": "[[1, 2]]",
        },
    )
    assert status == 400
    assert body["ok"] is False


def test_sidecar_watermark_missing_markllm_returns_500(monkeypatch, tmp_path):
    monkeypatch.setattr(synthid_text_server, "API_KEY", "")
    monkeypatch.setattr(synthid_text_server, "MARKLLM_DIR", str(tmp_path / "nonexistent"))
    srv = synthid_text_server.ThreadingHTTPServer(("127.0.0.1", 0), synthid_text_server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
    try:
        status, body = _post(conn, "/watermark", {"text": "hello prompt"})
        assert status == 500
        assert body["ok"] is False
        assert body["error"] == "watermark generation failed"
    finally:
        conn.close()
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def test_sidecar_watermark_batch_missing_markllm_returns_generation_error(monkeypatch, tmp_path):
    monkeypatch.setattr(synthid_text_server, "API_KEY", "")
    monkeypatch.setattr(synthid_text_server, "MARKLLM_DIR", str(tmp_path / "nonexistent"))
    srv = synthid_text_server.ThreadingHTTPServer(("127.0.0.1", 0), synthid_text_server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
    try:
        status, body = _post(
            conn, "/watermark/batch", {"files": [{"name": "f.txt", "text": "hello prompt"}]}
        )
        assert status == 200
        assert body["ok"] is True
        assert len(body["results"]) == 1
        assert body["results"][0]["ok"] is False
        assert body["results"][0]["error"] == "generation error"
    finally:
        conn.close()
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def test_sidecar_rejects_undocumented_options(sidecar_server):
    conn, _ = sidecar_server
    status, body = _post(
        conn,
        "/watermark",
        {
            "text": "Prompt text",
            "options": {"bogus_option": "value"},
        },
    )
    assert status == 400
    assert body["ok"] is False
    assert "unsupported option" in body["error"]


def test_sidecar_model_cache_bounded(monkeypatch):
    monkeypatch.setattr(synthid_text_server, "MAX_CACHED_MODELS", 2)
    synthid_text_server._MODEL_CACHE.clear()
    with synthid_text_server._MODEL_LOCK:
        synthid_text_server._MODEL_CACHE["m1"] = "val1"
        synthid_text_server._MODEL_CACHE["m2"] = "val2"
        synthid_text_server._MODEL_CACHE.move_to_end("m1")
        if len(synthid_text_server._MODEL_CACHE) >= 2:
            synthid_text_server._MODEL_CACHE.popitem(last=False)
        synthid_text_server._MODEL_CACHE["m3"] = "val3"
        assert "m2" not in synthid_text_server._MODEL_CACHE
        assert "m1" in synthid_text_server._MODEL_CACHE
        assert "m3" in synthid_text_server._MODEL_CACHE


def test_sidecar_generator_missing_markllm_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(synthid_text_server, "MARKLLM_DIR", str(tmp_path / "nonexistent"))
    with pytest.raises(RuntimeError, match="MarkLLM upstream directory not found"):
        synthid_text_server._generate_watermarked_sample("test prompt", None, {})


def test_sidecar_not_found(sidecar_server):
    conn, _ = sidecar_server
    status, body = _get(conn, "/unknown_route")
    assert status == 404
    assert body["ok"] is False


def test_sidecar_invalid_body(sidecar_server):
    conn, _ = sidecar_server
    conn.request(
        "POST",
        "/watermark",
        body=b"not json",
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    data = resp.read()
    body = json.loads(data)
    assert resp.status == 400
    assert body["ok"] is False


def test_sidecar_invalid_options(sidecar_server):
    conn, _ = sidecar_server
    status, body = _post(
        conn,
        "/watermark",
        {
            "text": "Prompt text",
            "options": {"max_new_tokens": "not-an-int"},
        },
    )
    assert status == 400
    assert body["ok"] is False
    assert "max_new_tokens" in body["error"]


def test_sidecar_keys_rejects_bool(sidecar_server):
    conn, _ = sidecar_server
    status, body = _post(
        conn,
        "/watermark",
        {
            "text": "Prompt text",
            "keys": [True, 123],
        },
    )
    assert status == 400
    assert body["ok"] is False
    assert "boolean" in body["error"]


def test_sidecar_non_utf8_file(sidecar_server):
    conn, _ = sidecar_server
    # Non-UTF8 byte sequence e.g. 0xFF 0xFE
    raw_invalid_utf8 = b"\xff\xfe\xfa"
    b64_invalid = base64.b64encode(raw_invalid_utf8).decode("ascii")
    status, body = _post(
        conn,
        "/watermark",
        {"file": b64_invalid},
    )
    assert status == 400
    assert body["ok"] is False
    assert "not valid UTF-8" in body["error"]


def test_sidecar_cache_key_distinguishes_default_and_empty_keys(monkeypatch, tmp_path):
    """keys=None (default scheme keys) and keys=[] must be applied at model construction."""
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

        def generate_watermarked_text(self, text: str) -> str:
            return f"{text}[keys:{self.config.keys}]"

    def _fake_load(upstream, alg, config, model, device, **kwargs):
        keys = kwargs.get("watermark_keys")
        construction_keys.append(keys)
        return _FakeWM(keys)

    def _fake_generate(wm, text, **kwargs):
        return f"{text}[keys:{wm.config.keys}]", {}

    monkeypatch.setattr(dtw, "_load_algorithm", _fake_load)
    monkeypatch.setattr(dtw, "_resolve_config", lambda *a, **k: upstream / "config.json")
    monkeypatch.setattr(dtw, "resolve_device", lambda *a, **k: "cpu")
    monkeypatch.setattr(dtw, "_generate", _fake_generate)
    monkeypatch.setattr(dtw, "SCHEMES", {"synthid": "SynthID"})
    monkeypatch.setattr(synthid_text_server, "MARKLLM_DIR", str(upstream))
    synthid_text_server._MODEL_CACHE.clear()

    default_out, _ = synthid_text_server._generate_watermarked_sample("hello", None, {})
    empty_out, _ = synthid_text_server._generate_watermarked_sample("hello", [], {})
    default_out2, _ = synthid_text_server._generate_watermarked_sample("hello", None, {})

    # Construction-time keys: None (default) and [] (empty) stay distinct.
    assert construction_keys == [None, []]
    # The model built with the default scheme keys is used for keys=None.
    assert default_out == "hello[keys:None]"
    assert default_out2 == "hello[keys:None]"
    # The separately-cached empty-keys model is used for keys=[].
    assert empty_out == "hello[keys:[]]"
    # Two distinct models were constructed and cached separately.
    assert len(construction_keys) == 2
    assert len(synthid_text_server._MODEL_CACHE) == 2
    # Reusing default keys hits the cache (no third construction).
    assert len(construction_keys) == 2


def test_load_algorithm_overrides_keys_before_construction(monkeypatch, tmp_path):
    """Custom SynthID keys must be baked into the config before model construction."""
    import types

    import detect_text_watermark as dtw

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    cfg = upstream / "config" / "SynthID.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"ngram_len": 4, "keys": [111, 222]}), "utf-8")

    captured: dict[str, object] = {}

    class _DummyLM:
        def to(self, device):
            return self

    class _AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model, **kw):
            return _DummyLM()

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, model, **kw):
            return object()

    class _TransformersConfig:
        def __init__(self, **kw):
            pass

    class _AutoWatermark:
        @staticmethod
        def load(algorithm_name, algorithm_config=None, transformers_config=None, **kw):
            path = Path(algorithm_config)
            captured["keys"] = json.loads(path.read_text("utf-8"))["keys"]
            captured["different_file"] = str(path) != str(cfg)
            return object()

    tf_mod = types.ModuleType("transformers")
    tf_mod.AutoModelForCausalLM = _AutoModelForCausalLM
    tf_mod.AutoTokenizer = _AutoTokenizer
    utils_pkg = types.ModuleType("utils")
    utils_tc = types.ModuleType("utils.transformers_config")
    utils_tc.TransformersConfig = _TransformersConfig
    utils_pkg.transformers_config = utils_tc
    watermark_pkg = types.ModuleType("watermark")
    auto_wm = types.ModuleType("watermark.auto_watermark")
    auto_wm.AutoWatermark = _AutoWatermark
    watermark_pkg.auto_watermark = auto_wm

    monkeypatch.setitem(sys.modules, "transformers", tf_mod)
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "utils.transformers_config", utils_tc)
    monkeypatch.setitem(sys.modules, "watermark", watermark_pkg)
    monkeypatch.setitem(sys.modules, "watermark.auto_watermark", auto_wm)

    dtw._load_algorithm(
        upstream,
        "SynthID",
        cfg,
        "facebook/opt-1.3b",
        "cpu",
        watermark_keys=[7, 8, 9],
    )

    # AutoWatermark.load received an override config with the request keys, not
    # the config file's defaults, and a distinct temp file.
    assert captured["keys"] == [7, 8, 9]
    assert captured["different_file"] is True
