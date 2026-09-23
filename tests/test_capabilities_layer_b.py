"""/capabilities says up front whether text /clean can run its Layer B rewrite.

Text /clean always runs a Layer B strategy (by default "paraphrase@0.8,mlm@0.2")
and answers 400 when a step can't run, but /capabilities had no field for the
rewrite backend or the mlm tactic: an agent only learned the default strategy
was broken by sending text and reading the 400. `layer_b` reports each tactic's
readiness from the same checks the /clean gate applies, next to the existing
keys, which stay as they were.
"""

from __future__ import annotations

import base64
import http.client
import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rewrite_text
import server

# Captured at import, before conftest's autouse fixtures swap them for stubs.
_REAL_APPLY_LAYER_B = server._apply_layer_b
_REAL_MLM_IMPORT_ERROR = server._mlm_import_error

DEFAULT = "paraphrase@0.8,mlm@0.2"
PIL_ERROR = (
    "RuntimeError: Failed to import transformers.pipelines because of the following "
    "error (look up to see its traceback): No module named 'PIL'"
)
_REWRITE_ENV = (
    "WATERMARKS_REWRITE_BACKEND",
    "WATERMARKS_REWRITE_MODEL",
    "WATERMARKS_REWRITE_BASE_URL",
    "WATERMARKS_REWRITE_API_KEY",
    "WATERMARKS_REWRITE_ALLOW_REMOTE",
)


@pytest.fixture
def no_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _REWRITE_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def ollama(monkeypatch: pytest.MonkeyPatch, no_backend) -> None:
    """A loopback Ollama backend, the reporter's configuration."""
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "ollama")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "llama3.2")
    monkeypatch.setenv("WATERMARKS_REWRITE_BASE_URL", "http://127.0.0.1:11434")


def _mlm_probe(monkeypatch: pytest.MonkeyPatch, error: str | None) -> None:
    monkeypatch.setattr(server, "_mlm_import_error", lambda: error)


def _layer_b(monkeypatch: pytest.MonkeyPatch, default: str | None, mlm_error: str | None):
    monkeypatch.setattr(server, "_DEFAULT_STRATEGY", default)
    _mlm_probe(monkeypatch, mlm_error)
    return server.capabilities()["layer_b"]


# --- the report ---------------------------------------------------------------


def test_default_strategy_usable_when_backend_and_mlm_are_ready(monkeypatch, ollama):
    layer_b = _layer_b(monkeypatch, DEFAULT, None)
    assert layer_b["default_strategy"] == DEFAULT
    assert layer_b["default_strategy_usable"] is True
    assert all(layer_b["tactics"].values())
    assert layer_b["rewrite_backend"] == {"backend": "ollama", "configured": True, "error": None}
    assert layer_b["mlm"]["importable"] is True
    assert layer_b["mlm"]["error"] is None


def test_broken_mlm_stack_marks_the_default_strategy_unusable(monkeypatch, ollama):
    """The reported host: backend configured, transformers can't import without PIL."""
    layer_b = _layer_b(monkeypatch, DEFAULT, PIL_ERROR)
    assert layer_b["default_strategy_usable"] is False
    assert layer_b["tactics"]["mlm"] is False
    # What the reporter fell back to by hand: an LLM-only override still runs.
    assert layer_b["tactics"]["paraphrase"] is True
    assert layer_b["mlm"] == {
        "importable": False,
        "model": "roberta-large",
        "requirements": "service/scripts/requirements-mlm.txt",
        "error": PIL_ERROR,
    }


def test_unconfigured_backend_disables_every_llm_tactic(monkeypatch, no_backend):
    layer_b = _layer_b(monkeypatch, DEFAULT, None)
    assert layer_b["default_strategy_usable"] is False
    assert layer_b["rewrite_backend"]["backend"] == "print-prompt"
    assert layer_b["rewrite_backend"]["configured"] is False
    assert "WATERMARKS_REWRITE_BACKEND" in layer_b["rewrite_backend"]["error"]
    assert {t for t, ok in layer_b["tactics"].items() if ok} == {"mlm"}


def test_mlm_only_default_needs_no_llm_backend(monkeypatch, no_backend):
    assert _layer_b(monkeypatch, "mlm@0.3", None)["default_strategy_usable"] is True


def test_missing_default_strategy_is_reported_unusable(monkeypatch, ollama):
    layer_b = _layer_b(monkeypatch, None, None)
    assert layer_b["default_strategy"] is None
    assert layer_b["default_strategy_usable"] is False


def test_every_strategy_tactic_is_listed(monkeypatch, ollama):
    assert set(_layer_b(monkeypatch, DEFAULT, None)["tactics"]) == set(rewrite_text.KNOWN_TACTICS)


def test_openai_backend_needs_its_api_key(monkeypatch, no_backend):
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "openai-compatible")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "m")
    monkeypatch.setenv("WATERMARKS_REWRITE_BASE_URL", "http://127.0.0.1:8000")
    backend = _layer_b(monkeypatch, DEFAULT, None)["rewrite_backend"]
    assert backend["configured"] is False
    assert "API_KEY" in backend["error"]


def test_remote_endpoint_without_opt_in_is_not_configured(monkeypatch, no_backend):
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "openai-compatible")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "m")
    monkeypatch.setenv("WATERMARKS_REWRITE_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("WATERMARKS_REWRITE_API_KEY", "k")
    backend = _layer_b(monkeypatch, DEFAULT, None)["rewrite_backend"]
    assert backend["configured"] is False
    assert "WATERMARKS_REWRITE_ALLOW_REMOTE" in backend["error"]


def test_capabilities_never_echo_rewrite_secrets(monkeypatch, no_backend):
    # /capabilities needs no auth when no API key is set, so it must not leak
    # the rewrite key or an internal endpoint.
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "openai-compatible")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "m")
    monkeypatch.setenv("WATERMARKS_REWRITE_BASE_URL", "https://llm.internal.example/v1")
    monkeypatch.setenv("WATERMARKS_REWRITE_API_KEY", "sk-do-not-leak")
    monkeypatch.setenv("WATERMARKS_REWRITE_ALLOW_REMOTE", "1")
    _layer_b(monkeypatch, DEFAULT, None)
    dumped = json.dumps(server.capabilities())
    assert "sk-do-not-leak" not in dumped
    assert "llm.internal.example" not in dumped


def test_existing_capability_keys_are_unchanged(monkeypatch):
    _mlm_probe(monkeypatch, None)
    caps = server.capabilities()
    assert set(caps) == {
        "version",
        "tools",
        "pixel_backends",
        "scorers",
        "text_detectors",
        "text_generators",
        "harnesses",
        "layer_b",
    }
    assert set(caps["tools"]) == {"c2patool", "exiftool", "qpdf", "ghostscript", "ffmpeg"}
    assert set(caps["pixel_backends"]) == {"ctrlregen", "diffusion"}
    assert set(caps["scorers"]) == {"synthid", "synthid_http", "stylometry"}
    assert set(caps["text_generators"]) == {"synthid_http", "markllm"}
    assert set(caps["harnesses"]) == {"markllm"}
    for group in ("tools", "pixel_backends", "scorers", "text_detectors", "text_generators"):
        assert all(isinstance(v, bool) for v in caps[group].values()), group


def test_mlm_probe_runs_once_per_process(monkeypatch):
    calls: list[int] = []

    def probe() -> str:
        calls.append(1)
        return PIL_ERROR

    monkeypatch.setattr(rewrite_text, "mlm_import_error", probe)
    _REAL_MLM_IMPORT_ERROR.cache_clear()
    try:
        assert _REAL_MLM_IMPORT_ERROR() == PIL_ERROR
        assert _REAL_MLM_IMPORT_ERROR() == PIL_ERROR
    finally:
        _REAL_MLM_IMPORT_ERROR.cache_clear()
    assert calls == [1]


# --- the OpenAPI contract -----------------------------------------------------


def _capabilities_schema() -> dict:
    op = server.openapi_spec()["paths"]["/capabilities"]["get"]
    return op["responses"]["200"]["content"]["application/json"]["schema"]


def test_openapi_documents_layer_b_next_to_the_existing_keys():
    props = _capabilities_schema()["properties"]
    for key in ("ok", "version", "tools", "pixel_backends", "scorers", "harnesses"):
        assert key in props
    assert set(props["text_generators"]["properties"]) == {"synthid_http", "markllm"}
    layer_b = props["layer_b"]
    assert layer_b["type"] == "object"
    assert layer_b["properties"]["tactics"]["additionalProperties"] == {"type": "boolean"}
    assert layer_b["properties"]["mlm"]["properties"]["error"]["nullable"] is True


def test_openapi_layer_b_schema_matches_the_payload(monkeypatch, ollama):
    """Every key the endpoint emits is documented, and nothing documented is missing."""
    payload = _layer_b(monkeypatch, DEFAULT, PIL_ERROR)
    schema = _capabilities_schema()["properties"]["layer_b"]["properties"]
    assert set(payload) == set(schema)
    for nested in ("rewrite_backend", "mlm"):
        assert set(payload[nested]) == set(schema[nested]["properties"]), nested


def test_openapi_spec_still_validates():
    validator = pytest.importorskip("openapi_spec_validator")
    validator.validate(server.openapi_spec())


# --- over HTTP: the report predicts the /clean outcome -------------------------


@pytest.fixture
def conn():
    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=30)
    yield c
    c.close()
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _request(conn, method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    conn.request(method, path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read())


def test_agent_can_predict_the_default_strategy_400(conn, monkeypatch, ollama):
    """The reported sequence, end to end over HTTP with the real Layer B gate.

    /capabilities flags the default strategy before any text is sent, and the
    /clean that follows is rejected before its paraphrase step is paid for.
    """
    monkeypatch.setattr(server, "_apply_layer_b", _REAL_APPLY_LAYER_B)
    monkeypatch.setattr(server, "_DEFAULT_STRATEGY", DEFAULT)
    _mlm_probe(monkeypatch, PIL_ERROR)
    # transformers counts as installed, as on the reporter's host...
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: object() if name == "transformers" else real_find_spec(name, *a, **k),
    )

    # ...but its pipelines do not import.
    def broken_mlm():
        raise RuntimeError(
            "mlm tactic unavailable: Failed to import transformers.pipelines: "
            "No module named 'PIL' (install service/scripts/requirements-mlm.txt)"
        )

    monkeypatch.setattr(rewrite_text, "_get_mlm", broken_mlm)
    llm_calls: list[str] = []
    monkeypatch.setattr(
        rewrite_text, "_generate_once", lambda *args, **kwargs: llm_calls.append("llm") or "x"
    )

    status, caps = _request(conn, "GET", "/capabilities")
    assert status == 200
    assert caps["layer_b"]["default_strategy_usable"] is False
    assert caps["layer_b"]["tactics"]["mlm"] is False

    text = b"The weather was mild and the meeting ended early.\n"
    status, body = _request(
        conn, "POST", "/clean", {"file": base64.b64encode(text).decode(), "name": "probe.txt"}
    )
    assert status == 400
    assert body["error"].startswith("Layer B rewrite failed: mlm tactic unavailable")
    assert "No module named 'PIL'" in body["error"]
    assert "requirements-mlm.txt" in body["error"]
    assert llm_calls == []
