"""Tests for Layer B strategy parsing/application and /clean wiring."""

from __future__ import annotations

import base64
import http.client
import http.server
import json
import sys
import threading
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rewrite_text
import server
from rewrite_text import apply_strategy, parse_strategy
from server import _apply_layer_b, _load_default_strategy, _parse_clean_options

# --- parse_strategy ---------------------------------------------------------


def test_parse_strategy_valid():
    assert parse_strategy("paraphrase@0.8,mlm@0.2") == [("paraphrase", 0.8), ("mlm", 0.2)]


def test_parse_strategy_single_mlm():
    assert parse_strategy("mlm@0.5") == [("mlm", 0.5)]


def test_parse_strategy_unknown_tactic():
    with pytest.raises(ValueError):
        parse_strategy("nope@0.2")


def test_parse_strategy_bad_intensity():
    with pytest.raises(ValueError):
        parse_strategy("paraphrase@1.5")


def test_parse_strategy_missing_at():
    with pytest.raises(ValueError):
        parse_strategy("paraphrase")


def test_parse_strategy_empty():
    with pytest.raises(ValueError):
        parse_strategy("")


# --- apply_strategy ---------------------------------------------------------


def test_apply_strategy_sequentially(monkeypatch):
    calls: list[str] = []

    def fake_gen(backend, base_url, model, api_key, prompt, timeout, temperature, reasoning_effort):
        calls.append(prompt)
        return "PARAPHRASED "

    monkeypatch.setattr(rewrite_text, "_generate_once", fake_gen)
    monkeypatch.setattr(rewrite_text, "_mlm_infill", lambda text, level: text + " [mlm]")

    out, stats = apply_strategy(
        "Hello world.",
        [("paraphrase", 0.8), ("mlm", 0.2)],
        backend="openai-compatible",
        model="m",
        base_url="https://example.test",
        api_key="k",
    )
    assert out == "PARAPHRASED  [mlm]"
    assert len(calls) == 1  # LLM step ran once
    assert stats["strategy"] == ["paraphrase@0.8", "mlm@0.2"]
    assert stats["steps"][0]["tactic"] == "paraphrase"
    assert stats["steps"][0]["intensity"] == 0.8
    assert stats["steps"][1]["tactic"] == "mlm"
    assert stats["steps"][1]["intensity"] == 0.2
    assert stats["input_chars"] == len("Hello world.")
    assert stats["output_chars"] == len(out)


def test_apply_strategy_llm_requires_config(monkeypatch):
    monkeypatch.setattr(rewrite_text, "_generate_once", lambda *a, **k: "x")
    with pytest.raises(RuntimeError):
        apply_strategy(
            "x",
            [("paraphrase", 0.8)],
            backend="openai-compatible",
            model=None,
            base_url=None,
            api_key=None,
        )


def test_apply_strategy_mlm_only(monkeypatch):
    monkeypatch.setattr(rewrite_text, "_mlm_infill", lambda text, level: text + " edited")
    out, stats = apply_strategy(
        "hi", [("mlm", 0.3)], backend="openai-compatible", model=None, base_url=None, api_key=None
    )
    assert out == "hi edited"
    assert stats["steps"][0]["tactic"] == "mlm"


# --- server: option validation ----------------------------------------------


def test_clean_options_strategy_valid():
    opts = _parse_clean_options({"strategy": "paraphrase@0.8,mlm@0.2", "nfkc": True})
    assert opts["strategy"] == "paraphrase@0.8,mlm@0.2"


def test_clean_options_strategy_invalid():
    with pytest.raises(ValueError):
        _parse_clean_options({"strategy": "bogus@9"})


def test_clean_options_unknown_option_still_rejected():
    with pytest.raises(ValueError):
        _parse_clean_options({"not_an_option": "x"})


# --- server: default strategy config load -----------------------------------


def test_load_default_strategy_valid(tmp_path):
    p = tmp_path / "clean_strategy.json"
    p.write_text('{"default_strategy": "paraphrase@0.8,mlm@0.2"}')
    assert _load_default_strategy(p) == "paraphrase@0.8,mlm@0.2"


def test_load_default_strategy_missing(tmp_path):
    assert _load_default_strategy(tmp_path / "nope.json") is None


def test_load_default_strategy_bad_json(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{bad json")
    with pytest.raises(SystemExit):
        _load_default_strategy(p)


def test_load_default_strategy_bad_strategy(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"default_strategy": "nope@9"}')
    with pytest.raises(ValueError):
        _load_default_strategy(p)


# --- server: Layer B is a required step for text ---------------------------


def test_clean_text_requires_layer_b(monkeypatch):
    monkeypatch.setattr(server, "_DEFAULT_STRATEGY", None)
    with pytest.raises(ValueError, match="Layer B rewrite is required"):
        server._clean_payload(b"hello world", "a.txt", {})


@pytest.mark.parametrize("name", ["tool.py", "data.json", "table.csv", "notes.rst", "app.po"])
def test_clean_skips_the_default_layer_b_on_non_prose_text(monkeypatch, name):
    # These share the text kind for Layer A, but the default paraphrase plus a
    # masked-LM infill would rewrite identifiers, keys and values.
    monkeypatch.setattr(server, "_DEFAULT_STRATEGY", "paraphrase@0.8,mlm@0.2")

    def fail(*args, **kwargs):
        raise AssertionError("the default Layer B ran on a non-prose file")

    monkeypatch.setattr(server, "_apply_layer_b", fail)
    payload = server._clean_payload("x = 1\u200b\n".encode(), name, {})
    assert base64.b64decode(payload["cleaned"]) == b"x = 1\n"
    assert payload["report"]["layer_b"]["skipped"] is True
    assert "options.strategy" in payload["report"]["layer_b"]["reason"]


def test_clean_skips_layer_b_on_code_even_without_a_default_strategy(monkeypatch):
    # Layer A on code must not depend on a rewrite backend being configured.
    monkeypatch.setattr(server, "_DEFAULT_STRATEGY", None)
    payload = server._clean_payload(b"x = 1\n", "tool.py", {})
    assert payload["ok"] is True


def test_clean_runs_layer_b_on_code_when_the_request_asks(monkeypatch):
    calls = []

    def fake(text, strategy, options):
        calls.append(strategy)
        return text, {"strategy": [strategy]}

    monkeypatch.setattr(server, "_apply_layer_b", fake)
    server._clean_payload(b"x = 1\n", "tool.py", {"strategy": "code@0.3"})
    assert calls == ["code@0.3"]


# --- server: reject when backend/model unavailable --------------------------


def test_apply_layer_b_llm_backend_unconfigured(monkeypatch):
    monkeypatch.delenv("WATERMARKS_REWRITE_BACKEND", raising=False)
    monkeypatch.delenv("WATERMARKS_REWRITE_MODEL", raising=False)
    monkeypatch.delenv("WATERMARKS_REWRITE_BASE_URL", raising=False)
    monkeypatch.delenv("WATERMARKS_REWRITE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        _apply_layer_b("x", "paraphrase@0.8", {})


def test_apply_layer_b_ollama_does_not_require_api_key(monkeypatch):
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "ollama")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "m")
    monkeypatch.setenv("WATERMARKS_REWRITE_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.delenv("WATERMARKS_REWRITE_API_KEY", raising=False)
    monkeypatch.setattr(rewrite_text, "apply_strategy", lambda *a, **k: ("out", {"steps": []}))
    out, _stats = _apply_layer_b("x", "paraphrase@0.8", {})
    assert out == "out"


def test_apply_layer_b_mlm_needs_transformers(monkeypatch):
    import importlib.util

    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "openai-compatible")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "m")
    monkeypatch.setenv("WATERMARKS_REWRITE_BASE_URL", "https://x")
    monkeypatch.setenv("WATERMARKS_REWRITE_API_KEY", "k")
    monkeypatch.setenv("WATERMARKS_REWRITE_ALLOW_REMOTE", "1")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ValueError):
        _apply_layer_b("x", "mlm@0.3", {})


def test_apply_layer_b_remote_denied(monkeypatch):
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "openai-compatible")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "m")
    monkeypatch.setenv("WATERMARKS_REWRITE_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("WATERMARKS_REWRITE_API_KEY", "k")
    monkeypatch.delenv("WATERMARKS_REWRITE_ALLOW_REMOTE", raising=False)
    with pytest.raises(ValueError):
        _apply_layer_b("x", "paraphrase@0.8", {})


# --- server: a broken mlm stack rejects before any LLM step -----------------

_REWRITE_ENV = (
    "WATERMARKS_REWRITE_BACKEND",
    "WATERMARKS_REWRITE_MODEL",
    "WATERMARKS_REWRITE_BASE_URL",
    "WATERMARKS_REWRITE_API_KEY",
    "WATERMARKS_REWRITE_ALLOW_REMOTE",
)


def _transformers_installed(monkeypatch):
    """find_spec sees transformers even where CI has none, as on the reporter's host."""
    import importlib.util

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: object() if name == "transformers" else real_find_spec(name, *a, **k),
    )


def _ollama(monkeypatch):
    for name in _REWRITE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "ollama")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "m")
    monkeypatch.setenv("WATERMARKS_REWRITE_BASE_URL", "http://127.0.0.1:11434")


def test_apply_layer_b_broken_mlm_rejects_before_the_llm_step(monkeypatch):
    """transformers is installed but its pipelines need PIL: the reported 400.

    The default strategy runs paraphrase first, so the mlm failure used to
    surface only after the LLM call had been made. It must now reject first.
    """
    _ollama(monkeypatch)
    _transformers_installed(monkeypatch)

    def broken_mlm():
        raise RuntimeError(
            "mlm tactic unavailable: Failed to import transformers.pipelines: "
            "No module named 'PIL' (install service/scripts/requirements-mlm.txt)"
        )

    monkeypatch.setattr(rewrite_text, "_get_mlm", broken_mlm)
    llm_calls: list[str] = []
    monkeypatch.setattr(
        rewrite_text, "_generate_once", lambda *a, **k: llm_calls.append("x") or "x"
    )
    with pytest.raises(ValueError) as exc:
        _apply_layer_b("The weather was mild.", "paraphrase@0.8,mlm@0.2", {})
    message = str(exc.value)
    assert message.startswith("Layer B rewrite failed: mlm tactic unavailable")
    assert "No module named 'PIL'" in message
    assert "requirements-mlm.txt" in message
    assert llm_calls == []


def test_apply_layer_b_runs_the_default_strategy_once_mlm_loads(monkeypatch):
    _ollama(monkeypatch)
    _transformers_installed(monkeypatch)
    events: list[str] = []
    monkeypatch.setattr(rewrite_text, "_get_mlm", lambda: events.append("load") or (None, "<mask>"))
    monkeypatch.setattr(
        rewrite_text, "_generate_once", lambda *a, **k: events.append("paraphrase") or "Rewritten."
    )
    monkeypatch.setattr(
        rewrite_text, "_mlm_infill", lambda text, level: events.append("mlm") or text + " [mlm]"
    )
    out, stats = _apply_layer_b("Original.", "paraphrase@0.8,mlm@0.2", {})
    assert out == "Rewritten. [mlm]"
    assert stats["strategy"] == ["paraphrase@0.8", "mlm@0.2"]
    assert events == ["load", "paraphrase", "mlm"]


def test_apply_layer_b_mlm_requires_transformers_names_the_requirements(monkeypatch):
    import importlib.util

    _ollama(monkeypatch)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    with pytest.raises(ValueError, match=r"requirements-mlm\.txt"):
        _apply_layer_b("x", "mlm@0.3", {})


# --- server: /capabilities reports what the gate enforces -------------------


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"WATERMARKS_REWRITE_BACKEND": "print-prompt"},
        {"WATERMARKS_REWRITE_BACKEND": "ollama"},
        {
            "WATERMARKS_REWRITE_BACKEND": "ollama",
            "WATERMARKS_REWRITE_MODEL": "m",
            "WATERMARKS_REWRITE_BASE_URL": "http://127.0.0.1:11434",
        },
        {
            "WATERMARKS_REWRITE_BACKEND": "openai-compatible",
            "WATERMARKS_REWRITE_MODEL": "m",
            "WATERMARKS_REWRITE_BASE_URL": "http://localhost:8000",
        },
        {
            "WATERMARKS_REWRITE_BACKEND": "openai-compatible",
            "WATERMARKS_REWRITE_MODEL": "m",
            "WATERMARKS_REWRITE_BASE_URL": "http://localhost:8000",
            "WATERMARKS_REWRITE_API_KEY": "k",
        },
        {
            "WATERMARKS_REWRITE_BACKEND": "openai-compatible",
            "WATERMARKS_REWRITE_MODEL": "m",
            "WATERMARKS_REWRITE_BASE_URL": "https://api.example.test",
            "WATERMARKS_REWRITE_API_KEY": "k",
        },
        {
            "WATERMARKS_REWRITE_BACKEND": "openai-compatible",
            "WATERMARKS_REWRITE_MODEL": "m",
            "WATERMARKS_REWRITE_BASE_URL": "https://api.example.test",
            "WATERMARKS_REWRITE_API_KEY": "k",
            "WATERMARKS_REWRITE_ALLOW_REMOTE": "1",
        },
    ],
)
def test_capability_report_matches_the_layer_b_gate(monkeypatch, env):
    """An LLM tactic is reported usable exactly when /clean would accept it."""
    for name in _REWRITE_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(server, "_mlm_import_error", lambda: None)
    monkeypatch.setattr(rewrite_text, "apply_strategy", lambda *a, **k: ("out", {"steps": []}))
    reported = server._layer_b_status()["tactics"]["paraphrase"]
    try:
        _apply_layer_b("x", "paraphrase@0.8", {})
        accepted = True
    except ValueError:
        accepted = False
    assert reported is accepted


# --- LLM meta-commentary and length drift ------------------------------------

PROBE_INPUT = "The weather was mild and the meeting ended early.\n"


def test_unicode_only_http_clean_does_not_invoke_rewrite(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unicode-only cleaning must not contact a model")

    monkeypatch.setattr(server, "_apply_layer_b", forbidden)
    monkeypatch.setattr(server, "_DEFAULT_STRATEGY", None)
    result = server._clean_payload("Texto\u200b $x$.".encode(), "notes.txt", {"rewrite": False})
    assert base64.b64decode(result["cleaned"]).decode() == "Texto $x$."
    assert result["report"]["layer_b"]["skipped"] is True


PROBE_REWRITE = "The atmosphere was agreeable, and the conference concluded ahead of schedule."
# The rewrite wrapped in a "Here is ...:" preamble and a list of the changes, as
# an Ollama model returned it for PROBE_INPUT in the bug report.
WRAPPED = (ROOT / "tests" / "fixtures" / "rewrite_wrapped_ollama.txt").read_text(encoding="utf-8")
# Commentary inside the rewrite's own paragraph: nothing to strip, but ~5x longer.
CHATTY_INLINE = (
    f"{PROBE_REWRITE} In this version the nouns and verbs were swapped for synonyms, "
    "the clause order was kept, and roughly eighty percent of the tokens differ from "
    "the source sentence, which matches the requested intensity."
)
_LLM = {"backend": "ollama", "model": "m", "base_url": "http://127.0.0.1:11434", "api_key": None}


def _fake_backend(monkeypatch, *replies: str) -> list[str]:
    """Make call_ollama return *replies* in order (the last one repeats)."""
    prompts: list[str] = []

    def fake_ollama(base_url, model, prompt, timeout, temperature, reasoning_effort=None):
        prompts.append(prompt)
        return replies[min(len(prompts), len(replies)) - 1]

    monkeypatch.setattr(rewrite_text, "call_ollama", fake_ollama)
    return prompts


def test_apply_strategy_strips_wrapped_llm_output(monkeypatch):
    _fake_backend(monkeypatch, WRAPPED)
    out, stats = apply_strategy(PROBE_INPUT, [("paraphrase", 0.8)], **_LLM)
    assert out == PROBE_REWRITE
    step = stats["steps"][0]
    assert step["ok"] is True
    assert step["attempts"] == 1
    assert step["out_chars"] == len(PROBE_REWRITE)
    assert step["length_checked"] is True
    assert step["wrappers_stripped"] == ["preamble", "trailer"]
    assert step["rejected"] == []
    assert stats["ok"] is True
    assert stats["errors"] == []
    assert stats["attempt_budget"] == 1
    assert any("meta-commentary" in w for w in stats["warnings"])


def test_apply_strategy_retries_length_drift_within_budget(monkeypatch):
    prompts = _fake_backend(monkeypatch, CHATTY_INLINE, PROBE_REWRITE)
    out, stats = apply_strategy(PROBE_INPUT, [("paraphrase", 0.8)], candidates=2, **_LLM)
    assert out == PROBE_REWRITE
    assert len(prompts) == 2
    step = stats["steps"][0]
    assert step["ok"] is True
    assert step["attempts"] == 2
    assert step["rejected"][0]["length_ratio"] > 2.0
    assert stats["ok"] is True
    assert stats["attempt_budget"] == 2
    assert any("regenerated" in w for w in stats["warnings"])


def test_apply_strategy_reports_drift_when_budget_exhausted(monkeypatch):
    prompts = _fake_backend(monkeypatch, CHATTY_INLINE)
    monkeypatch.setattr(rewrite_text, "_mlm_infill", lambda text, level: text + " [mlm]")
    out, stats = apply_strategy(PROBE_INPUT, [("paraphrase", 0.8), ("mlm", 0.2)], **_LLM)
    # Default budget 1: no retry. The drifted paraphrase never reaches the result;
    # its input goes on to the next step instead.
    assert len(prompts) == 1
    assert out == PROBE_INPUT + " [mlm]"
    step = stats["steps"][0]
    assert step["ok"] is False
    assert step["attempts"] == 1
    assert step["out_chars"] == step["in_chars"]
    assert "length drifted" in step["error"]
    assert stats["ok"] is False
    assert stats["errors"] == [step["error"]]
    assert stats["steps"][1]["ok"] is True


def test_apply_strategy_structural_step_not_length_checked(monkeypatch):
    _fake_backend(monkeypatch, CHATTY_INLINE)
    out, stats = apply_strategy(PROBE_INPUT, [("structural", 0.5)], **_LLM)
    assert out == CHATTY_INLINE
    assert stats["steps"][0]["length_checked"] is False
    assert stats["ok"] is True


def test_apply_layer_b_uses_env_attempt_budget(monkeypatch):
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "ollama")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "m")
    monkeypatch.setenv("WATERMARKS_REWRITE_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("WATERMARKS_REWRITE_CANDIDATES", "2")
    monkeypatch.setenv("WATERMARKS_REWRITE_LOOPS", "3")
    captured: dict = {}

    def fake_apply(*args, **kwargs):
        captured.update(kwargs)
        return "out", {"steps": []}

    monkeypatch.setattr(rewrite_text, "apply_strategy", fake_apply)
    _apply_layer_b("x", "paraphrase@0.8", {})
    assert (captured["candidates"], captured["max_loops"]) == (2, 3)


# --- /clean end to end against a fake Ollama server ---------------------------


class _FakeOllama(http.server.BaseHTTPRequestHandler):
    """Answer /api/chat with the server's queued replies (the last one repeats)."""

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        srv = self.server
        reply = srv.replies[min(srv.calls, len(srv.replies) - 1)]
        srv.calls += 1
        body = json.dumps({"message": {"role": "assistant", "content": reply}}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def fake_ollama(monkeypatch):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllama)
    srv.replies = [WRAPPED]
    srv.calls = 0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "ollama")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "m")
    monkeypatch.setenv("WATERMARKS_REWRITE_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    for var in (
        "WATERMARKS_REWRITE_API_KEY",
        "WATERMARKS_REWRITE_ALLOW_REMOTE",
        "WATERMARKS_REWRITE_CANDIDATES",
        "WATERMARKS_REWRITE_LOOPS",
        "WATERMARKS_REWRITE_REASONING_EFFORT",
        "WATERMARKS_REWRITE_TEMPERATURE",
    ):
        monkeypatch.delenv(var, raising=False)
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def service():
    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=30)
    yield conn
    conn.close()
    srv.shutdown()
    srv.server_close()


def _clean(conn: http.client.HTTPConnection, text: str, strategy: str) -> tuple[int, dict]:
    payload = {
        "file": base64.b64encode(text.encode("utf-8")).decode("ascii"),
        "name": "probe.txt",
        "options": {"strategy": strategy},
    }
    conn.request(
        "POST", "/clean", body=json.dumps(payload), headers={"Content-Type": "application/json"}
    )
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read())


def test_clean_strips_llm_meta_commentary(fake_ollama, service):
    status, body = _clean(service, PROBE_INPUT, "paraphrase@0.8")
    assert status == 200
    assert body["ok"] is True
    assert base64.b64decode(body["cleaned"]).decode("utf-8") == PROBE_REWRITE
    layer_b = body["report"]["layer_b"]
    # The pre-existing report.layer_b fields keep their meaning.
    assert layer_b["strategy"] == ["paraphrase@0.8"]
    assert layer_b["output_chars"] == len(PROBE_REWRITE)
    step = layer_b["steps"][0]
    assert (step["tactic"], step["in_chars"], step["out_chars"]) == (
        "paraphrase",
        len(PROBE_INPUT),
        len(PROBE_REWRITE),
    )
    assert step["wrappers_stripped"] == ["preamble", "trailer"]
    assert layer_b["ok"] is True
    assert layer_b["errors"] == []


def test_clean_reports_unrecoverable_length_drift(fake_ollama, service):
    fake_ollama.replies = [CHATTY_INLINE]
    status, body = _clean(service, PROBE_INPUT, "paraphrase@0.8")
    assert status == 200
    # The commentary never becomes the cleaned text: the step is skipped...
    assert base64.b64decode(body["cleaned"]).decode("utf-8") == PROBE_INPUT
    # ...and the report says so instead of a silent ok.
    layer_b = body["report"]["layer_b"]
    assert layer_b["ok"] is False
    assert len(layer_b["errors"]) == 1
    assert "length drifted" in layer_b["errors"][0]
    assert layer_b["steps"][0]["ok"] is False
    assert fake_ollama.calls == 1


def test_clean_retries_length_drift_within_env_budget(fake_ollama, service, monkeypatch):
    monkeypatch.setenv("WATERMARKS_REWRITE_LOOPS", "2")
    fake_ollama.replies = [CHATTY_INLINE, WRAPPED]
    status, body = _clean(service, PROBE_INPUT, "paraphrase@0.8")
    assert status == 200
    assert base64.b64decode(body["cleaned"]).decode("utf-8") == PROBE_REWRITE
    layer_b = body["report"]["layer_b"]
    assert layer_b["ok"] is True
    assert layer_b["attempt_budget"] == 2
    assert layer_b["steps"][0]["attempts"] == 2
    assert fake_ollama.calls == 2


# --- server: backend timeout and reasoning control ---------------------------


def _ollama_env(monkeypatch):
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "ollama")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "m")
    monkeypatch.setenv("WATERMARKS_REWRITE_BASE_URL", "http://127.0.0.1:11434")


def _capture_apply_strategy(monkeypatch) -> dict:
    seen: dict = {}

    def fake(text, steps, **kwargs):
        seen.update(kwargs)
        return text, {"steps": []}

    monkeypatch.setattr(rewrite_text, "apply_strategy", fake)
    return seen


def test_apply_layer_b_defaults_timeout_and_reasoning_effort(monkeypatch):
    _ollama_env(monkeypatch)
    monkeypatch.delenv("WATERMARKS_REWRITE_TIMEOUT", raising=False)
    monkeypatch.delenv("WATERMARKS_REWRITE_REASONING_EFFORT", raising=False)
    seen = _capture_apply_strategy(monkeypatch)
    _apply_layer_b("x", "paraphrase@0.8", {})
    # "none" is the rewrite_text.py CLI default too. The service used to send
    # nothing, and a thinking model then reasoned for minutes on every /clean.
    assert seen["timeout"] == 120.0
    assert seen["reasoning_effort"] == "none"


def test_apply_layer_b_reads_timeout_and_effort_from_env(monkeypatch):
    _ollama_env(monkeypatch)
    monkeypatch.setenv("WATERMARKS_REWRITE_TIMEOUT", "900")
    monkeypatch.setenv("WATERMARKS_REWRITE_REASONING_EFFORT", "off")
    seen = _capture_apply_strategy(monkeypatch)
    _apply_layer_b("x", "paraphrase@0.8", {})
    assert seen["timeout"] == 900.0
    assert seen["reasoning_effort"] is None


def test_apply_layer_b_caps_the_timeout(monkeypatch):
    _ollama_env(monkeypatch)
    monkeypatch.setenv("WATERMARKS_REWRITE_TIMEOUT", "99999")
    seen = _capture_apply_strategy(monkeypatch)
    _apply_layer_b("x", "paraphrase@0.8", {})
    assert seen["timeout"] == server.REWRITE_TIMEOUT_MAX


def test_apply_layer_b_rejects_a_non_numeric_timeout(monkeypatch):
    _ollama_env(monkeypatch)
    monkeypatch.setenv("WATERMARKS_REWRITE_TIMEOUT", "soon")
    with pytest.raises(ValueError, match="WATERMARKS_REWRITE_TIMEOUT"):
        _apply_layer_b("x", "paraphrase@0.8", {})


@pytest.mark.parametrize(
    "error",
    [TimeoutError("timed out"), urllib.error.URLError(TimeoutError("timed out"))],
    ids=["read-timeout", "connect-timeout"],
)
def test_apply_layer_b_timeout_names_the_variable(monkeypatch, error):
    _ollama_env(monkeypatch)
    monkeypatch.setenv("WATERMARKS_REWRITE_TIMEOUT", "30")

    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(rewrite_text, "apply_strategy", boom)
    with pytest.raises(ValueError, match=r"within 30 s .*WATERMARKS_REWRITE_TIMEOUT"):
        _apply_layer_b("x", "paraphrase@0.8", {})


def test_apply_layer_b_passes_other_backend_errors_through(monkeypatch):
    _ollama_env(monkeypatch)

    def boom(*args, **kwargs):
        raise urllib.error.URLError(ConnectionRefusedError(10061, "refused"))

    monkeypatch.setattr(rewrite_text, "apply_strategy", boom)
    with pytest.raises(ValueError, match="Layer B rewrite failed: <urlopen error") as info:
        _apply_layer_b("x", "paraphrase@0.8", {})
    assert "WATERMARKS_REWRITE_TIMEOUT" not in str(info.value)


# --- server: /clean runs two-step tactics and reports no-ops ----------------

_CLEAN_SOURCE = "The gluon propagator is suppressed in the infrared below the Gribov horizon."


def _ollama_env_isolated(monkeypatch):
    """Point Layer B at a loopback ollama backend, isolated from the user's env."""
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "ollama")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "m")
    monkeypatch.setenv("WATERMARKS_REWRITE_BASE_URL", "http://127.0.0.1:11434")
    for name in (
        "WATERMARKS_REWRITE_API_KEY",
        "WATERMARKS_REWRITE_ALLOW_REMOTE",
        "WATERMARKS_REWRITE_TEMPERATURE",
        "WATERMARKS_REWRITE_REASONING_EFFORT",
    ):
        monkeypatch.delenv(name, raising=False)


def _clean_txt(options):
    """Clean _CLEAN_SOURCE as a .txt upload the way POST /clean does."""
    resp = server._clean_payload(
        _CLEAN_SOURCE.encode("utf-8"), "notes.txt", _parse_clean_options(options)
    )
    return base64.b64decode(resp["cleaned"]).decode("utf-8"), resp["report"]["layer_b"]


@pytest.mark.parametrize(
    ("strategy", "keys", "intermediate", "final"),
    [
        pytest.param(
            "backtranslate@0.8",
            ("backtranslate_out", "backtranslate_back"),
            "Le propagateur du gluon est supprimé dans l'infrarouge sous l'horizon de Gribov.",
            "Below the Gribov horizon, the gluon propagator is suppressed in the infrared.",
            id="backtranslate",
        ),
        pytest.param(
            "structural@0.8",
            ("structural_outline", "structural_write"),
            "- gluon propagator: suppressed in the infrared\n- regime: below the Gribov horizon",
            "In the infrared, below the Gribov horizon, the gluon propagator is suppressed.",
            id="structural",
        ),
    ],
)
def test_clean_two_step_strategy_runs_two_generations_in_order(
    monkeypatch, strategy, keys, intermediate, final
):
    _ollama_env_isolated(monkeypatch)
    langs = {"LANG": "French", "ORIGINAL_LANG": "English"}
    first_prompt = rewrite_text.PROMPTS[keys[0]].format(TEXT=_CLEAN_SOURCE, **langs)
    final_prompt = rewrite_text.PROMPTS[keys[1]].format(TEXT=intermediate, **langs)
    script = {first_prompt: intermediate, final_prompt: final}
    prompts: list[str] = []

    def backend(base_url, model, prompt, timeout, temperature, reasoning_effort=None):
        # Any unscripted prompt (the combined one-shot one included) is
        # shortcut by echoing the input, as the real model did.
        prompts.append(prompt)
        return script.get(prompt, _CLEAN_SOURCE)

    monkeypatch.setattr(rewrite_text, "call_ollama", backend)

    cleaned, layer_b = _clean_txt({"strategy": strategy})

    assert prompts == [first_prompt, final_prompt]
    assert cleaned == final  # the final document only, never the outline too
    assert layer_b["steps"][0]["generations"] == 2
    assert layer_b["noop"] is False
    assert layer_b["warnings"] == []


def test_clean_strategy_noop_reported_in_layer_b(monkeypatch):
    # The model hands every input back unchanged: the clean still succeeds, but
    # report.layer_b must flag the rewrite as a no-op with a warning.
    _ollama_env_isolated(monkeypatch)
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *_a, **_k: _CLEAN_SOURCE)

    cleaned, layer_b = _clean_txt({"strategy": "backtranslate@0.8"})

    assert cleaned == _CLEAN_SOURCE
    assert layer_b["noop"] is True
    assert layer_b["lexical_divergence"] == 0.0
    assert len(layer_b["warnings"]) == 1
    assert "no-op" in layer_b["warnings"][0]
