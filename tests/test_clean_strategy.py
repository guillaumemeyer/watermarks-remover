"""Tests for Layer B strategy parsing/application and /clean wiring."""

from __future__ import annotations

import sys
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
