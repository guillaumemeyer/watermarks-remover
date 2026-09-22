"""Layer B on long and academic inputs: chunking, abbreviations, Ollama budget,
and the optional rewrite of LaTeX / Markdown sources through /clean."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service" / "scripts"))

import rewrite_text
import server
from latex_mask import mask_latex, restore_latex
from rewrite_text import _prose_chunks, _split_units, apply_strategy

_LLM = {"backend": "ollama", "model": "m", "base_url": "http://127.0.0.1:11434", "api_key": None}


# --- sentence splitting knows abbreviations ------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "See Sect. 2 for the proof. It is short.",
        "This holds, e.g. for SU(2). The rest follows.",
        "As shown by Gribov et al. in 1978. The horizon appears.",
        "Compare Eq. (3) with Fig. 2. Both agree.",
        "Written by J. Doe and A. Smith. It was reviewed.",
    ],
)
def test_split_units_does_not_break_after_abbreviations(text):
    units = [u for u, _ in _split_units(text)]
    assert len(units) == 2, units
    assert units[0].endswith(".") and units[1].endswith(".")
    assert "".join(u + " " for u in units).strip() == text


def test_split_units_still_splits_real_sentences_and_paragraphs():
    text = "First one. Second one!\n\nThird one?"
    assert [u for u, _ in _split_units(text)] == ["First one.", "Second one!", "Third one?"]


# --- paragraph chunking -----------------------------------------------------------


def test_prose_chunks_join_back_exactly_and_prefer_blank_lines():
    paragraphs = [f"Paragraph {i} " + "word " * 40 + "end." for i in range(6)]
    text = "\n\n".join(paragraphs)
    pieces = _prose_chunks(text, 500)
    assert "".join(pieces) == text
    assert all(len(p) <= 500 for p in pieces)
    assert len(pieces) > 1
    # Every cut but the last falls right after a blank line.
    assert all(p.endswith("\n\n") for p in pieces[:-1])


def test_prose_chunks_fall_back_to_sentence_ends_inside_a_long_paragraph():
    text = " ".join(f"Sentence number {i} is here, cf. the rest." for i in range(60))
    pieces = _prose_chunks(text, 400)
    assert "".join(pieces) == text
    assert all(len(p) <= 400 for p in pieces)
    # "cf." never ends a piece; real sentence ends do.
    assert all(p.rstrip().endswith("rest.") for p in pieces[:-1])


def test_prose_chunks_never_split_a_placeholder():
    masked, mask = mask_latex("Text $a+b$ more " * 200, mode="on")
    pieces = _prose_chunks(masked, 300)
    assert "".join(pieces) == masked
    assert sum(len(mask.token_re.findall(p)) for p in pieces) == mask.count


def test_prose_chunks_disabled_with_zero_limit():
    text = "a " * 5000
    assert _prose_chunks(text, 0) == [text]


# --- Ollama sampling budget grows with the prompt --------------------------------


def test_ollama_budget_floors_and_growth(monkeypatch):
    for key in ("CONTEXT", "MAX_TOKENS", "THREADS"):
        monkeypatch.delenv("WATERMARKS_OLLAMA_" + key, raising=False)
    small = rewrite_text._ollama_budget("short prompt")
    assert small["num_ctx"] >= 8192 and small["num_predict"] >= 2048
    assert "num_thread" not in small
    big = rewrite_text._ollama_budget("x" * 60000)
    assert big["num_predict"] > small["num_predict"]
    assert big["num_ctx"] > big["num_predict"] + 60000 // 3
    monkeypatch.setenv("WATERMARKS_OLLAMA_THREADS", "4")
    assert rewrite_text._ollama_budget("p")["num_thread"] == 4


# --- apply_strategy rewrites piece by piece --------------------------------------


def _paragraph_text(n: int) -> str:
    return "\n\n".join(
        f"Paragraph {i} discusses the gap and the horizon in some detail here." for i in range(n)
    )


def test_apply_strategy_chunks_long_input_and_keeps_layout(monkeypatch):
    prompts: list[str] = []

    def fake_ollama(base_url, model, prompt, timeout, temperature, reasoning_effort=None):
        prompts.append(prompt)
        body = prompt.split("---\n", 1)[1].split("\n\nModulate this rewrite")[0]
        return body.replace("discusses", "treats")

    monkeypatch.setattr(rewrite_text, "call_ollama", fake_ollama)
    text = _paragraph_text(8)
    out, stats = apply_strategy(text, [("paraphrase", 0.5)], chunk_chars=200, **_LLM)
    step = stats["steps"][0]
    assert step["chunks"] > 1
    assert step["generations"] == len(prompts) == step["chunks"]
    assert step["ok"] is True
    assert stats["chunk_chars"] == 200
    # Layout survives: same paragraph count, every paragraph rewritten.
    assert out.count("\n\n") == text.count("\n\n")
    assert "discusses" not in out and out.count("treats") == 8


def test_apply_strategy_keeps_a_damaged_chunk_and_reports_it(monkeypatch):
    calls = {"n": 0}

    def fake_ollama(base_url, model, prompt, timeout, temperature, reasoning_effort=None):
        calls["n"] += 1
        body = prompt.split("---\n", 1)[1].split("\n\nModulate this rewrite")[0]
        if "Paragraph 2 " in body:
            return "Here is a much longer commentary. " * 30  # length drift
        return body.replace("discusses", "treats")

    monkeypatch.setattr(rewrite_text, "call_ollama", fake_ollama)
    text = _paragraph_text(4)
    out, stats = apply_strategy(text, [("paraphrase", 0.5)], chunk_chars=120, **_LLM)
    step = stats["steps"][0]
    assert step["ok"] is False and stats["ok"] is False
    assert len(stats["errors"]) == 1 and "chunk" in stats["errors"][0]
    assert step["error"] == stats["errors"][0]
    assert "Paragraph 2 discusses" in out  # the failed piece passed through unchanged
    assert out.count("treats") == 3
    assert step["rejected"][0]["chunk"] >= 1


def test_apply_strategy_academic_rejects_a_chunk_that_changes_a_number(monkeypatch):
    def fake_ollama(base_url, model, prompt, timeout, temperature, reasoning_effort=None):
        body = prompt.split("---\n", 1)[1].split("\n\nModulate this rewrite")[0]
        return body.replace("three", "3").replace(" 2 ", " 5 ")

    monkeypatch.setattr(rewrite_text, "call_ollama", fake_ollama)
    text = "We take 2 copies. There are three of them."
    out, stats = apply_strategy(text, [("academic", 0.4)], **_LLM)
    assert out == text
    assert stats["steps"][0]["ok"] is False
    assert "numbers changed" in stats["errors"][0]


def test_apply_strategy_protects_latex_across_chunks(monkeypatch):
    def fake_ollama(base_url, model, prompt, timeout, temperature, reasoning_effort=None):
        body = prompt.split("---\n", 1)[1].split("\n\nModulate this rewrite")[0]
        return body.replace("operator", "map")

    monkeypatch.setattr(rewrite_text, "call_ollama", fake_ollama)
    paragraphs = [
        f"The operator $M_{i}$ acts on $\\Omega^\\circ$ as in \\cite{{Gribov{i}}}."
        for i in range(6)
    ]
    text = "\n\n".join(paragraphs)
    out, stats = apply_strategy(text, [("academic", 0.4)], chunk_chars=150, **_LLM)
    assert stats["latex"] == {
        "protected": stats["latex_protected"],
        "restored": stats["latex_protected"],
        "missing": 0,
        "duplicated": 0,
    }
    for i in range(6):
        assert f"$M_{i}$" in out and f"\\cite{{Gribov{i}}}" in out
    assert "operator" not in out and out.count("map") == 6


def test_apply_strategy_two_step_carries_placeholders(monkeypatch):
    seen: list[str] = []

    def fake_ollama(base_url, model, prompt, timeout, temperature, reasoning_effort=None):
        seen.append(prompt)
        return prompt.split("---\n", 1)[1].split("\n\nThe text contains")[0]

    monkeypatch.setattr(rewrite_text, "call_ollama", fake_ollama)
    text = "The bound $x<1$ holds."
    out, stats = apply_strategy(text, [("backtranslate", 0.8)], **_LLM)
    assert len(seen) == 2 and all("protected token" in p for p in seen)
    assert "$x<1$" in out
    assert stats["steps"][0]["generations"] == 2


# --- /clean on LaTeX and Markdown sources ------------------------------------------


def _clean(name: str, body: str, options: dict) -> dict:
    return server._clean_payload(body.encode("utf-8"), name, server._parse_clean_options(options))


def _model_env(monkeypatch):
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "ollama")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "m")
    monkeypatch.setenv("WATERMARKS_REWRITE_BASE_URL", "http://127.0.0.1:11434")
    for name in (
        "WATERMARKS_REWRITE_API_KEY",
        "WATERMARKS_REWRITE_ALLOW_REMOTE",
        "WATERMARKS_REWRITE_TIMEOUT",
        "WATERMARKS_REWRITE_REASONING_EFFORT",
        "WATERMARKS_REWRITE_TEMPERATURE",
        "WATERMARKS_REWRITE_CHUNK_CHARS",
        "WATERMARKS_PROTECT_LATEX",
    ):
        monkeypatch.delenv(name, raising=False)


TEX = (
    "\\documentclass{revtex4-2}\n\\begin{document}\n\\section{Introduction}\n"
    "The Faddeev-Popov operator is discussed in \\cite{Gribov1978}; see $\\Omega^\\circ$.\n"
    "\\begin{equation}\nM = -\\partial D\n\\end{equation}\n\\end{document}\n"
)


def test_clean_tex_skips_layer_b_by_default_and_says_how_to_enable_it(monkeypatch):
    monkeypatch.setattr(server, "_apply_layer_b", lambda *a, **k: pytest.fail("rewrote"))
    resp = _clean("paper.tex", TEX, {})
    assert resp["kind"] == "container"
    layer_b = resp["report"]["layer_b"]
    assert layer_b["skipped"] is True and "options.rewrite" in layer_b["reason"]
    assert base64.b64decode(resp["cleaned"]).decode("utf-8") == TEX


def test_clean_tex_with_rewrite_true_rewrites_prose_only(monkeypatch):
    _model_env(monkeypatch)
    monkeypatch.setattr(server, "_DEFAULT_STRATEGY", "academic@0.4")

    def fake_ollama(base_url, model, prompt, timeout, temperature, reasoning_effort=None):
        body = prompt.split("---\n", 1)[1].split("\n\nModulate this rewrite")[0]
        return body.replace("is discussed", "is treated")

    monkeypatch.setattr(rewrite_text, "call_ollama", fake_ollama)
    resp = _clean("paper.tex", TEX, {"rewrite": True})
    out = base64.b64decode(resp["cleaned"]).decode("utf-8")
    layer_b = resp["report"]["layer_b"]
    assert layer_b["ok"] is True and layer_b["protect_latex"] == "on"
    assert layer_b["latex"]["missing"] == 0 and layer_b["latex_protected"] >= 4
    assert "is treated" in out and "is discussed" not in out
    for span in (
        "\\documentclass{revtex4-2}",
        "\\cite{Gribov1978}",
        "$\\Omega^\\circ$",
        "\\begin{equation}\nM = -\\partial D\n\\end{equation}",
        "\\section{Introduction}",
    ):
        assert span in out


def test_clean_tex_rewrite_false_keeps_layer_a_only(monkeypatch):
    monkeypatch.setattr(server, "_apply_layer_b", lambda *a, **k: pytest.fail("rewrote"))
    resp = _clean("paper.tex", TEX.replace("operator", "oper\u200bator"), {"rewrite": False})
    assert base64.b64decode(resp["cleaned"]).decode("utf-8") == TEX
    assert resp["report"]["layer_b"]["skipped"] is True


def test_clean_markdown_masks_front_matter_and_link_targets(monkeypatch):
    _model_env(monkeypatch)
    prompts: list[str] = []

    def fake_ollama(base_url, model, prompt, timeout, temperature, reasoning_effort=None):
        prompts.append(prompt)
        body = prompt.split("---\n", 1)[1].split("\n\nModulate this rewrite")[0]
        return body.replace("explains", "describes")

    monkeypatch.setattr(rewrite_text, "call_ollama", fake_ollama)
    md = "---\ntitle: Notes\nauthor: me\n---\n\nThis note explains the [gap](https://x.y/z?q=1).\n"
    resp = _clean("notes.md", md, {"strategy": "academic@0.4"})
    out = base64.b64decode(resp["cleaned"]).decode("utf-8")
    assert out.startswith("---\ntitle: Notes\nauthor: me\n---\n")
    assert "(https://x.y/z?q=1)" in out and "describes" in out
    assert all("https://x.y" not in p and "title: Notes" not in p for p in prompts)


def test_mask_latex_front_matter_and_links_round_trip():
    md = "---\nk: v\n---\nSee [a](http://h/p(1)) and `code`.\n"
    masked, mask = mask_latex(md, mode="on")
    assert "http://h" not in masked and "k: v" not in masked
    assert restore_latex(masked, mask)[0] == md
