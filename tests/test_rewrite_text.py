"""Tests for Layer B rewrite_text hook (offline / print-prompt + client hardening)."""

from __future__ import annotations

import http.server
import json
import sys
import threading
import time
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rewrite_text
from rewrite_text import (
    _candidate_pass,
    _check_remote,
    _flag_env,
    _lexical_divergence,
    _select_candidate,
    _tokens,
    build_prompt,
    rewrite,
    strip_model_wrappers,
)


def _rewrite_kwargs(**overrides):
    kwargs = dict(
        backend="print-prompt",
        model=None,
        base_url=None,
        api_key=None,
        tactic="paraphrase",
        lang="French",
        original_lang="English",
        timeout=5.0,
        layer_a_after=True,
        temperature=0.9,
        candidates=1,
    )
    kwargs.update(overrides)
    return kwargs


def test_build_prompt_paraphrase_is_word_choice_plus_syntax():
    p = build_prompt("paraphrase", "Hello world facts 42.", lang="French", original_lang="English")
    assert "Hello world facts 42." in p
    assert "clause order" in p
    assert "function words" in p


def test_build_prompt_humanize_and_code_contain_text():
    for tactic, keyword in (("humanize", "human wrote it"), ("code", "comments")):
        p = build_prompt(tactic, "ABC 123", lang="French", original_lang="English")
        assert "ABC 123" in p
        assert keyword in p


def test_build_prompt_humanize_lists_humanizer_rules():
    p = build_prompt("humanize", "ABC 123", lang="French", original_lang="English")
    assert "human wrote it" in p
    for rule in ("active voice", "utilize", "em dashes", "rule-of-three", "in order to"):
        assert rule in p


def test_build_prompt_unknown_tactic_raises():
    with pytest.raises(ValueError):
        build_prompt("nope", "ABC", lang="French", original_lang="English")


def test_build_prompt_level_modulates_tactic():
    p = build_prompt(
        "paraphrase", "Hello 42.", lang="French", original_lang="English", rewrite_level=0.3
    )
    assert "Hello 42." in p
    assert "0.30" in p
    # tactic + level keeps the tactic-specific instruction AND the clause
    assert "clause order" in p


def test_build_prompt_level_alone_keeps_generic_prompt():
    # A bare intensity (no tactic) still yields the generic level-only prompt.
    p = build_prompt(None, "Hello 42.", lang="French", original_lang="English", rewrite_level=0.3)
    assert "0.30" in p
    assert "clause order" not in p


def test_build_prompt_style_appended_to_humanize():
    p = build_prompt(
        "humanize", "ABC 123", lang="French", original_lang="English", style="write like hemingway"
    )
    assert "ABC 123" in p
    assert "human wrote it" in p
    assert "write like hemingway" in p


def test_build_prompt_style_combines_with_level():
    p = build_prompt(
        "humanize",
        "Hello 42.",
        lang="French",
        original_lang="English",
        rewrite_level=0.3,
        style="terse and plain",
    )
    assert "0.30" in p
    assert "terse and plain" in p
    assert "human wrote it" in p


def test_build_prompt_no_style_when_unset():
    p = build_prompt("humanize", "ABC 123", lang="French", original_lang="English")
    assert "Apply this writing style" not in p


def test_rewrite_level_modulates_tactic():
    out, info = rewrite(
        "Sample prose about water marks 42.",
        **_rewrite_kwargs(tactic="paraphrase", rewrite_level=0.4),
    )
    assert info["mode"] == "print-prompt"
    assert info["tactic"] == "paraphrase"
    assert info["rewrite_level"] == 0.4
    assert info["noop"] is False  # print-prompt echoes the prompt; long input
    assert "0.40" in out
    assert "clause order" in out  # paraphrase instruction retained


def test_rewrite_style_recorded_and_echoed():
    out, info = rewrite(
        "Sample prose about water marks 42.",
        **_rewrite_kwargs(tactic="humanize", style="terse and plain"),
    )
    assert info["mode"] == "print-prompt"
    assert info["tactic"] == "humanize"
    assert info["style"] == "terse and plain"
    assert "terse and plain" in out  # print-prompt echoes the styled prompt
    assert "human wrote it" in out  # humanize instruction retained


def test_rewrite_style_default_is_none():
    _out, info = rewrite("Sample prose about water marks 42.", **_rewrite_kwargs())
    assert info["style"] is None
    assert "Apply this writing style" not in _out


def test_rewrite_level_out_of_range_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["rewrite_text.py", "--rewrite-level", "0", "x.txt"])
    assert rewrite_text.main() == 2


def test_print_prompt_backend():
    out, info = rewrite("Sample prose about water marks.", **_rewrite_kwargs())
    assert info["mode"] == "print-prompt"
    assert "Sample prose" in out
    assert info["backend"] == "print-prompt"
    assert info["temperature"] == 0.9


def test_print_prompt_ignores_candidates():
    out, info = rewrite("Sample prose about water marks.", **_rewrite_kwargs(candidates=2))
    assert info["mode"] == "print-prompt"
    assert isinstance(out, str)
    assert "Sample prose" in out


def test_structural_and_backtranslate_prompts():
    for tactic in ("structural", "backtranslate"):
        p = build_prompt(tactic, "ABC 123", lang="German", original_lang="English")
        assert "ABC 123" in p


def test_humanize_tactic_applies_deterministic_pass(monkeypatch):
    """The humanize candidate is cleaned before evaluation: dashes and filler go."""
    monkeypatch.setattr(
        rewrite_text,
        "call_ollama",
        lambda *a, **k: "In order to see the result\u2014the answer\u2014utilize the tool",
    )
    out, info = rewrite(
        "the cat sat on the mat",
        **_rewrite_candidates_kwargs(tactic="humanize", candidates=1),
    )
    assert out == "To see the result, the answer, use the tool"
    assert info["tactic"] == "humanize"
    assert info["evaluator"] == "lexical-divergence"


def test_non_humanize_tactic_skips_deterministic_pass(monkeypatch):
    """Only the humanize tactic runs the humanizer pass."""
    monkeypatch.setattr(
        rewrite_text,
        "call_ollama",
        lambda *a, **k: "alpha\u2014beta in order to gamma",
    )
    out, _info = rewrite(
        "the cat sat on the mat",
        **_rewrite_candidates_kwargs(tactic="paraphrase", candidates=1),
    )
    assert out == "alpha\u2014beta in order to gamma"


def test_lexical_divergence_identical_is_zero():
    assert _lexical_divergence("the cat sat", "the cat sat") == 0.0


def test_lexical_divergence_fully_different_higher_than_similar():
    similar = _lexical_divergence("the cat sat on the mat", "the dog sat on the mat")
    different = _lexical_divergence("the cat sat on the mat", "alpha beta gamma delta")
    assert different > similar


def test_lexical_divergence_empty_inputs():
    assert _lexical_divergence("", "") == 0.0
    assert _lexical_divergence("", "text") == 1.0
    assert _lexical_divergence("text", "") == 1.0


def test_lexical_divergence_unicode_diacritics_not_shattered():
    """Non-ASCII diacritics (e.g. Polish, French) must not shatter into single-char fragments."""
    polish = "Właściwość języka polskiego"
    tokens = _tokens(polish)
    assert tokens == ["właściwość", "języka", "polskiego"]
    assert _lexical_divergence(polish, polish) == 0.0
    # Modifying one word should yield a clean bigram divergence rather than fragmentation noise
    modified = "Struktura języka polskiego"
    div = _lexical_divergence(polish, modified)
    assert 0.0 < div < 1.0
    assert div == pytest.approx(2 / 3, rel=1e-3)


def test_select_candidate_prefers_more_divergent():
    original = "the cat sat on the mat"
    best, scores = _select_candidate(
        original,
        ["the cat sat on the mat", "the dog sat on the mat", "alpha beta gamma delta"],
    )
    assert best == "alpha beta gamma delta"
    assert len(scores) == 3


# ---------------------------------------------------------------------------
# Iterative rewrite loop with detection-guided evaluation
# ---------------------------------------------------------------------------


class _FakeMarkLLM:
    """Stand-in for text_detectors.MarkLLMTextDetector (no subprocess).

    Verdict by exact text match: "the cat sat on the mat" is watermarked,
    everything else is not.
    """

    name = "markllm"

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def available(self) -> bool:
        return True

    def detect(self, text: str) -> dict:
        return {
            "detector": "markllm",
            "scheme": "kgw",
            "vendor": "open-llm",
            "available": True,
            "is_watermarked": text == "the cat sat on the mat",
            "score": 3.0 if text == "the cat sat on the mat" else 0.5,
            "threshold": 3.0,
        }


def _rewrite_candidates_kwargs(**overrides):
    kwargs = dict(
        backend="ollama",
        model="m",
        base_url="http://127.0.0.1:11434",
        api_key=None,
        tactic="paraphrase",
        lang="French",
        original_lang="English",
        timeout=10,
        layer_a_after=False,
        temperature=0.9,
        candidates=2,
    )
    kwargs.update(overrides)
    return kwargs


def _two_candidates(monkeypatch):
    """call_ollama yields an identical then a fully divergent candidate."""
    texts = iter(["the cat sat on the mat", "alpha beta gamma delta"])
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: next(texts))


def test_default_candidates_and_max_loops_are_one(monkeypatch):
    monkeypatch.delenv("WATERMARKS_REWRITE_CANDIDATES", raising=False)
    monkeypatch.delenv("WATERMARKS_REWRITE_LOOPS", raising=False)
    assert rewrite_text.DEFAULT_CANDIDATES == 1
    assert rewrite_text.DEFAULT_MAX_LOOPS == 1
    args = rewrite_text.build_parser().parse_args(["x.txt"])
    assert args.candidates == 1
    assert args.max_loops == 1
    monkeypatch.setenv("WATERMARKS_REWRITE_CANDIDATES", "5")
    monkeypatch.setenv("WATERMARKS_REWRITE_LOOPS", "7")
    args = rewrite_text.build_parser().parse_args(["x.txt"])
    assert args.candidates == 5
    assert args.max_loops == 7


def test_evaluator_is_lexical_without_markllm_scheme(monkeypatch):
    monkeypatch.setattr(
        rewrite_text,
        "MarkLLMTextDetector",
        lambda *a, **k: pytest.fail("markllm must not be built without --markllm-scheme"),
    )
    _two_candidates(monkeypatch)
    out, info = rewrite("the cat sat on the mat", **_rewrite_candidates_kwargs())
    # no detector configured: lexical divergence runs all attempts, no verdict
    assert info["evaluator"] == "lexical-divergence"
    assert info["passed"] is None
    assert info["attempts_made"] == 2
    assert out == "alpha beta gamma delta"
    assert all(
        c["evaluation"]["evaluator"] == "lexical-divergence" for c in info["candidate_scores"]
    )
    assert "markllm" not in info


def test_duplicate_candidates_select_first(monkeypatch):
    texts = iter(["alpha beta gamma delta", "alpha beta gamma delta"])
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: next(texts))
    out, info = rewrite("the cat sat on the mat", **_rewrite_candidates_kwargs())
    assert out == "alpha beta gamma delta"
    assert [c["selected"] for c in info["candidate_scores"]] == [True, False]


def test_markllm_evaluator_loop_stops_on_pass(monkeypatch):
    monkeypatch.setattr(rewrite_text, "MarkLLMTextDetector", _FakeMarkLLM)
    _two_candidates(monkeypatch)
    out, info = rewrite(
        "the cat sat on the mat",
        **_rewrite_candidates_kwargs(markllm_scheme="kgw", markllm_dir="/x"),
    )
    assert info["evaluator"] == "markllm"
    assert out == "alpha beta gamma delta"
    assert info["attempts_made"] == 2
    assert info["passed"] is True
    cs = info["candidate_scores"]
    assert [c["passed"] for c in cs] == [False, True]
    assert [c["selected"] for c in cs] == [False, True]
    assert cs[0]["lexical_divergence"] == 0.0
    assert cs[1]["lexical_divergence"] == 1.0
    assert cs[0]["evaluation"]["is_watermarked"] is True
    assert cs[1]["evaluation"]["is_watermarked"] is False
    # before/after detection on the original and the final output
    mk = info["markllm"]
    assert mk["before"]["is_watermarked"] is True
    assert mk["after"]["is_watermarked"] is False
    assert mk["cleared"] is True


def test_markllm_detector_parameterized_from_cli(monkeypatch):
    captured = {}

    class _Capture(_FakeMarkLLM):
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            super().__init__(**kwargs)

    monkeypatch.setattr(rewrite_text, "MarkLLMTextDetector", _Capture)
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: "alpha beta gamma delta")
    rewrite(
        "the cat sat on the mat",
        **_rewrite_candidates_kwargs(markllm_scheme="kgw", markllm_dir="/x"),
    )
    assert captured["kwargs"] == {
        "scheme": "kgw",
        "upstream_dir": "/x",
        "model": "facebook/opt-1.3b",
        "timeout": 180.0,
    }


def test_evaluates_all_candidates_and_selects_pass(monkeypatch):
    monkeypatch.setattr(rewrite_text, "MarkLLMTextDetector", _FakeMarkLLM)
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: "alpha beta gamma delta")
    out, info = rewrite(
        "the cat sat on the mat",
        **_rewrite_candidates_kwargs(
            candidates=3, max_loops=3, markllm_scheme="kgw", markllm_dir="/x"
        ),
    )
    assert out == "alpha beta gamma delta"
    assert info["candidates"] == 3
    assert info["max_loops"] == 3
    # All three candidates pass; they are evaluated and the least-divergent
    # one is selected (all identical here, so the first).
    assert info["attempts_made"] == 3
    assert info["passed"] is True
    assert info["candidate_scores"][0]["selected"] is True


def test_loop_exhausts_max_attempts_without_pass(monkeypatch):
    class _NeverClears:
        name = "markllm"

        def __init__(self, **kwargs):
            pass

        def available(self):
            return True

        def detect(self, text):
            score = {"aaa": 3.0, "bbb": 2.0, "ccc": 1.0}.get(text, 2.5)
            return {
                "detector": "markllm",
                "available": True,
                "is_watermarked": True,
                "score": score,
            }

    monkeypatch.setattr(rewrite_text, "MarkLLMTextDetector", _NeverClears)
    texts = iter(["aaa", "bbb", "ccc"])
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: next(texts))
    out, info = rewrite(
        "the cat sat on the mat",
        **_rewrite_candidates_kwargs(candidates=3, markllm_scheme="kgw", markllm_dir="/x"),
    )
    assert out == "ccc"  # best-effort: lowest watermark score wins
    assert info["attempts_made"] == 3
    assert info["max_loops"] == 1
    assert info["passed"] is False
    assert all(c["passed"] is False for c in info["candidate_scores"])
    selected = [c for c in info["candidate_scores"] if c["selected"]]
    assert len(selected) == 1
    assert selected[0]["evaluation"]["score"] == 1.0
    assert info["markllm"]["cleared"] is False
    assert "Exhausted" in info["note"]


def test_max_loops_retry_new_variants_until_pass(monkeypatch):
    class _PassOnThird:
        name = "markllm"

        def __init__(self, **kwargs):
            pass

        def available(self):
            return True

        def detect(self, text):
            wm = text in ("aaa", "bbb")
            score = {"aaa": 3.0, "bbb": 2.0}.get(text, 0.5)
            return {
                "detector": "markllm",
                "available": True,
                "is_watermarked": wm,
                "score": score,
            }

    monkeypatch.setattr(rewrite_text, "MarkLLMTextDetector", _PassOnThird)
    texts = iter(["aaa", "bbb", "ccc"])
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: next(texts))
    out, info = rewrite(
        "the cat sat on the mat",
        **_rewrite_candidates_kwargs(
            candidates=1, max_loops=3, markllm_scheme="kgw", markllm_dir="/x"
        ),
    )
    assert out == "ccc"  # loops retry new variants until an evaluation passes
    assert info["max_loops"] == 3
    assert info["attempts_made"] == 3
    assert info["passed"] is True
    assert [c["loop"] for c in info["candidate_scores"]] == [0, 1, 2]
    assert [c["passed"] for c in info["candidate_scores"]] == [False, False, True]
    assert info["candidate_scores"][2]["selected"] is True


def test_max_loops_exhausted_across_loops(monkeypatch):
    class _NeverClears:
        name = "markllm"

        def __init__(self, **kwargs):
            pass

        def available(self):
            return True

        def detect(self, text):
            score = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}.get(text, 2.5)
            return {
                "detector": "markllm",
                "available": True,
                "is_watermarked": True,
                "score": score,
            }

    monkeypatch.setattr(rewrite_text, "MarkLLMTextDetector", _NeverClears)
    texts = iter(["a", "b", "c", "d"])
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: next(texts))
    out, info = rewrite(
        "the cat sat on the mat",
        **_rewrite_candidates_kwargs(
            candidates=2, max_loops=2, markllm_scheme="kgw", markllm_dir="/x"
        ),
    )
    assert info["max_loops"] == 2
    assert info["attempts_made"] == 4  # 2 loops x 2 candidates
    assert info["passed"] is False
    assert out == "d"  # best-effort: lowest watermark score across all loops
    assert [c["loop"] for c in info["candidate_scores"]] == [0, 0, 1, 1]
    selected = [c for c in info["candidate_scores"] if c["selected"]]
    assert len(selected) == 1 and selected[0]["evaluation"]["score"] == 1.0


def test_evaluator_fail_soft_verdict_unavailable(monkeypatch):
    class _Boom:
        name = "markllm"

        def __init__(self, **kwargs):
            pass

        def available(self):
            return True

        def detect(self, text):
            raise RuntimeError("detector exploded")

    monkeypatch.setattr(rewrite_text, "MarkLLMTextDetector", _Boom)
    _two_candidates(monkeypatch)
    out, info = rewrite(
        "the cat sat on the mat",
        **_rewrite_candidates_kwargs(markllm_scheme="kgw", markllm_dir="/x"),
    )
    # no verdicts available: the loop runs all attempts, falls back to
    # divergence, and never fails the rewrite
    assert out == "alpha beta gamma delta"
    assert info["passed"] is False
    assert info["attempts_made"] == 2
    entry = info["candidate_scores"][0]["evaluation"]
    assert entry["available"] is False
    assert "exploded" in entry["error"]
    assert info["markllm"]["before"]["available"] is False
    assert info["markllm"]["cleared"] is None


def test_single_candidate_attempt(monkeypatch):
    monkeypatch.setattr(rewrite_text, "MarkLLMTextDetector", _FakeMarkLLM)
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: "REWRITTEN OUTPUT")
    out, info = rewrite(
        "the cat sat on the mat",
        **_rewrite_candidates_kwargs(candidates=1, markllm_scheme="kgw", markllm_dir="/x"),
    )
    assert out == "REWRITTEN OUTPUT"
    assert info["candidates"] == 1
    assert info["max_loops"] == 1
    assert info["attempts_made"] == 1
    assert info["passed"] is True
    assert len(info["candidate_scores"]) == 1
    assert info["candidate_scores"][0]["selected"] is True
    assert info["markllm"]["cleared"] is True


# ---------------------------------------------------------------------------
# Robust-removal margin selection (--target-margin / --select) and chunk mode
# ---------------------------------------------------------------------------


def test_candidate_pass_treats_pvalue_threshold_as_no_margin():
    # keyed-Gumbel reports a p-value threshold (1e-06) that does not scale with
    # its score, so threshold - score is meaningless and must not gate a clear.
    gumbel_report = {"available": True, "is_watermarked": False, "score": 0.21, "threshold": 1e-06}
    assert _candidate_pass(gumbel_report, 0.0) == (True, None, None)
    assert _candidate_pass(gumbel_report, 0.5) == (True, None, None)


def test_target_margin_gates_small_margin_pass(monkeypatch):
    """A not-watermarked candidate whose margin is below --target-margin is
    not counted as a pass; the loop must keep the margin-meeting candidate."""

    class _TinyMargin:
        name = "markllm"

        def __init__(self, **kwargs):
            pass

        def available(self):
            return True

        def detect(self, text):
            if text == "tiny":
                return {"available": True, "is_watermarked": False, "score": 2.95, "threshold": 3.0}
            if text == "big":
                return {"available": True, "is_watermarked": False, "score": 1.0, "threshold": 3.0}
            return {"available": True, "is_watermarked": True, "score": 3.0, "threshold": 3.0}

    monkeypatch.setattr(rewrite_text, "MarkLLMTextDetector", _TinyMargin)
    texts = iter(["tiny", "big"])  # margins 0.05 then 2.0
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: next(texts))
    out, info = rewrite(
        "the cat sat on the mat",
        **_rewrite_candidates_kwargs(
            candidates=2, markllm_scheme="kgw", markllm_dir="/x", target_margin=1.0
        ),
    )
    # "tiny" is not watermarked but its margin (0.05) never reaches the 1.0
    # floor, so it is gated; "big"'s margin (2.0) clears the floor and wins.
    assert out == "big"
    assert info["passed"] is True
    cs = info["candidate_scores"]
    # "tiny" is gated: it cleared the watermark detector but not the margin
    # floor, so it contributes no pass verdict; "big" meets the floor and wins.
    assert cs[0]["passed"] is None
    assert cs[0]["selected"] is False
    assert cs[1]["passed"] is True
    assert cs[1]["selected"] is True


def test_select_max_margin_prefers_largest_margin(monkeypatch):
    class _TwoMargins:
        name = "markllm"

        def __init__(self, **kwargs):
            pass

        def available(self):
            return True

        def detect(self, text):
            return {
                "available": True,
                "is_watermarked": False,
                "score": {"low": 2.0, "high": 1.0}[text],
                "threshold": 3.0,
            }

    monkeypatch.setattr(rewrite_text, "MarkLLMTextDetector", _TwoMargins)
    texts = iter(["low", "high"])  # margins 1.0 then 2.0
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: next(texts))
    out, info = rewrite(
        "the cat sat on the mat",
        **_rewrite_candidates_kwargs(
            candidates=2, markllm_scheme="kgw", markllm_dir="/x", selection="max-margin"
        ),
    )
    assert out == "high"  # largest margin wins, not least divergence
    assert info["passed"] is True
    cs = info["candidate_scores"]
    assert cs[0]["margin"] == 1.0
    assert cs[1]["margin"] == 2.0
    assert cs[1]["selected"] is True


def test_tactic_chunk_reassembles_fragments(monkeypatch):
    calls = []

    def fake_ollama(base_url, model, prompt, timeout, temperature, reasoning_effort=None):
        calls.append(prompt.split("---")[-1].strip())
        return "RE: " + prompt.split("---")[-1].strip()

    monkeypatch.setattr(rewrite_text, "call_ollama", fake_ollama)
    text = "First sentence. Second sentence!\n\nThird paragraph?"
    out, info = rewrite(
        text,
        backend="ollama",
        model="m",
        base_url="http://127.0.0.1:11434",
        api_key=None,
        tactic="chunk",
        lang="French",
        original_lang="English",
        timeout=5.0,
        layer_a_after=False,
        temperature=0.9,
        candidates=1,
    )
    assert info["chunked"] is True
    assert info["chunk_shuffle"] is False
    # each fragment is rewritten with a fresh context (fresh per-token key)
    assert calls == ["First sentence.", "Second sentence!", "Third paragraph?"]
    assert out == "RE: First sentence. RE: Second sentence!\n\nRE: Third paragraph?"


def test_tactic_chunk_leading_blank_line_kept(monkeypatch):
    calls = []

    def fake_ollama(base_url, model, prompt, timeout, temperature, reasoning_effort=None):
        calls.append(prompt.split("---")[-1].strip())
        return "RE: " + prompt.split("---")[-1].strip()

    monkeypatch.setattr(rewrite_text, "call_ollama", fake_ollama)
    # A blank line at the top is a separator, not a fragment: it must not be
    # sent to the backend, but unshuffled mode keeps it in the reassembly.
    text = "\n\nFirst sentence. Second sentence!"
    out, _ = rewrite(
        text,
        backend="ollama",
        model="m",
        base_url="http://127.0.0.1:11434",
        api_key=None,
        tactic="chunk",
        lang="French",
        original_lang="English",
        timeout=5.0,
        layer_a_after=False,
        temperature=0.9,
        candidates=1,
    )
    assert calls == ["First sentence.", "Second sentence!"]
    assert out == "\n\nRE: First sentence. RE: Second sentence!"


def test_tactic_chunk_shuffle_reorders_fragments(monkeypatch):
    calls = []

    def fake_ollama(base_url, model, prompt, timeout, temperature, reasoning_effort=None):
        calls.append(prompt.split("---")[-1].strip())
        return "RE: " + prompt.split("---")[-1].strip()

    monkeypatch.setattr(rewrite_text, "call_ollama", fake_ollama)
    monkeypatch.setattr(rewrite_text.random, "shuffle", lambda units: units.reverse())
    text = "First sentence. Second sentence!\n\nThird paragraph?"
    out, info = rewrite(
        text,
        backend="ollama",
        model="m",
        base_url="http://127.0.0.1:11434",
        api_key=None,
        tactic="chunk",
        lang="French",
        original_lang="English",
        timeout=5.0,
        layer_a_after=False,
        temperature=0.9,
        candidates=1,
        chunk_shuffle=True,
    )
    assert info["chunk_shuffle"] is True
    assert calls == ["Third paragraph?", "Second sentence!", "First sentence."]
    assert out == "RE: Third paragraph? RE: Second sentence! RE: First sentence."


# ---------------------------------------------------------------------------
# HTTP client hardening: default-deny allowlist, scheme guard, no redirects
# ---------------------------------------------------------------------------


def _rewrite_http_kwargs(base_url: str, **overrides):
    kwargs = dict(
        backend="openai-compatible",
        model="m",
        base_url=base_url,
        api_key="sk-test-key-123",
        tactic="paraphrase",
        lang="French",
        original_lang="English",
        timeout=5.0,
        layer_a_after=False,
        temperature=0.9,
        candidates=1,
    )
    kwargs.update(overrides)
    return kwargs


def test_check_remote_loopback_allowed_without_opt_in():
    # Must not raise.
    _check_remote("http://127.0.0.1:11434", allow_remote=False)
    _check_remote("http://localhost:11434", allow_remote=False)
    _check_remote("http://[::1]:11434", allow_remote=False)


def test_check_remote_denies_non_loopback_without_opt_in():
    with pytest.raises(SystemExit):
        _check_remote("http://example.com:11434", allow_remote=False)


def test_check_remote_allows_non_loopback_with_opt_in(capsys):
    _check_remote("http://example.com:11434", allow_remote=True)
    err = capsys.readouterr().err
    assert "content will leave this machine" in err


def test_check_remote_denies_non_http_scheme():
    with pytest.raises(SystemExit):
        _check_remote("file:///etc/passwd", allow_remote=True)


def test_flag_env(monkeypatch):
    assert not _flag_env("WATERMARKS_REWRITE_ALLOW_REMOTE")
    monkeypatch.setenv("WATERMARKS_REWRITE_ALLOW_REMOTE", "1")
    assert _flag_env("WATERMARKS_REWRITE_ALLOW_REMOTE")
    monkeypatch.setenv("WATERMARKS_REWRITE_ALLOW_REMOTE", "true")
    assert _flag_env("WATERMARKS_REWRITE_ALLOW_REMOTE")
    monkeypatch.setenv("WATERMARKS_REWRITE_ALLOW_REMOTE", "0")
    assert not _flag_env("WATERMARKS_REWRITE_ALLOW_REMOTE")


def test_openai_compatible_sends_reasoning_effort_when_set():
    captured = {}

    class Collector(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            captured["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"choices": [{"message": {"content": "rewritten"}}]}')

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        result, _ = rewrite(
            "hello",
            **_rewrite_http_kwargs(
                f"http://127.0.0.1:{server.server_address[1]}",
                reasoning_effort="none",
            ),
        )
        assert result == "rewritten"
        assert captured["body"]["reasoning_effort"] == "none"

        captured.clear()
        rewrite(
            "hello",
            **_rewrite_http_kwargs(
                f"http://127.0.0.1:{server.server_address[1]}",
                reasoning_effort=None,
            ),
        )
        assert "reasoning_effort" not in captured["body"]
    finally:
        server.shutdown()


def test_ollama_sends_think_false_only_for_reasoning_effort_none():
    # Ollama runs a thinking model's reasoning by default (gemma4:12b spent
    # ~200 s on a two-word paraphrase), while "think": true is an error on
    # models without a thinking mode -- so only "none" may send the flag.
    bodies = []

    class Collector(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"message": {"content": "rewritten"}}')

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        for effort in ("none", "high", None):
            result, _ = rewrite(
                "hello",
                **_rewrite_http_kwargs(
                    base_url, backend="ollama", api_key=None, reasoning_effort=effort
                ),
            )
            assert result == "rewritten"
    finally:
        server.shutdown()

    assert bodies[0]["think"] is False
    assert "think" not in bodies[1]
    assert "think" not in bodies[2]


def test_rewrite_denies_remote_host_without_opt_in():
    with pytest.raises(SystemExit):
        rewrite("secret text", **_rewrite_http_kwargs("http://example.com:11434"))


def test_rewrite_blocks_redirect_and_never_sends_key():
    """A 302 from the (loopback) endpoint must not re-send the API key to the
    redirect target — the request must fail instead."""
    state: dict = {"collector_port": None}
    captured: dict = {}

    class Redirector(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                self.rfile.read(content_length)
            self.send_response(302)
            self.send_header(
                "Location",
                f"http://127.0.0.1:{state['collector_port']}/collect",
            )
            self.end_headers()

        def log_message(self, format, *args):
            pass

    class Collector(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            captured["auth"] = self.headers.get("Authorization")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"choices": [{"message": {"content": "rewritten"}}]}')

        def log_message(self, format, *args):
            pass

    collector = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    redirector = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Redirector)
    state["collector_port"] = collector.server_address[1]
    threading.Thread(target=collector.serve_forever, daemon=True).start()
    threading.Thread(target=redirector.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError):
            rewrite(
                "secret text",
                **_rewrite_http_kwargs(f"http://127.0.0.1:{redirector.server_address[1]}"),
            )
        time.sleep(0.2)
        assert captured == {}, "redirect target received a request (key leak?)"
    finally:
        collector.shutdown()
        redirector.shutdown()


def test_candidate_pass_raw_margin_precision():
    # 0.50000 - 0.37656 = 0.12344, which is >= 0.123439 raw, but if rounded first could drift
    evaluation = {"is_watermarked": False, "score": 0.37656, "threshold": 0.50000}
    passed, margin, raw_margin = rewrite_text._candidate_pass(evaluation, target_margin=0.123439)
    assert passed is True
    assert margin == 0.1234
    assert raw_margin == pytest.approx(0.12344)


def test_rewrite_metadata_records_selection_and_target_margin(monkeypatch):
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: "rewritten")
    _out, info = rewrite(
        "sample input",
        **_rewrite_candidates_kwargs(candidates=1, target_margin=0.25, selection="max-margin"),
    )
    assert info["target_margin"] == 0.25
    assert info["selection"] == "max-margin"


def test_max_margin_ranks_by_p_value_when_margins_tied():
    rec_high_p = {
        "passed": True,
        "margin": None,
        "raw_margin": None,
        "evaluation": {"is_watermarked": False, "p_value": 0.05},
        "lexical_divergence": 0.5,
    }
    rec_low_p = {
        "passed": True,
        "margin": None,
        "raw_margin": None,
        "evaluation": {"is_watermarked": False, "p_value": 1e-6},
        "lexical_divergence": 0.5,
    }
    # rec_low_p has a lower p-value (1e-6 < 0.05), so it represents a safer pass and ranks higher
    assert rewrite_text._margin_of(rec_low_p) > rewrite_text._margin_of(rec_high_p)


def test_max_margin_ranks_by_raw_margin_when_rounded_margins_tie():
    # 0.12343 and 0.12344 both round to 0.1234 for telemetry, so ranking on the
    # rounded value would let the p-value tie-breaker pick the smaller raw margin.
    rec_small_raw = {
        "passed": True,
        "margin": 0.1234,
        "raw_margin": 0.12343,
        "evaluation": {"is_watermarked": False},
        "lexical_divergence": 0.5,
    }
    rec_large_raw = {
        "passed": True,
        "margin": 0.1234,
        "raw_margin": 0.12344,
        "evaluation": {"is_watermarked": False},
        "lexical_divergence": 0.5,
    }
    assert rewrite_text._margin_of(rec_large_raw) > rewrite_text._margin_of(rec_small_raw)


def test_rewrite_noop_guard_flags_verbatim_output(monkeypatch):
    # A rewrite that returns the input unchanged must be flagged noop, so a
    # benchmark never reads it as "0% clear".
    text = "the watermark removal theory is interesting to study"
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: text)
    out, info = rewrite(
        text,
        **_rewrite_kwargs(
            backend="ollama",
            model="m",
            base_url="http://127.0.0.1:11434",
            noop_lex_floor=0.05,
        ),
    )
    assert info["noop"] is True
    assert out == text


# ---------------------------------------------------------------------------
# Model-wrapper stripping and the length-drift guard
# ---------------------------------------------------------------------------

PROBE_INPUT = "The weather was mild and the meeting ended early.\n"
PROBE_REWRITE = "The atmosphere was agreeable, and the conference concluded ahead of schedule."
# What a chatty model (Ollama backend) returned for PROBE_INPUT in the bug report:
# the rewrite between a "Here is ...:" preamble and a list of the changes.
WRAPPED = (ROOT / "tests" / "fixtures" / "rewrite_wrapped_ollama.txt").read_text(encoding="utf-8")
# Commentary inside the rewrite's own paragraph: nothing to strip, but ~5x longer.
CHATTY_INLINE = (
    f"{PROBE_REWRITE} In this version the nouns and verbs were swapped for synonyms, "
    "the clause order was kept, and roughly eighty percent of the tokens differ from "
    "the source sentence, which matches the requested intensity."
)


def test_strip_model_wrappers_removes_preamble_and_change_list():
    out, removed = strip_model_wrappers(WRAPPED, PROBE_INPUT)
    assert out == PROBE_REWRITE
    assert removed == ["preamble", "trailer"]


@pytest.mark.parametrize(
    ("wrapped", "removed"),
    [
        ("Sure! Here's a paraphrased version:\n\nThe sky was clear.", ["preamble"]),
        ("Sure!\n\nHere is the rewritten text:\n\nThe sky was clear.", ["preamble"]),
        ("Here is the rewritten text: The sky was clear.", ["preamble"]),
        ("**Rewritten text:**\n\nThe sky was clear.", ["preamble"]),
        ("The sky was clear.\n\nChanges made:\n- day -> sky", ["trailer"]),
        ("The sky was clear.\n\n**Note:** every fact was kept.", ["trailer"]),
        ("The sky was clear.\n\n(Note: I kept every fact.)", ["trailer"]),
        ("The sky was clear.\n\nLet me know if you'd like any other changes!", ["trailer"]),
        ("The sky was clear.\n\nThis rewrite keeps every fact of the original.", ["trailer"]),
        ("```\nThe sky was clear.\n```", ["code_fence"]),
        ("\u201cThe sky was clear.\u201d", ["quotes"]),
        ('"The sky was clear."', ["quotes"]),
        ("<think>Swap a few words.</think>\n\nThe sky was clear.", ["think"]),
        (
            "Here is the rewritten text:\n\n---\n\nThe sky was clear.\n\n---\n\nNote: kept facts.",
            ["preamble", "separator", "trailer"],
        ),
        (
            "Here is the rewritten text:\r\n\r\nThe sky was clear.\r\n\r\n---\r\n\r\nNote: kept.",
            ["preamble", "trailer", "separator"],
        ),
    ],
)
def test_strip_model_wrappers_common_wrappers(wrapped, removed):
    assert strip_model_wrappers(wrapped, "It was a clear day.") == ("The sky was clear.", removed)
    assert set(removed) <= set(rewrite_text.WRAPPER_KINDS)


def test_strip_model_wrappers_unwraps_code_and_keeps_indentation():
    original = "def f(x):\n    return sum(x)\n"
    wrapped = (
        "Here's the updated code:\n\n```python\ndef f(values):\n    return sum(values)\n```"
        "\n\nChanges made:\n- renamed x to values"
    )
    assert strip_model_wrappers(wrapped, original) == (
        "def f(values):\n    return sum(values)",
        ["preamble", "trailer", "code_fence"],
    )


@pytest.mark.parametrize(
    ("text", "original"),
    [
        # A "Here is ...:" line that does not mention the rewrite is content.
        ("Here's what you need to know:\n\n- pack water", "This is what matters:\n\n- pack water"),
        # The input has its own "Note:" paragraph.
        ("The sky was clear.\n\nNote: carry water.", "It was a clear day.\n\nNote: bring water."),
        # First-person input: "I kept ..." is narrative, not commentary.
        (
            "I stuck with the plan.\n\nI kept the original route.",
            "I kept to the plan.\n\nI took the first route.",
        ),
        # A lone interjection may be dialogue.
        ("Sure.\n\nThe sky was clear.", "Yes.\n\nIt was a clear day."),
        # Quoted or fenced input stays quoted or fenced.
        ('"The sky was clear."', '"It was a clear day."'),
        ("```\nprint(1)\n```", "```\nprint(2)\n```"),
        # A code comment is not commentary.
        ("# Note: fast path\ndef f():\n    pass", "# Remark: fast path\ndef f():\n    pass"),
        # Ordinary closing lines that only look like model sign-offs.
        (
            "The budget grew.\n\nIn the revised budget, rent is lower.",
            "Spending rose.\n\nUnder the updated budget, rent drops.",
        ),
        (
            "Your order has shipped.\n\nIf you need changes to your order, call us.",
            "The order is on its way.\n\nCall us to modify the order.",
        ),
        # A "Here are ...:" lead-in that is not about the rewrite is content.
        (
            "We met twice. Here are the revised dates:\n\n- May 3\n- May 9",
            "There were two meetings. The new dates follow:\n\n- May 3\n- May 9",
        ),
        # Prompt-like words in ordinary prose are not an echo of the prompt.
        (
            "Training was hard.\n\nHigh-intensity intervals helped, by the same token.",
            "Workouts were tough.\n\nStrenuous intervals helped, likewise.",
        ),
        # Nothing would be left after the preamble.
        ("Here is the rewritten text:", "x y z"),
        # Nothing to strip: returned byte for byte, whitespace included.
        ("  The sky was clear.  \n", "It was a clear day."),
    ],
)
def test_strip_model_wrappers_keeps_content(text, original):
    assert strip_model_wrappers(text, original) == (text, [])


LLAMA32 = json.loads(
    (ROOT / "tests" / "fixtures" / "rewrite_wrapped_llama32.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("sample", LLAMA32["samples"], ids=lambda s: s["rewrite"][:24])
def test_strip_model_wrappers_real_llama32_commentary(sample):
    # Commentary that echoes the prompt ("At low intensity:", "0.32 tokens",
    # "Function words:", "Modulation adjustment:") after the rewrite.
    out, removed = strip_model_wrappers(sample["raw"], LLAMA32["input"])
    assert out == sample["rewrite"]
    assert "trailer" in removed
    assert not rewrite_text._length_drift(len(LLAMA32["input"]), len(out))


def test_strip_model_wrappers_keeps_prompt_jargon_the_input_uses():
    # The same echo phrases are content when the input already uses them.
    original = "Tokenizers split text.\n\nAbout 30 tokens fit in a line."
    rewrite = "A tokenizer splits text.\n\nRoughly 30 tokens fit on one line."
    assert strip_model_wrappers(rewrite, original) == (rewrite, [])


def test_length_drift_bounds_and_short_input_slack():
    assert not rewrite_text._length_drift(len(PROBE_INPUT), len(PROBE_REWRITE))
    assert rewrite_text._length_drift(len(PROBE_INPUT), len(WRAPPED))
    assert rewrite_text._length_drift(400, 150)  # truncated
    assert not rewrite_text._length_drift(3, 12)  # "Hi." -> "Hello there.": within slack
    assert not rewrite_text._length_drift(50, 111)  # a verbose one-sentence paraphrase
    assert not rewrite_text._length_drift(0, 10)


def test_select_candidate_skips_length_drift():
    best, scores = _select_candidate(PROBE_INPUT, [PROBE_REWRITE, WRAPPED])
    assert best == PROBE_REWRITE
    # Divergence alone would have picked the commentary-laden candidate.
    assert scores[1] > scores[0]


def test_rewrite_strips_wrapped_output_from_fake_backend(monkeypatch):
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: WRAPPED)
    out, info = rewrite(PROBE_INPUT, **_rewrite_candidates_kwargs(candidates=1))
    assert out == PROBE_REWRITE
    assert info["length_guard"] is True
    assert info["wrappers_stripped"] == ["preamble", "trailer"]
    assert info["length_drift_rejected"] == 0
    rec = info["candidate_scores"][0]
    assert rec["length_drift"] is False
    assert rec["length_ratio"] == round(len(PROBE_REWRITE) / len(PROBE_INPUT), 4)


def test_rewrite_chunk_strips_wrappers_per_fragment(monkeypatch):
    def fake_ollama(base_url, model, prompt, timeout, temperature, reasoning_effort=None):
        fragment = prompt.split("---")[-1].strip()
        return f"Here is the rewritten fragment:\n\nRE: {fragment}\n\nNote: kept the facts."

    monkeypatch.setattr(rewrite_text, "call_ollama", fake_ollama)
    out, info = rewrite(
        "First sentence. Second sentence!",
        **_rewrite_candidates_kwargs(tactic="chunk", candidates=1),
    )
    assert out == "RE: First sentence. RE: Second sentence!"
    assert info["wrappers_stripped"] == ["preamble", "trailer"]


def test_rewrite_rejects_length_drift_even_when_more_divergent(monkeypatch):
    texts = iter([CHATTY_INLINE, PROBE_REWRITE])
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: next(texts))
    out, info = rewrite(PROBE_INPUT, **_rewrite_candidates_kwargs(candidates=2))
    # The lexical evaluator maximises divergence, which the padded candidate
    # wins; the length guard rejects it anyway.
    assert out == PROBE_REWRITE
    cs = info["candidate_scores"]
    assert cs[0]["lexical_divergence"] > cs[1]["lexical_divergence"]
    assert cs[0]["length_drift"] is True
    assert cs[0]["passed"] is False
    assert [c["selected"] for c in cs] == [False, True]
    assert info["length_drift_rejected"] == 1


def test_rewrite_length_drift_retries_within_loop_budget(monkeypatch):
    # _FakeMarkLLM reports the padded candidate not watermarked, so without the
    # guard the loop would stop on it; instead it is a failed attempt and the
    # next loop generates a fresh variant.
    monkeypatch.setattr(rewrite_text, "MarkLLMTextDetector", _FakeMarkLLM)
    texts = iter([CHATTY_INLINE, PROBE_REWRITE])
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: next(texts))
    out, info = rewrite(
        PROBE_INPUT,
        **_rewrite_candidates_kwargs(
            candidates=1, max_loops=2, markllm_scheme="kgw", markllm_dir="/x"
        ),
    )
    assert out == PROBE_REWRITE
    assert info["attempts_made"] == 2
    assert info["passed"] is True
    cs = info["candidate_scores"]
    assert [c["passed"] for c in cs] == [False, True]
    assert "length drifted" in cs[0]["evaluation"]["error"]
    assert info["markllm"]["after"]["is_watermarked"] is False


def test_rewrite_fails_when_every_attempt_drifts(monkeypatch):
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: CHATTY_INLINE)
    with pytest.raises(RuntimeError, match="drifted in length"):
        rewrite(PROBE_INPUT, **_rewrite_candidates_kwargs(candidates=2))


def test_structural_tactic_is_not_length_guarded(monkeypatch):
    # structural rebuilds the text from an outline: its length legitimately drifts.
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: CHATTY_INLINE)
    out, info = rewrite(
        PROBE_INPUT, **_rewrite_candidates_kwargs(tactic="structural", candidates=1)
    )
    assert out == CHATTY_INLINE
    assert info["length_guard"] is False
    assert info["candidate_scores"][0]["length_drift"] is False


def _cli_argv(src, dest, *extra):
    return [
        "rewrite_text.py",
        str(src),
        "-o",
        str(dest),
        "--backend",
        "ollama",
        "--model",
        "m",
        "--base-url",
        "http://127.0.0.1:11434",
        *extra,
    ]


@pytest.fixture
def _cli_env(monkeypatch):
    for var in (
        "WATERMARKS_REWRITE_CANDIDATES",
        "WATERMARKS_REWRITE_LOOPS",
        "WATERMARKS_GUMBEL_KEY",
        "WATERMARKS_REWRITE_ALLOW_REMOTE",
    ):
        monkeypatch.delenv(var, raising=False)


def test_cli_exits_nonzero_when_every_attempt_drifts(monkeypatch, tmp_path, capsys, _cli_env):
    src = tmp_path / "in.txt"
    src.write_text(PROBE_INPUT, encoding="utf-8")
    dest = tmp_path / "out.txt"
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: CHATTY_INLINE)
    monkeypatch.setattr(sys, "argv", _cli_argv(src, dest))
    assert rewrite_text.main() == 1
    assert "drifted in length" in capsys.readouterr().err
    assert not dest.exists()


def test_cli_strategy_retries_drift_within_candidates_budget(monkeypatch, tmp_path, _cli_env):
    src = tmp_path / "in.txt"
    src.write_text(PROBE_INPUT, encoding="utf-8")
    dest = tmp_path / "out.txt"
    texts = iter([CHATTY_INLINE, WRAPPED])
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: next(texts))
    monkeypatch.setattr(
        sys, "argv", _cli_argv(src, dest, "--strategy", "paraphrase@0.8", "--candidates", "2")
    )
    assert rewrite_text.main() == 0
    assert dest.read_text(encoding="utf-8") == PROBE_REWRITE


# ---------------------------------------------------------------------------
# apply_strategy: two-generation tactics and the no-op guard
# ---------------------------------------------------------------------------

_SOURCE = "The gluon propagator is suppressed in the infrared below the Gribov horizon."


class _ShortcutModel:
    """Stub ollama backend that shortcuts the way the real model did on /clean.

    Scripted prompts get their scripted answer; any other prompt (notably the
    combined one-shot backtranslate/structural prompt) gets the original text
    echoed back verbatim. Prompts are recorded in call order.
    """

    def __init__(self, original: str, script: dict[str, str] | None = None):
        self.original = original
        self.script = script or {}
        self.prompts: list[str] = []

    def __call__(self, base_url, model, prompt, timeout, temperature, reasoning_effort=None):
        self.prompts.append(prompt)
        return self.script.get(prompt, self.original)


def _strategy_kwargs(**overrides):
    kwargs = dict(backend="ollama", model="m", base_url="http://127.0.0.1:11434", api_key=None)
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    ("tactic", "keys", "intermediate", "final"),
    [
        pytest.param(
            "backtranslate",
            ("backtranslate_out", "backtranslate_back"),
            "Le propagateur du gluon est supprimé dans l'infrarouge sous l'horizon de Gribov.",
            "Below the Gribov horizon, the gluon propagator is suppressed in the infrared.",
            id="backtranslate",
        ),
        pytest.param(
            "structural",
            ("structural_outline", "structural_write"),
            "- gluon propagator: suppressed in the infrared\n- regime: below the Gribov horizon",
            "In the infrared, below the Gribov horizon, the gluon propagator is suppressed.",
            id="structural",
        ),
    ],
)
def test_strategy_two_step_tactic_runs_two_generations_in_order(
    monkeypatch, tactic, keys, intermediate, final
):
    langs = {"LANG": "French", "ORIGINAL_LANG": "English"}
    first_prompt = rewrite_text.PROMPTS[keys[0]].format(TEXT=_SOURCE, **langs)
    final_prompt = rewrite_text.PROMPTS[keys[1]].format(TEXT=intermediate, **langs)
    # What the strategy path used to send; this model shortcuts it (echo).
    combined_prompt = build_prompt(tactic, _SOURCE, rewrite_level=0.8)
    model = _ShortcutModel(
        _SOURCE,
        {combined_prompt: _SOURCE, first_prompt: intermediate, final_prompt: final},
    )
    monkeypatch.setattr(rewrite_text, "call_ollama", model)

    out, stats = rewrite_text.apply_strategy(_SOURCE, [(tactic, 0.8)], **_strategy_kwargs())

    # Two generations, in order, and the final one sees only the intermediate
    # (pivot translation / outline), never the source. The exact match also
    # pins that no intensity clause rides on either prompt.
    assert model.prompts == [first_prompt, final_prompt]
    assert out == final
    assert stats["steps"][0]["generations"] == 2
    assert stats["noop"] is False
    assert stats["warnings"] == []


@pytest.mark.parametrize("tactic", ["backtranslate", "structural"])
def test_strategy_two_step_style_joins_final_prompt_only(monkeypatch, tactic):
    model = _ShortcutModel(_SOURCE)
    monkeypatch.setattr(rewrite_text, "call_ollama", model)

    rewrite_text.apply_strategy(
        _SOURCE, [(tactic, 0.3)], **_strategy_kwargs(style="terse and plain")
    )

    first_prompt, final_prompt = model.prompts
    style_clause = rewrite_text._style_clause("terse and plain")
    assert style_clause not in first_prompt
    assert final_prompt.endswith(style_clause)
    # A token-fraction request has no meaning for a translation or an outline.
    assert rewrite_text._intensity_clause(0.3) not in first_prompt + final_prompt


def test_strategy_two_step_empty_intermediate_raises(monkeypatch):
    first_prompt = rewrite_text.PROMPTS["backtranslate_out"].format(TEXT=_SOURCE, LANG="French")
    model = _ShortcutModel(_SOURCE, {first_prompt: "  \n"})
    monkeypatch.setattr(rewrite_text, "call_ollama", model)

    with pytest.raises(RuntimeError, match="first generation returned empty output"):
        rewrite_text.apply_strategy(_SOURCE, [("backtranslate", 0.8)], **_strategy_kwargs())
    assert model.prompts == [first_prompt]  # nothing is written from an empty pivot


def test_strategy_noop_guard_flags_verbatim_output(monkeypatch, capsys):
    # A model that shortcuts even the two-step prompts hands the input back;
    # that must be reported as a no-op, not passed off as a rewrite.
    monkeypatch.setattr(rewrite_text, "call_ollama", _ShortcutModel(_SOURCE))

    out, stats = rewrite_text.apply_strategy(
        _SOURCE, [("backtranslate", 0.8)], **_strategy_kwargs()
    )

    assert out == _SOURCE
    assert stats["noop"] is True
    assert stats["lexical_divergence"] == 0.0
    assert stats["noop_lex_floor"] == rewrite_text.DEFAULT_NOOP_LEX_FLOOR
    assert stats["steps"][0]["noop"] is True
    assert len(stats["warnings"]) == 1
    assert "no-op" in stats["warnings"][0]
    assert "no-op" in capsys.readouterr().err


def test_strategy_noop_guard_disabled_by_zero_floor(monkeypatch):
    monkeypatch.setattr(rewrite_text, "call_ollama", _ShortcutModel(_SOURCE))

    _out, stats = rewrite_text.apply_strategy(
        _SOURCE, [("paraphrase", 0.8)], **_strategy_kwargs(noop_lex_floor=0)
    )

    assert stats["noop"] is False
    assert stats["steps"][0]["noop"] is False
    assert stats["warnings"] == []


def test_strategy_names_noop_step_hidden_by_later_step(monkeypatch):
    paraphrased = "Beneath the Gribov horizon, infrared gluon propagation is damped."
    paraphrase_prompt = build_prompt("paraphrase", _SOURCE, rewrite_level=0.5)
    # backtranslate is shortcut (echo); the paraphrase after it does rewrite.
    model = _ShortcutModel(_SOURCE, {paraphrase_prompt: paraphrased})
    monkeypatch.setattr(rewrite_text, "call_ollama", model)

    out, stats = rewrite_text.apply_strategy(
        _SOURCE, [("backtranslate", 0.8), ("paraphrase", 0.5)], **_strategy_kwargs()
    )

    assert out == paraphrased
    assert stats["noop"] is False
    assert [s["noop"] for s in stats["steps"]] == [True, False]
    assert len(stats["warnings"]) == 1
    assert stats["warnings"][0].startswith("step 1 (backtranslate@0.8)")


def test_cli_strategy_honors_noop_lex_floor(monkeypatch, tmp_path, capsys):
    src = tmp_path / "in.txt"
    src.write_text(_SOURCE, encoding="utf-8")
    monkeypatch.setattr(rewrite_text, "call_ollama", _ShortcutModel(_SOURCE))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rewrite_text.py",
            str(src),
            "-o",
            str(tmp_path / "out.txt"),
            "--backend",
            "ollama",
            "--model",
            "m",
            "--base-url",
            "http://127.0.0.1:11434",
            "--strategy",
            "backtranslate@0.8",
            "--noop-lex-floor",
            "0.5",
            "--json-stats",
        ],
    )

    assert rewrite_text.main() == 0
    err = capsys.readouterr().err
    stats = json.loads(err[err.index("{") :])
    assert stats["noop_lex_floor"] == 0.5
    assert stats["noop"] is True
