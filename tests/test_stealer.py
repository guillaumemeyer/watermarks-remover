"""Tests for the black-box watermark-stealing module (scorer core + downloader)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STEALER = ROOT / "stealer"
sys.path.insert(0, str(STEALER))

import download_prompts
import scorer
import tokens

WM_TEXTS = [
    "the quick brown fox jumps over the lazy dog . " * 20,
    "a red fox ran through the green field . " * 20,
]
BASE_TEXTS = [
    "the lazy dog sleeps by the fire all day . " * 20,
    "a blue car drove across the bridge slowly . " * 20,
]


def test_default_tokenize_handles_words_and_punctuation():
    toks = tokens.default_tokenize("Hello, World! 3.5")
    assert toks == ["hello", ",", "world", "!", "3", ".", "5"]


def test_count_ngrams_counts_next_tokens():
    wm, totals, unis = tokens.count_ngrams(["a b a b"], 1, tokens.default_tokenize)
    # sequence: tokens = [a, b, a, b]; pairs (a->b), (b->a), (a->b)
    assert wm[("a",)]["b"] == 2
    assert wm[("b",)]["a"] == 1
    assert totals[("a",)] == 2
    assert unis["a"] == 2 and unis["b"] == 2


def test_build_scorer_ranks_boosted_tokens_above_others():
    wm = tokens.count_ngrams(WM_TEXTS, 3)
    base = tokens.count_ngrams(BASE_TEXTS, 3)
    s = scorer.build_scorer(wm, base, context_len=3, topk=20)
    # "fox" appears in the watermarked corpus, not the baseline, so its score for
    # the context "the quick brown" should be positive and beat an unrelated token.
    entry = s["scorer"].get("the quick brown", [])
    scores = {item["token"]: item["score"] for item in entry}
    assert "fox" in scores
    assert scores["fox"] > 0


def test_apply_delta_demotes_green_token():
    wm = tokens.count_ngrams(WM_TEXTS, 3)
    base = tokens.count_ngrams(BASE_TEXTS, 3)
    s = scorer.build_scorer(wm, base, context_len=3, topk=20)
    logits = {"fox": 0.0, "cat": 0.0}
    adjusted = scorer.apply_delta(s, ("the", "quick", "brown"), logits, delta=1.0)
    assert adjusted["fox"] < 0.0  # green token demoted
    assert adjusted["cat"] == 0.0  # unknown token untouched


def test_score_sequence_applies_lookups():
    wm = tokens.count_ngrams(WM_TEXTS, 3)
    base = tokens.count_ngrams(BASE_TEXTS, 3)
    s = scorer.build_scorer(wm, base, context_len=3, topk=20)
    result = scorer.score_sequence(s, tokens.default_tokenize("the quick brown fox jumps"), 3)
    assert result["applied"] > 0


@pytest.fixture
def fake_pages():
    """Two datasets-server pages for the downloader test."""

    def make(offset):
        return {
            "num_rows_per_page": 2,
            "rows": [
                {"row": {"text": f"prompt-{offset}", "timestamp": 1, "url": "u"}},
                {"row": {"text": f"prompt-{offset + 1}", "timestamp": 1, "url": "u"}},
            ],
        }

    return make


def test_downloader_writes_counted_rows_and_filters(monkeypatch, tmp_path, fake_pages):
    monkeypatch.setattr(download_prompts, "fetch_rows_retrying", lambda *a, **k: fake_pages(100))
    out = tmp_path / "prompts"
    rc = download_prompts.main(
        [
            "--count",
            "3",
            "--out",
            str(out),
            "--base-url",
            "https://example.invalid",
            "--delay",
            "0",
            "--start-over",
        ]
    )
    assert rc == 0
    lines = (out / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    # The ``--count 3`` keeps the first three rows; filtering is applied separately.
    assert '"prompt-100"' in lines[0]
