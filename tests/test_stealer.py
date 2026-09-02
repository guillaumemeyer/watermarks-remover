"""Tests for the black-box watermark-stealing module (scorer core + downloader)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STEALER = ROOT / "stealer"
sys.path.insert(0, str(STEALER))

import download_prompts
import scorer
import steal
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
    """The default tokenizer keeps words, punctuation, and number segments."""

    toks = tokens.default_tokenize("Hello, World! 3.5")
    assert toks == ["hello", ",", "world", "!", "3", ".", "5"]


def test_count_ngrams_counts_next_tokens():
    """(context, next-token) counts respect token order and boundaries."""

    wm, totals, unis = tokens.count_ngrams(["a b a b"], 1, tokens.default_tokenize)
    # sequence: tokens = [a, b, a, b]; pairs (a->b), (b->a), (a->b)
    assert wm[("a",)]["b"] == 2
    assert wm[("b",)]["a"] == 1
    assert totals[("a",)] == 2
    assert unis["a"] == 2 and unis["b"] == 2


def test_context_key_is_collision_free():
    """Whitespace-containing tokens must not collapse into one context key."""

    assert scorer.context_key(("a b", "c")) != scorer.context_key(("a", "b c"))


def test_build_scorer_ranks_boosted_tokens_above_others():
    """Tokenisms boosted in the watermarked corpus score higher than baseline tokens."""

    wm = tokens.count_ngrams(WM_TEXTS, 3)
    base = tokens.count_ngrams(BASE_TEXTS, 3)
    s = scorer.build_scorer(wm, base, context_len=3, topk=20)
    # "fox" appears in the watermarked corpus, not the baseline, so its score for
    # the context "the quick brown" should be positive and beat an unrelated token.
    entry = s["scorer"].get(scorer.context_key(("the", "quick", "brown")), [])
    scores = {item["token"]: item["score"] for item in entry}
    assert "fox" in scores
    assert scores["fox"] > 0


def test_apply_delta_demotes_green_token():
    """Positive delta lowers the logit of watermarked-boosted (green) tokens."""

    wm = tokens.count_ngrams(WM_TEXTS, 3)
    base = tokens.count_ngrams(BASE_TEXTS, 3)
    s = scorer.build_scorer(wm, base, context_len=3, topk=20)
    logits = {"fox": 0.0, "cat": 0.0}
    adjusted = scorer.apply_delta(s, ("the", "quick", "brown"), logits, delta=1.0)
    assert adjusted["fox"] < 0.0  # green token demoted
    assert adjusted["cat"] == 0.0  # unknown token untouched


def test_score_sequence_applies_lookups():
    """score_sequence hits the scorer table for at least one (context, token) pair."""

    wm = tokens.count_ngrams(WM_TEXTS, 3)
    base = tokens.count_ngrams(BASE_TEXTS, 3)
    s = scorer.build_scorer(wm, base, context_len=3, topk=20)
    result = scorer.score_sequence(s, tokens.default_tokenize("the quick brown fox jumps"), 3)
    assert result["applied"] > 0


def test_detect_ctx_override_is_honored(tmp_path, capsys):
    """A nonzero detect --ctx overrides the stored context length."""

    wm = tokens.count_ngrams(WM_TEXTS, 2)
    base = tokens.count_ngrams(BASE_TEXTS, 2)
    s = scorer.build_scorer(wm, base, context_len=2, topk=20)
    path = tmp_path / "s.json"
    path.write_text(json.dumps(s), encoding="utf-8")

    def run(ctx):
        args = argparse.Namespace(s_star=str(path), text="the quick brown fox", file=None, ctx=ctx)
        assert steal.cmd_detect(args) == 0
        return json.loads(capsys.readouterr().out.strip())

    stored = run(0)  # stored context_len == 2
    overridden = run(4)  # explicit --ctx 4 must win
    # A different context length changes how many lookups are applied.
    assert stored["applied"] != overridden["applied"]


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
    """The downloader writes exactly --count rows to prompts.jsonl."""

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
    assert '"prompt-100"' in lines[0]


def test_downloader_resume_after_interrupt_does_not_duplicate(monkeypatch, tmp_path):
    """An interrupt must leave a consistent cursor + file so resume adds rows, not dups."""

    def page(*a, **k):  # offset is the 5th positional arg to fetch_rows_retrying
        offset = a[4]
        return {
            "num_rows_per_page": 2,
            "rows": [
                {"row": {"text": f"r{offset}"}},
                {"row": {"text": f"r{offset + 1}"}},
            ],
        }

    monkeypatch.setattr(download_prompts, "fetch_rows_retrying", page)
    # The first data page commits, then the follow-up sleep is interrupted.
    monkeypatch.setattr(
        download_prompts.time,
        "sleep",
        lambda _s: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    out = tmp_path / "prompts"
    rc = download_prompts.main(
        [
            "--count",
            "1000",
            "--out",
            str(out),
            "--base-url",
            "https://example.invalid",
            "--delay",
            "0",
            "--start-over",
        ]
    )
    assert rc == 130
    lines = (out / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["text"] for line in lines] == ["r0", "r1"]
    state = json.loads((out / ".download-state.json").read_text(encoding="utf-8"))
    assert state == {"next_offset": 2, "written": 2}

    # Resume continues from offset 2 and appends r2/r3 — no "r0"/"r1" duplicates.
    monkeypatch.setattr(download_prompts.time, "sleep", lambda _s: None)
    rc = download_prompts.main(
        [
            "--count",
            "4",
            "--out",
            str(out),
            "--base-url",
            "https://example.invalid",
            "--delay",
            "0",
        ]
    )
    assert rc == 0
    final = (out / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["text"] for line in final] == ["r0", "r1", "r2", "r3"]


def test_downloader_start_over_clears_stale_state_before_probe(monkeypatch, tmp_path):
    """A failed --start-over probe must not leave a stale cursor for the next run."""

    out = tmp_path / "prompts"
    out.mkdir()
    (out / "prompts.jsonl").write_text('{"text":"old"}\n', encoding="utf-8")
    (out / ".download-state.json").write_text(
        json.dumps({"next_offset": 900, "written": 900}), encoding="utf-8"
    )

    def fail_probe(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(download_prompts, "fetch_rows_retrying", fail_probe)
    with pytest.raises(RuntimeError, match="offline"):
        download_prompts.main(
            [
                "--count",
                "1",
                "--out",
                str(out),
                "--base-url",
                "https://example.invalid",
                "--delay",
                "0",
                "--start-over",
            ]
        )

    assert not (out / "prompts.jsonl").exists()
    assert not (out / ".download-state.json").exists()

    calls = []

    def page(*args, **kwargs):
        calls.append(args[4])
        return {"num_rows_per_page": 1, "rows": [{"row": {"text": "fresh"}}]}

    monkeypatch.setattr(download_prompts, "fetch_rows_retrying", page)
    assert (
        download_prompts.main(
            [
                "--count",
                "1",
                "--out",
                str(out),
                "--base-url",
                "https://example.invalid",
                "--delay",
                "0",
            ]
        )
        == 0
    )
    assert calls == [0, 0]
    assert json.loads((out / "prompts.jsonl").read_text(encoding="utf-8")) == {"text": "fresh"}
