"""Regression tests for empty-sentence burstiness handling (#132).

A body whose sentences yield nothing parseable (e.g. entirely wrapped in a
code fence) must be reported as unmeasurable, not as maximally LLM-like.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from score_stylometry import compute_burstiness, extract_sentences
from score_stylometry import score_text_stylometry as score_text

PLAIN = (
    "We ran the probe for three weeks. Traffic was uneven. "
    "Some days nothing arrived at all, and then a single autonomous system would saturate the link for "
    "forty hours before disappearing without any obvious trigger, which made the early averages useless. "
    "I assumed misconfiguration at first. It wasn't. "
    "The second dataset disagreed with the first, which was inconvenient, because the whole analysis plan "
    "had quietly assumed the two would move together. They do not move together. "
    "One caveat remains. Coverage is thin."
)

FENCED = "```tex\n" + PLAIN + "\n```\n"


def test_sentence_split_does_not_fragment_abbreviations_or_decimals():
    # Regression: the sentence split broke after every ". " (and friends), so
    # "The U.S. is", "2.0 is", and "e.g. on" became standalone fragments.
    # That inflated sentence_count and skewed the burstiness CV the composite
    # score leans on, even though RE_SENTENCE_SPLIT always carried the
    # intended sentence-start guard.
    text = (
        "The U.S. has 50 states. Version 2.0 is out now. "
        "Dr. Smith agrees, e.g. on taxes. It was released."
    )
    sentences = extract_sentences(text)
    assert "The U.S. has 50 states." in sentences
    assert "Version 2.0 is out now." in sentences
    assert "on taxes." not in sentences
    assert "The U.S." not in sentences


def test_burstiness_reports_unmeasurable_cv():
    assert compute_burstiness([])[2] is None
    assert compute_burstiness(["One sentence only."])[2] is None


def test_fenced_body_is_not_scored_as_maximally_llm_like():
    plain = score_text(PLAIN, path="plain.md")
    fenced = score_text(FENCED, path="fenced.md")

    assert plain.sentence_count >= 5
    assert fenced.sentence_count == 0

    # The unmeasurable case surfaces as null CV plus an explicit note…
    assert fenced.burstiness_cv is None
    assert any("burstiness unavailable" in n for n in fenced.notes)

    # …and the composite no longer credits the strongest LLM signal: the
    # 13.6x inflation from the issue collapses to the same neighborhood as
    # the plain file (both are ordinary prose without AI markers).
    assert fenced.score <= plain.score + 0.05


def test_plain_body_still_scores_with_burstiness():
    plain = score_text(PLAIN, path="plain.md")
    assert plain.burstiness_cv is not None and plain.burstiness_cv > 0.25
    assert not any("burstiness unavailable" in n for n in plain.notes)
