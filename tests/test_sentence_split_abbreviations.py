"""Regression tests for abbreviation-safe sentence splitting (#417).

``extract_sentences`` re-split on a bare ``(?<=[.!?])\\s+`` while the guarded
``RE_SENTENCE_SPLIT`` constant sat unused, so "U.S.", "2.0" and "e.g." were cut
in half. The fragments are what ``compute_burstiness`` measures, and burstiness
carries the heaviest weight in the composite score (0.45): prose that merely
mentions "U.S." had its cadence measured over one- and two-word fragments, and
``sentence_count`` was inflated at the same time.

"Dr." still splits: a guard cannot know which dotted token is an abbreviation
without an abbreviation table, which is deliberately out of scope here.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from score_stylometry import extract_sentences, score_text_stylometry

# The exact sample from the report: seven fragments where the prose has four
# sentences, because "U.S.", "2.0" and "e.g." each terminated a chunk.
ABBREVIATED = (
    "The U.S. has 50 states. Version 2.0 is out now. "
    "Dr. Smith agrees, e.g. on taxes. It was released."
)


def test_abbreviations_and_decimals_stay_in_their_sentence():
    assert extract_sentences(ABBREVIATED) == [
        "The U.S. has 50 states.",
        "Version 2.0 is out now.",
        "Dr.",
        "Smith agrees, e.g. on taxes.",
        "It was released.",
    ]


def test_truncated_abbreviation_fragments_are_gone():
    sentences = extract_sentences(ABBREVIATED)
    # Each of these used to be its own "sentence", and each has 1-2 words.
    assert "The U.S." not in sentences
    assert "Version 2.0 is out now." in sentences
    assert "Smith agrees, e.g." not in sentences
    assert "on taxes." not in sentences


def test_terminator_followed_by_lowercase_is_not_a_boundary():
    assert extract_sentences("The value is 2.5 on a 10.0 scale.") == [
        "The value is 2.5 on a 10.0 scale."
    ]


def test_paragraph_break_splits_even_when_next_line_starts_lowercase():
    assert extract_sentences("First paragraph ends here.\n\nand a new one starts.") == [
        "First paragraph ends here.",
        "and a new one starts.",
    ]


def test_code_fences_and_blank_lines_are_still_dropped():
    text = "Before.\n\n```python\nprint('x')\n```\n\nAfter."
    assert extract_sentences(text) == ["Before.", "After."]


def test_report_sentence_count_matches_the_split():
    report = score_text_stylometry(ABBREVIATED)
    assert len(extract_sentences(ABBREVIATED)) == 5
    assert report.sentence_count == 5
