"""Tests for the deterministic humanizer pass behind the humanize tactic."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from humanize_pass import _capitalize_like, humanize_pass


def test_straightens_curly_quotes():
    text = "\u2018single\u2019 and \u201cdouble\u201d quotes, it\u2019s an apostrophe"
    assert humanize_pass(text) == "'single' and \"double\" quotes, it's an apostrophe"


def test_replaces_spaced_em_and_en_dashes():
    assert humanize_pass("a \u2014 b") == "a, b"
    assert humanize_pass("a \u2013 b") == "a, b"


def test_replaces_unspaced_dashes():
    assert humanize_pass("long\u2014according to critics\u2014will") == (
        "long, according to critics, will"
    )
    assert humanize_pass("a--b") == "a, b"


def test_replaces_double_hyphens():
    assert humanize_pass("a -- b") == "a, b"


def test_trailing_dash_before_punctuation():
    assert humanize_pass("word \u2014.") == "word."


def test_collapses_filler_phrases_case_preserving():
    assert humanize_pass("In order to achieve this") == "To achieve this"
    assert humanize_pass("The system has the ability to process") == "The system can process"
    assert humanize_pass("due to the fact that it rained") == "because it rained"


def test_swaps_utilize_family_case_preserving():
    assert humanize_pass("We utilize a tool") == "We use a tool"
    assert humanize_pass("Utilizing fewer tokens helps") == "Using fewer tokens helps"


def test_delve_into_becomes_explore():
    assert humanize_pass("The post delves into the history") == "The post explores the history"


def test_leaves_plain_text_unchanged():
    text = "The cat sat on the mat. Numbers like 1,000 stay intact."
    assert humanize_pass(text) == text


def test_is_idempotent():
    text = "In order to utilize the tool\u2014the report says\u2014it works."
    once = humanize_pass(text)
    assert humanize_pass(once) == once


def test_capitalize_like_preserves_all_uppercase():
    assert _capitalize_like("UTILIZE", "use") == "USE"
    assert _capitalize_like("Utilize", "use") == "Use"
    assert _capitalize_like("utilize", "use") == "use"


def test_swaps_all_uppercase_utilize():
    assert humanize_pass("UTILIZE a tool") == "USE a tool"


def test_preserves_numeric_ranges():
    text = "The decade 2019\u20132020 was busy."
    assert humanize_pass(text) == text


def test_preserves_spaced_numeric_ranges():
    text = "The decade 2019 \u2013 2020 was busy."
    assert humanize_pass(text) == text


def test_preserves_cli_options():
    text = "run the tool --dry-run now"
    assert humanize_pass(text) == text


def test_preserves_delimited_cli_options():
    text = "pass the flag run(--dry-run) to the runner"
    assert humanize_pass(text) == text
