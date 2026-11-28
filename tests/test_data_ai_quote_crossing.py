"""Regression tests for quote-crossing `data-ai*` attribute values (#415).

``[^"']*`` stopped a value at the first quote of *either* kind, so
``data-ai-summary="Bob's review"`` matched only as far as ``Bob'`` and
``clean_html`` deleted that truncated span - leaving ``s review"`` behind as
stray markup inside the tag. The ``<meta content>`` scan was fixed the same way
in #373 / #401; the ``data-ai*`` branch kept the flat character class.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from container_meta import clean_html, inspect_html

# A double-quoted value holding an apostrophe, and its mirror image.
APOSTROPHE = """<div class="wrap" data-ai-summary="Bob's review of the page">text</div>"""
DOUBLE_QUOTE = """<div class="wrap" data-ai-note='say "hi" now'>text</div>"""


def test_double_quoted_value_with_apostrophe_is_dropped_whole():
    out, actions = clean_html(APOSTROPHE)
    assert out == '<div class="wrap">text</div>'
    assert actions == ["drop data-ai* attributes x1"]


def test_single_quoted_value_with_double_quote_is_dropped_whole():
    out, actions = clean_html(DOUBLE_QUOTE)
    assert out == '<div class="wrap">text</div>'
    assert actions == ["drop data-ai* attributes x1"]


def test_recorded_sample_is_not_truncated():
    _, has_ai, findings, _ = inspect_html(APOSTROPHE)
    assert has_ai is True
    assert findings == ["""attr: data-ai-summary="Bob's review of the page\""""]


def test_plain_attribute_still_dropped():
    out, actions = clean_html('<div class="wrap" data-ai-generated="yes">text</div>')
    assert out == '<div class="wrap">text</div>'
    assert actions == ["drop data-ai* attributes x1"]


def test_neighbouring_attributes_survive():
    out, actions = clean_html('<div data-ai="x" data-ai-model="gpt" class="z">t</div>')
    assert out == '<div class="z">t</div>'
    assert actions == ["drop data-ai* attributes x2"]


def test_lookalike_attributes_are_untouched():
    # Regression guard for #395: `data-aid` / `data-airline` are not provenance.
    html = '<div data-aid="1" data-airline="x">text</div>'
    out, actions = clean_html(html)
    assert out == html
    assert actions == ["no HTML AI meta removed"]
    _, has_ai, findings, _ = inspect_html(html)
    assert has_ai is False
    assert findings == []
