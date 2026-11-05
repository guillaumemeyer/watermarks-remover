"""Regression tests for Layer B rewrite candidate quality.

Add to tests/test_rewrite_candidate_quality.py in watermarks-remover.
Run: python -m pytest tests/test_rewrite_candidate_quality.py -q
"""
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / 'service' / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from rewrite_text import _select_candidate


def test_empty_candidate_cannot_win_by_lexical_divergence():
    original = 'This paragraph discusses three important research questions.'
    valid = 'This passage explores three important research questions.'
    selected, _ = _select_candidate(original, ['', valid])
    assert selected == valid


def test_extreme_truncation_does_not_beat_substantive_candidate():
    original = 'The agreement establishes a review committee and requires annual reports.'
    valid = 'The agreement creates an oversight committee and mandates yearly reports.'
    selected, _ = _select_candidate(original, ['Unrelated.', valid])
    assert selected == valid


def test_all_invalid_candidates_fall_back_to_original():
    original = 'A complete sentence with substantive content.'
    selected, _ = _select_candidate(original, ['', '  '])
    assert selected == original
