"""Shared pytest fixtures for the watermarks-remover test suite.

The Layer B rewrite is a required step for text cleaning, but it needs a rewrite
backend (WATERMARKS_REWRITE_*) and/or transformers for the `mlm` tactic that the
test environment does not configure. Tests that are not about Layer B (options,
detection, traversal, batch, images) use text cleaning as a vehicle, so we no-op
the Layer B choke point for them; `tests/test_clean_strategy.py` is excluded and
exercises the real apply/reject logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import server


@pytest.fixture(autouse=True)
def _noop_layer_b_for_non_strategy_tests(monkeypatch: pytest.MonkeyPatch, request):
    """Replace `server._apply_layer_b` with a no-op except in test_clean_strategy.

    The Layer B default strategy is loaded by `server.main()`, which the tests do
    not run, so `_DEFAULT_STRATEGY` is None here and a text clean would reject.
    Give tests a valid default and run Layer B as a pass-through.
    """
    if "test_clean_strategy" in request.node.nodeid:
        return
    monkeypatch.setattr(server, "_DEFAULT_STRATEGY", "mlm@0.3")
    monkeypatch.setattr(
        server,
        "_apply_layer_b",
        lambda text, strategy, options: (
            text,
            {"strategy": list(strategy.split(",")), "steps": []},
        ),
    )
