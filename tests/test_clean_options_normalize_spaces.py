"""The HTTP API must be able to keep exotic spaces, like the CLI already can.

``clean_text.py`` has ``--no-normalize-spaces``; ``ALLOWED_CLEAN_OPTIONS`` had
no equivalent, so every caller going through ``/clean`` rewrote U+00A0 and
U+202F to ordinary spaces with no way to decline. The agent skill is one of
those callers, which put French, and any language whose typography relies on
a non-breaking space, out of its reach.

Default stays True: existing callers see no change.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import server

NBSP = " "
NNBSP = " "
ZWSP = "​"

FRENCH = f"Le doute{NBSP}: il reste entier.{NNBSP}Et un vrai porteur{ZWSP} ici.\n"


def _clean(options):
    data = FRENCH.encode("utf-8")
    result = server._clean_payload(data, "chapter.txt", options)
    return base64.b64decode(result["cleaned"]).decode("utf-8")


def test_option_is_allowed():
    assert server.ALLOWED_CLEAN_OPTIONS.get("normalize_spaces") is bool


def test_default_still_normalizes():
    out = _clean({})
    assert NBSP not in out
    assert NNBSP not in out
    assert ZWSP not in out


def test_opting_out_keeps_the_spaces_and_still_strips_carriers():
    out = _clean({"normalize_spaces": False})
    assert out.count(NBSP) == FRENCH.count(NBSP)
    assert out.count(NNBSP) == FRENCH.count(NNBSP)
    assert ZWSP not in out
    assert out == FRENCH.replace(ZWSP, "")
