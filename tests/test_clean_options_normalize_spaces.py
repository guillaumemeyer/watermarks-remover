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


def _clean(options, name="chapter.txt"):
    data = FRENCH.encode("utf-8")
    result = server._clean_payload(data, name, options)
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


def test_markdown_payload_honours_the_option():
    """The container path is what a manuscript actually takes."""
    out = _clean({"normalize_spaces": False}, name="chapter.md")
    assert out.count(NBSP) == FRENCH.count(NBSP)
    assert out.count(NNBSP) == FRENCH.count(NNBSP)
    assert ZWSP not in out


def test_markdown_payload_still_normalizes_by_default():
    out = _clean({}, name="chapter.md")
    assert NBSP not in out
    assert NNBSP not in out


def test_html_payload_honours_the_option():
    out = _clean({"normalize_spaces": False}, name="page.html")
    assert out.count(NBSP) == FRENCH.count(NBSP)
    assert ZWSP not in out


def test_ooxml_and_odt_runs_honour_the_option():
    """The keyword reaches the XML text-run scrubbers, not only the body ones."""
    import container_meta

    docx_run = f"<w:t>Le doute{NBSP}: entier{ZWSP}</w:t>"
    kept, _, _ = container_meta._scrub_docx_text(docx_run, normalize_spaces=False)
    assert NBSP in kept and ZWSP not in kept
    flattened, _, _ = container_meta._scrub_docx_text(docx_run)
    assert NBSP not in flattened

    odt_run = f"<text:p>Le doute{NBSP}: entier{ZWSP}</text:p>"
    kept_odt, _, _ = container_meta._scrub_odt_text(odt_run, normalize_spaces=False)
    assert NBSP in kept_odt and ZWSP not in kept_odt
