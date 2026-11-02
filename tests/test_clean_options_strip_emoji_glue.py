"""The HTTP API must be able to strip load-bearing invisibles, like the CLI can.

``clean_text.py`` has ``--strip-emoji-glue``; ``ALLOWED_CLEAN_OPTIONS`` had no
equivalent, so every caller going through ``/clean`` kept emoji glue, script
joiners and flag tags with no way to ask for the paranoid pass. The agent skill
is one of those callers: a manuscript whose emoji or Indic text carries a
carrier it wants gone could not reach the option the CLI already had.

Default stays False: existing callers see no change, because the option only
removes what the default deliberately preserves.
"""

from __future__ import annotations

import base64
import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import server

ZWJ = "\u200d"
ZWSP = "\u200b"
VS16 = "\ufe0f"
FAMILY = "\U0001f468\u200d\U0001f469\u200d\U0001f467"
HEART_ON_FIRE = "\u2764\ufe0f\u200d\U0001f525"

# Glue in a load-bearing position: ZWJ between emoji bases, VS16 after an emoji
# base. Both survive the default pass on purpose, so they are what the option is
# for.
EMOJI = f"Family {FAMILY} and fire {HEART_ON_FIRE}.\n"

# The same characters between plain ASCII are carriers, and are stripped with
# or without the option.
CARRIERS = f"Intro{ZWSP} and a{ZWJ}b.\n"


def _clean(options, name="chapter.txt"):
    data = EMOJI.encode("utf-8")
    result = server._clean_payload(data, name, options)
    return base64.b64decode(result["cleaned"]).decode("utf-8")


def _clean_carriers(options, name="chapter.txt"):
    data = CARRIERS.encode("utf-8")
    result = server._clean_payload(data, name, options)
    return base64.b64decode(result["cleaned"]).decode("utf-8")


def test_option_is_allowed():
    assert server.ALLOWED_CLEAN_OPTIONS.get("strip_emoji_glue") is bool


def test_default_keeps_load_bearing_invisibles():
    out = _clean({})
    assert out == EMOJI
    assert out.count(ZWJ) == EMOJI.count(ZWJ)
    assert VS16 in out


def test_opting_in_strips_them():
    out = _clean({"strip_emoji_glue": True})
    assert ZWJ not in out
    assert VS16 not in out
    assert out == EMOJI.replace(ZWJ, "").replace(VS16, "")


def test_carriers_go_either_way():
    """The paranoid pass is a superset: it never keeps more than the default."""
    for options in ({}, {"strip_emoji_glue": True}):
        out = _clean_carriers(options)
        assert ZWSP not in out and ZWJ not in out, options


def test_markdown_payload_honours_the_option():
    """The container path is what a manuscript actually takes."""
    kept = _clean({}, name="chapter.md")
    assert ZWJ in kept and VS16 in kept
    stripped = _clean({"strip_emoji_glue": True}, name="chapter.md")
    assert ZWJ not in stripped and VS16 not in stripped


def test_html_payload_honours_the_option():
    out = _clean({"strip_emoji_glue": True}, name="page.html")
    assert ZWJ not in out and VS16 not in out


def test_ooxml_and_odt_runs_honour_the_option():
    """The keyword reaches the XML text-run scrubbers, not only the body ones."""
    import container_meta

    docx_run = f"<w:t>Family {FAMILY}</w:t>"
    kept, removed, _ = container_meta._scrub_docx_text(docx_run, strip_emoji_glue=False)
    assert ZWJ in kept and removed == 0
    stripped, removed, _ = container_meta._scrub_docx_text(docx_run, strip_emoji_glue=True)
    assert ZWJ not in stripped and removed == 2

    odt_run = f"<text:p>Family {FAMILY}</text:p>"
    kept_odt, _, _ = container_meta._scrub_odt_text(odt_run)
    assert ZWJ in kept_odt
    stripped_odt, _, _ = container_meta._scrub_odt_text(odt_run, strip_emoji_glue=True)
    assert ZWJ not in stripped_odt


def test_xlsx_and_pptx_runs_honour_the_option():
    """XLSX and PPTX go through the same run scrubber under different tags."""
    import container_meta

    cases = [
        (container_meta._scrub_xlsx_text, f"<t>Family {FAMILY}</t>"),
        (container_meta._scrub_pptx_text, f"<a:t>Family {FAMILY}</a:t>"),
    ]
    for scrub, run in cases:
        kept, _, _ = scrub(run)
        assert ZWJ in kept
        stripped, _, _ = scrub(run, strip_emoji_glue=True)
        assert ZWJ not in stripped


def _minimal_epub() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr(
            "OEBPS/chapter.xhtml",
            f"<html><body><p>Family {FAMILY}</p></body></html>",
        )
    return buf.getvalue()


def test_epub_body_honours_the_option():
    import container_meta

    data = _minimal_epub()
    kept, _ = container_meta.clean_epub(data, strip_emoji_glue=False)
    with zipfile.ZipFile(io.BytesIO(kept)) as zf:
        assert ZWJ in zf.read("OEBPS/chapter.xhtml").decode("utf-8")

    stripped, _ = container_meta.clean_epub(data, strip_emoji_glue=True)
    with zipfile.ZipFile(io.BytesIO(stripped)) as zf:
        assert ZWJ not in zf.read("OEBPS/chapter.xhtml").decode("utf-8")
