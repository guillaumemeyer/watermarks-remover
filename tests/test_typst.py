"""Typst (.typ) support: detection, Layer A, prose-only Layer B segmenter, metadata."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import typst_segments
from format_dispatch import classify, classify_bytes
from text_unicode import clean_text
from typst_segments import PROSE, PROTECTED, compile_check, rewrite_prose, segment, typst_metadata

ZWSP = "​"
ZWJ = "‍"
VS16 = "️"
RLM = "‏"
TAG_A = "\U000e0041"
TAG_B = "\U000e0042"

# --- fixtures (built from escapes so the repo holds no invisible characters) ---
PLAIN_ZW = f"= Title\nHello{ZWSP} wor{ZWSP}ld in prose.\n"
TAGS = f"Tagged{TAG_A}{TAG_B} text here.\n"
ARABIC = f"مرحبا{RLM} بالعالم\n"
EMOJI_ZWJ = f"Family \U0001f468{ZWJ}\U0001f469{ZWJ}\U0001f467 and heart ❤{VS16}.\n"
CODE_INVISIBLE = f'#let s = "a{ZWSP}b"\n#let g() = {{\n  "x{ZWSP}y"\n}}\n'
MATH_RAW = "Math $ sum_(i=0)^n i^2 $ and `raw *text* #not-code` and\n```py\nx = 1 # c\n```\n"
COMMENTS = "// line comment *x*\nProse. /* block /* nested */ still */ more.\n"
MIXED = (
    '#set document(title: "Doc", author: "Ann")\n'
    '#import "lib.typ": tool\n'
    "= Intro <intro>\n"
    f"Hello{ZWSP} world, see @intro and https://example.com/a//b.\n"
    "- item with *bold* and _emph_\n"
    "// comment\n"
    "Math $ x^2 $ and `raw` ok. #strong[kept] and #text(fill: red)[also].\n"
    f'#let f(x) = {{\n  "s{ZWSP} ]"\n}}\n'
    "#if true [yes] else if x {a} else [no]\n"
    "Trailing \\# escaped.\n"
)


def _protected_text(src: str) -> list[str]:
    return [s.text for s in segment(src) if s.kind == PROTECTED]


# --- detection ---------------------------------------------------------------
def test_typ_is_text_not_unknown(tmp_path):
    f = tmp_path / "doc.typ"
    f.write_text("= Hi\n", encoding="utf-8")
    assert classify(f) == "text"
    assert classify_bytes(b"= Hi\n", ".typ") == "text"
    assert classify_bytes(b"= Hi\n", ".TYP") == "text"


def test_other_unknown_extensions_still_refused(tmp_path):
    f = tmp_path / "doc.typx"
    f.write_bytes(b"random bytes \x00\xff")
    assert classify(f) == "unknown"
    assert classify_bytes(b"random \x00\xff", ".typx") == "unknown"


# --- Layer A -----------------------------------------------------------------
def test_layer_a_strips_zero_width_in_prose():
    cleaned, stats = clean_text(PLAIN_ZW)
    assert cleaned == "= Title\nHello world in prose.\n"
    assert stats["removed_count"] == 2


def test_layer_a_strips_tag_characters():
    cleaned, _ = clean_text(TAGS)
    assert cleaned == "Tagged text here.\n"


def test_layer_a_preserves_arabic_rlm():
    cleaned, _ = clean_text(ARABIC)
    assert cleaned == ARABIC


def test_layer_a_preserves_emoji_zwj_and_vs16():
    cleaned, _ = clean_text(EMOJI_ZWJ)
    assert cleaned == EMOJI_ZWJ


def test_layer_a_strips_invisibles_in_code_and_strings():
    # Decision: Layer A is syntax-blind, like every other text format.
    cleaned, _ = clean_text(CODE_INVISIBLE)
    assert cleaned == '#let s = "ab"\n#let g() = {\n  "xy"\n}\n'


def test_literal_nbsp_and_shy_follow_existing_policy():
    # Source-level ``~`` and ``-?`` are ASCII and untouched; literal U+00A0 /
    # U+00AD are normalized/stripped like any other text.
    cleaned, _ = clean_text("a~b -? c d e­f\n")
    assert cleaned == "a~b -? c d ef\n"


@pytest.mark.parametrize("src", [PLAIN_ZW, TAGS, ARABIC, EMOJI_ZWJ, CODE_INVISIBLE, MIXED])
def test_layer_a_idempotent(src):
    once, _ = clean_text(src)
    twice, stats = clean_text(once)
    assert twice == once
    assert stats["removed_count"] == 0


# --- segmenter ---------------------------------------------------------------
@pytest.mark.parametrize(
    "src",
    [PLAIN_ZW, TAGS, ARABIC, EMOJI_ZWJ, CODE_INVISIBLE, MATH_RAW, COMMENTS, MIXED, "", "#", "$ x"],
)
def test_segments_reassemble_exactly(src):
    assert "".join(s.text for s in segment(src)) == src


@pytest.mark.parametrize("src", [MATH_RAW, COMMENTS, CODE_INVISIBLE, MIXED])
def test_stub_rewrite_leaves_protected_regions_identical(src):
    out, _ = rewrite_prose(src, str.upper)
    out_segs = segment(out)
    # Protected text is byte-identical, in order, and prose really was changed.
    assert [s.text for s in out_segs if s.kind == PROTECTED] == _protected_text(src)
    assert out != src or not any(s.kind == PROSE and s.text.strip() for s in segment(src))


def test_math_and_raw_byte_identical():
    out, _ = rewrite_prose(MATH_RAW, str.upper)
    for chunk in ("$ sum_(i=0)^n i^2 $", "`raw *text* #not-code`", "```py\nx = 1 # c\n```"):
        assert chunk in out
    assert "MATH" in out


def test_comments_untouched():
    out, _ = rewrite_prose(COMMENTS, str.upper)
    assert "// line comment *x*" in out
    assert "/* block /* nested */ still */" in out
    assert "PROSE." in out


def test_code_expressions_strings_imports_labels_refs_protected():
    out, _ = rewrite_prose(MIXED, str.upper)
    for kept in (
        '#set document(title: "Doc", author: "Ann")',
        '#import "lib.typ": tool',
        "<intro>",
        "@intro",
        "https://example.com/a//b",
        "#strong[kept]",
        "#text(fill: red)[also]",
        f'#let f(x) = {{\n  "s{ZWSP} ]"\n}}',
        "#if true [yes] else if x {a} else [no]",
        "\\#",
    ):
        assert kept in out, kept
    assert "HELLO" in out and "ITEM WITH" in out


def test_unterminated_constructs_protect_to_eof():
    for src in ("Text $ never closed and more", "Text `raw never closed", "A #f(unclosed prose"):
        out, _ = rewrite_prose(src, str.upper)
        assert out.startswith(src[:4].upper()) or out.startswith("A ")
        tail = src[src.index(next(c for c in "$`#" if c in src)) :]
        assert tail in out


def test_unsafe_rewrite_is_rejected():
    src = "plain words here\n"
    for bad in ("*now bold*", "two\nlines", "#evil()", "= heading"):
        out, st = rewrite_prose(src, lambda _s, b=bad: b)
        assert out == src
        assert st["prose_rejected"] == 1


def test_never_calls_rewrite_on_syntax():
    seen: list[str] = []
    rewrite_prose(MIXED, lambda s: (seen.append(s), s)[1])
    for chunk in seen:
        assert "\n" not in chunk
        assert not any(c in chunk for c in "#$`")


# --- metadata ----------------------------------------------------------------
def test_metadata_reported_not_removed():
    meta = typst_metadata('#set document(title: "T", author: ("A", "B"), date: auto)\nx\n')
    assert meta["present"] and meta["fields"]["title"] == "T"
    assert meta["fields"]["author"] == ["A", "B"]


def test_metadata_ignored_in_comments_and_raw():
    assert not typst_metadata('// #set document(title: "x")\n`#set document(title: "y")`\n')[
        "present"
    ]


def test_cli_typst_meta_and_clean(tmp_path):
    f = tmp_path / "d.typ"
    f.write_text('#set document(title: "T")\nHi' + ZWSP + " there\n", encoding="utf-8")
    out = subprocess.run(
        [sys.executable, str(SCRIPTS / "inspect_file.py"), str(f), "--typst-meta", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    data = json.loads(out.stdout)
    assert data["kind"] == "text" and data["typst_metadata"]["fields"]["title"] == "T"
    dest = tmp_path / "o.typ"
    subprocess.run(
        [sys.executable, str(SCRIPTS / "clean_file.py"), str(f), "-o", str(dest)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert dest.read_text(encoding="utf-8") == '#set document(title: "T")\nHi there\n'
    # metadata is kept by default
    assert "#set document" in dest.read_text(encoding="utf-8")


# --- compile check -----------------------------------------------------------
def test_compile_check_skipped_without_typst(monkeypatch, tmp_path):
    monkeypatch.setattr(typst_segments.shutil, "which", lambda _n: None)
    res = compile_check("a", "b", tmp_path)
    assert res["status"] == "skipped"


def test_compile_check_regression_and_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(typst_segments.shutil, "which", lambda _n: "/usr/bin/typst")
    monkeypatch.setattr(typst_segments, "_compiles", lambda src, _d: (src == "good", "err"))
    assert compile_check("good", "good", tmp_path)["status"] == "unchanged"
    assert compile_check("good", "bad", tmp_path)["status"] == "regression"
    assert compile_check("bad", "good", tmp_path)["status"] == "ok"
    assert compile_check("bad", "bad2", tmp_path)["status"] == "original_does_not_compile"


# --- HTTP service ------------------------------------------------------------
def test_server_clean_typ_rewrites_prose_only(monkeypatch):
    import server

    monkeypatch.setattr(server, "_DEFAULT_STRATEGY", "stub")
    monkeypatch.setattr(server, "_apply_layer_b", lambda text, _s, _o: (text.upper(), {"n": 1}))
    res = server._clean_payload(MIXED.encode(), "doc.typ", {})
    out = base64.b64decode(res["cleaned"]).decode()
    assert "HELLO WORLD" in out and ZWSP not in out
    assert '#set document(title: "Doc", author: "Ann")' in out
    assert res["report"]["typst_compile"]["status"] in {
        "skipped",
        "ok",
        "unchanged",
        "original_does_not_compile",
    }


def test_server_inspect_typ_metadata():
    import server

    data = b'#set document(title: "T")\nhi\n'
    plain = server._inspect_payload(data, "d.typ", False)
    assert "typst_metadata" not in plain["report"]
    with_meta = server._inspect_payload(data, "d.typ", False, True)
    assert with_meta["report"]["typst_metadata"]["fields"]["title"] == "T"
