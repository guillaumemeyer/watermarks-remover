"""Regression tests: U+200B handling must not damage word boundaries.

The cleaner may remove a zero-width space, but it must never insert a space
that was not there, never merge two real words separated by an ordinary space,
and must keep U+200B where it is the orthographic word separator.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import pytest
from text_unicode import clean_text, describe_zwsp_joins, inspect_text

ZWSP = "\u200b"


# --- Ordinary prose ---------------------------------------------------------


def test_zwsp_next_to_real_space_keeps_one_space():
    cleaned, _ = clean_text(f"Hello {ZWSP}world and {ZWSP} again")
    assert cleaned == "Hello world and  again"


def test_zwsp_as_only_separator_is_removed_and_reported():
    # Zero width: the original already rendered as "Helloworld", so removal
    # does not change what a reader saw. It is reported for review instead of
    # guessed at.
    text = f"Line one.\nHello{ZWSP}world\n"
    cleaned, stats = clean_text(text)
    assert cleaned == "Line one.\nHelloworld\n"
    assert stats["zwsp_word_joins"] == [{"offset": text.index(ZWSP), "line": 2}]


def test_zwsp_next_to_punctuation_is_not_reported_as_join():
    _, stats = clean_text(f"End.{ZWSP} Next{ZWSP}, then")
    assert stats["zwsp_word_joins"] == []


def test_clean_prose_is_byte_identical():
    text = "Plain prose, with punctuation; nothing hidden.\n"
    cleaned, stats = clean_text(text)
    assert cleaned == text
    assert stats["removed_count"] == 0
    assert stats["zwsp_word_joins"] == []


# --- Compound words ---------------------------------------------------------


def test_compound_break_hints_never_gain_spaces():
    cleaned, stats = clean_text(f"Donau{ZWSP}dampf{ZWSP}schiff and well{ZWSP}-known")
    assert cleaned == "Donaudampfschiff and well-known"
    assert " " not in cleaned.split(" and ")[0]
    assert len(stats["zwsp_word_joins"]) == 2  # letter|letter only, not before "-"


def test_zwsp_inside_url_break_hint_is_removed_without_spaces():
    cleaned, _ = clean_text(f"https://example.com/{ZWSP}very/{ZWSP}long/path")
    assert cleaned == "https://example.com/very/long/path"


# --- Intentional Unicode formatting -----------------------------------------


def test_thai_word_separator_is_preserved():
    thai = f"\u0e20\u0e32\u0e29\u0e32{ZWSP}\u0e44\u0e17\u0e22"
    cleaned, stats = clean_text(thai)
    assert cleaned == thai
    assert stats["removed_count"] == 0
    assert inspect_text(thai).suspicious_total == 0


def test_japanese_line_break_hint_is_preserved():
    text = f"\u6771\u4eac{ZWSP}\u30bf\u30ef\u30fc"
    cleaned, _ = clean_text(text)
    assert cleaned == text


def test_zwsp_next_to_cjk_iteration_marks_is_preserved():
    # U+3005 and U+303B are letters (Lm) outside the kana block: 日々, 時〻.
    for text in (
        f"\u65e5\u3005{ZWSP}\u65e5\u672c",
        f"\u65e5\u672c{ZWSP}\u3005",
        f"\u6642\u303b{ZWSP}\u30ab",
    ):
        assert clean_text(text)[0] == text


def test_zwsp_next_to_cjk_brackets_and_numerals_is_still_stripped():
    # U+3008 bracket (Ps) and U+3007 ideographic zero (Nl) share the range.
    for neighbour in ("\u3008", "\u3007"):
        text = f"\u65e5{ZWSP}{neighbour}"
        assert clean_text(text)[0] == text.replace(ZWSP, "")


def test_zwsp_next_to_kana_punctuation_is_still_stripped():
    # U+30FB KATAKANA MIDDLE DOT and U+30A0 double hyphen sit in the kana block
    # but are punctuation, so a ZWSP beside them is not a word break.
    for punct in ("\u30fb", "\u30a0"):
        text = f"\u30ab{ZWSP}{punct}\u30ab"
        cleaned, _ = clean_text(text)
        assert cleaned == text.replace(ZWSP, "")
        assert inspect_text(text).suspicious_total == 1


def test_zwsp_before_kana_length_mark_is_preserved():
    # U+30FC PROLONGED SOUND MARK is a letter (Lm), so the break is kept.
    text = f"\u30ab{ZWSP}\u30fc"
    assert clean_text(text)[0] == text


def test_zwsp_between_different_scripts_is_still_stripped():
    cleaned, _ = clean_text(f"\u0e44\u0e17\u0e22{ZWSP}Latin")
    assert cleaned == "\u0e44\u0e17\u0e22Latin"


def test_leading_or_trailing_zwsp_in_thai_is_still_stripped():
    cleaned, _ = clean_text(f"{ZWSP}\u0e44\u0e17\u0e22{ZWSP}")
    assert cleaned == "\u0e44\u0e17\u0e22"


def test_existing_load_bearing_invisibles_unchanged():
    persian = "\u0645\u06cc\u200c\u0631\u0648\u0645"
    family = "\U0001f468\u200d\U0001f469\u200d\U0001f467"
    cleaned, stats = clean_text(f"{persian} {family}")
    assert cleaned == f"{persian} {family}"
    assert stats["removed_count"] == 0


def test_strip_emoji_glue_flag_also_strips_thai_zwsp():
    # The explicit "strip load-bearing invisibles" switch keeps its meaning.
    cleaned, _ = clean_text(f"\u0e44\u0e17\u0e22{ZWSP}\u0e44\u0e17\u0e22", strip_emoji_glue=True)
    assert ZWSP not in cleaned


# --- Source code ------------------------------------------------------------


def test_source_code_gains_no_spaces():
    code = (
        "def greet(name):\n"
        f'    message = "Hello{ZWSP}world"\n'
        f"    user{ZWSP}_id = 42\n"
        "    return message\n"
    )
    cleaned, stats = clean_text(code)
    assert cleaned == code.replace(ZWSP, "")
    assert cleaned.count(" ") == code.count(" ")
    compile(cleaned, "<test>", "exec")
    assert [join["line"] for join in stats["zwsp_word_joins"]] == [2]


def test_source_code_indentation_and_tabs_untouched():
    code = f"if x:\n\tif y:{ZWSP}\n\t\treturn 1\n"
    cleaned, _ = clean_text(code)
    assert cleaned == "if x:\n\tif y:\n\t\treturn 1\n"


# --- Joined-word warning ----------------------------------------------------


def test_join_warning_is_none_without_joins():
    _, stats = clean_text("Nothing hidden here.\n")
    assert describe_zwsp_joins(stats) is None


def test_join_warning_names_each_line_once_in_order():
    text = f"a{ZWSP}b\n\nc{ZWSP}d e{ZWSP}f g{ZWSP}h\nplain\ni{ZWSP}j\n"
    _, stats = clean_text(text)
    assert [join["line"] for join in stats["zwsp_word_joins"]] == [1, 3, 3, 3, 5]
    assert describe_zwsp_joins(stats) == (
        "5 zero-width space(s) between letters or digits removed; check for joined words or numbers on line(s) 1, 3, 5"
    )


def test_join_warning_names_digit_joins_accurately():
    _, stats = clean_text(f"Call 555{ZWSP}0199 now\n")
    assert describe_zwsp_joins(stats) == (
        "1 zero-width space(s) between letters or digits removed; "
        "check for joined words or numbers on line(s) 1"
    )


def test_join_warning_sorts_lines_given_out_of_order():
    stats = {"zwsp_word_joins": [{"offset": 9, "line": 8}, {"offset": 1, "line": 2}]}
    assert describe_zwsp_joins(stats).endswith("on line(s) 2, 8")


def test_join_lines_are_exact_across_many_lines_and_blank_runs():
    text = "".join(("\n" * (n % 3)) + f"x{ZWSP}y\n" for n in range(200))
    _, stats = clean_text(text)
    expected = [
        {"offset": offset, "line": text.count("\n", 0, offset) + 1}
        for offset, ch in enumerate(text)
        if ch == ZWSP
    ]
    assert stats["zwsp_word_joins"] == expected


def test_join_reporting_scales_linearly():
    text = f"w{ZWSP}w\n" * 40_000
    start = time.perf_counter()
    _, stats = clean_text(text)
    assert len(stats["zwsp_word_joins"]) == 40_000
    assert stats["zwsp_word_joins"][-1]["line"] == 40_000
    assert time.perf_counter() - start < 2.0


def test_join_warning_truncates_after_five_distinct_lines():
    text = "".join(f"w{ZWSP}w w{ZWSP}w\n" for _ in range(7))
    _, stats = clean_text(text)
    assert describe_zwsp_joins(stats) == (
        "14 zero-width space(s) between letters or digits removed; "
        "check for joined words or numbers on line(s) 1, 2, 3, 4, 5 (+2 more)"
    )


# --- PostToolUse hook, end to end -------------------------------------------

HOOK = SCRIPTS / "hook_written_file.py"


def _run_clean_hook(path: Path) -> dict:
    env = os.environ.copy()
    env.pop("CLAUDE_PLUGIN_OPTION_HOOK_MODE", None)
    env["WATERMARKS_HOOK_MODE"] = "clean"
    payload = {
        "tool_name": "Write",
        "tool_input": {"file_path": str(path)},
        "cwd": str(path.parent),
    }
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout) if proc.stdout.strip() else {}


@pytest.mark.parametrize("name", ["note.md", "note.txt", "page.html", "paper.tex"])
def test_hook_reports_joined_words_for_every_text_route(tmp_path, name):
    target = tmp_path / name
    target.write_text(f"Title\n\nsecond{ZWSP}line and a real {ZWSP}space\n", encoding="utf-8")
    message = _run_clean_hook(target)["systemMessage"]
    assert target.read_text(encoding="utf-8") == "Title\n\nsecondline and a real space\n"
    assert "1 zero-width space(s) between letters or digits removed" in message
    assert "line(s) 3" in message


def test_hook_lists_a_line_with_several_joins_once(tmp_path):
    target = tmp_path / "note.md"
    target.write_text(f"Title\n\nDonau{ZWSP}dampf{ZWSP}schiff\nfoo{ZWSP}bar\n", encoding="utf-8")
    message = _run_clean_hook(target)["systemMessage"]
    assert target.read_text(encoding="utf-8") == "Title\n\nDonaudampfschiff\nfoobar\n"
    assert "3 zero-width space(s) between letters or digits removed" in message
    assert "check for joined words or numbers on line(s) 3, 4)" in message


def test_hook_leaves_thai_markdown_untouched(tmp_path):
    target = tmp_path / "thai.md"
    original = f"ภาษา{ZWSP}ไทย\n"
    target.write_text(original, encoding="utf-8")
    assert _run_clean_hook(target) == {}
    assert target.read_text(encoding="utf-8") == original
