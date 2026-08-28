"""Tests for Layer A text Unicode scrub."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from text_unicode import clean_text, inspect_text


def test_strips_zero_width_and_soft_hyphen():
    raw = "Hello\u200bWorld\u00ad!"
    cleaned, stats = clean_text(raw)
    assert cleaned == "HelloWorld!"
    assert stats["removed_count"] >= 2


def test_normalizes_exotic_spaces():
    raw = "a\u2003b\u3000c"  # em space, ideographic space
    cleaned, stats = clean_text(raw)
    assert cleaned == "a b c"
    assert stats["replaced_count"] >= 2


def test_inspect_finds_zwsp():
    report = inspect_text("x\u200by")
    assert report.suspicious_total >= 1
    kinds = {h.kind for h in report.hits}
    assert "zwj_family" in kinds or "strip" in kinds


def test_inspect_tag_chars():
    # Language tag character U+E0041 (TAG LATIN CAPITAL LETTER A)
    raw = "hi" + chr(0xE0041) + "there"
    report = inspect_text(raw)
    assert report.suspicious_total >= 1
    assert any(h.kind == "tag_chars" for h in report.hits)
    cleaned, stats = clean_text(raw)
    assert chr(0xE0041) not in cleaned
    assert stats["removed_count"] >= 1


def test_inspect_bidi():
    raw = "ab\u202eef"  # RLO
    report = inspect_text(raw)
    assert any(h.kind == "bidi" for h in report.hits)
    cleaned, _ = clean_text(raw)
    assert "\u202e" not in cleaned


def test_preserves_legitimate_bidi_marks_and_isolates_by_default():
    raw = "السعر \u2066123 USD\u2069\u200f"
    report = inspect_text(raw)
    assert any(h.kind == "bidi" for h in report.hits)
    assert clean_text(raw)[0] == raw
    assert clean_text(raw, strip_bidi=True)[0] == "السعر 123 USD"


def test_preserves_legacy_bidi_embeddings_by_default():
    raw = "English \u202bالعربية\u202c end"
    assert clean_text(raw)[0] == raw
    assert clean_text(raw, strip_bidi=True)[0] == "English العربية end"


def test_strips_override_and_its_pdf_terminator():
    raw = "abc\u202edef\u202c"
    cleaned, stats = clean_text(raw)
    assert cleaned == "abcdef"
    assert stats["removed_count"] == 2


def test_strips_orphaned_bidi_embedding_controls():
    assert clean_text("abc\u202c")[0] == "abc"
    assert clean_text("abc\u202bdef")[0] == "abcdef"


def test_clean_preserves_normal_text():
    raw = "Normal ASCII and café — fine."
    cleaned, stats = clean_text(raw)
    assert cleaned == raw
    assert stats["removed_count"] == 0


def test_aggressive_confusable():
    # Cyrillic 'а' (U+0430) looks like Latin 'a'
    raw = "p\u0430y"  # p + cyrillic a + y
    cleaned, _ = clean_text(raw, aggressive_homoglyphs=True)
    assert cleaned == "pay"


def test_clean_preserves_emoji_vs16():
    raw = "Balance returns. \u2696\ufe0f"  # ⚖️
    cleaned, stats = clean_text(raw)
    assert cleaned == raw
    assert stats["removed_count"] == 0


def test_clean_preserves_arrow_emoji_vs16():
    raw = "Move \u2194\ufe0f"
    cleaned, stats = clean_text(raw)
    assert cleaned == raw
    assert stats["removed_count"] == 0


def test_clean_preserves_punctuation_and_letterlike_emoji_vs16():
    # Emoji=Yes singletons outside the block ranges: double exclamation,
    # exclamation question, information source, curved up/down arrows.
    for base in ("\u203c", "\u2049", "\u2139", "\u2934", "\u2935"):
        raw = f"note {base}\ufe0f end"
        cleaned, stats = clean_text(raw)
        assert cleaned == raw
        assert stats["removed_count"] == 0
    # And a ZWJ chain across one of them stays bound.
    raw = "\u2139\ufe0f\u200d\U0001f4a1"
    cleaned, _ = clean_text(raw)
    assert cleaned == raw


def test_inspect_punctuation_emoji_glue_not_suspicious():
    report = inspect_text("\u203c\ufe0f \u2049\ufe0f \u2139\ufe0f")
    assert report.suspicious_total == 0


def test_clean_preserves_cjk_ideographic_variation_selector():
    raw = "\u845b\U000e0100"  # CJK ideograph + VS17
    cleaned, stats = clean_text(raw)
    assert cleaned == raw
    assert stats["removed_count"] == 0


def test_clean_strips_repeated_cjk_variation_selector():
    raw = "\u845b\U000e0100\U000e0101"
    cleaned, stats = clean_text(raw)
    assert cleaned == "\u845b\U000e0100"
    assert stats["removed_count"] == 1


def test_clean_preserves_mongolian_variation_selector():
    raw = "\u1820\u180b"
    cleaned, stats = clean_text(raw)
    assert cleaned == raw
    assert stats["removed_count"] == 0


def test_clean_preserves_zwj_family():
    raw = "Family time: \U0001f468\u200d\U0001f469\u200d\U0001f467"  # 👨‍👩‍👧
    cleaned, stats = clean_text(raw)
    assert cleaned == raw
    assert stats["removed_count"] == 0


def test_clean_preserves_zwj_chain():
    raw = "\u2764\ufe0f\u200d\U0001f525"  # ❤️‍🔥
    cleaned, stats = clean_text(raw)
    assert cleaned == raw
    assert stats["removed_count"] == 0


def test_clean_strips_floating_emoji_glue():
    raw = "a\u200db\ufe0f"
    cleaned, stats = clean_text(raw)
    assert cleaned == "ab"
    assert stats["removed_count"] == 2


def test_inspect_emoji_glue_not_suspicious_by_default():
    raw = "Balance returns. \u2696\ufe0f Family time: \U0001f468\u200d\U0001f469\u200d\U0001f467"
    report = inspect_text(raw)
    assert report.suspicious_total == 0


def test_inspect_floating_emoji_glue_is_suspicious():
    raw = "a\u200d"
    report = inspect_text(raw)
    assert report.suspicious_total >= 1


def test_clean_strip_emoji_glue_flag():
    raw = "\u2696\ufe0f"
    cleaned, stats = clean_text(raw, strip_emoji_glue=True)
    assert cleaned == "\u2696"
    assert stats["removed_count"] == 1


def test_inspect_strip_emoji_glue_flag():
    raw = "\u2696\ufe0f"
    report = inspect_text(raw, strip_emoji_glue=True)
    assert report.suspicious_total >= 1


def test_clean_preserves_script_joiners():
    # Persian mi-ravam (ZWNJ) and a Devanagari conjunct (ZWJ) \u2014 orthographic.
    for raw in ("\u0645\u06cc\u200c\u0631\u0648\u0645", "\u0915\u094d\u200d\u0937"):
        cleaned, _ = clean_text(raw)
        assert cleaned == raw


def test_clean_preserves_flag_tag_sequence():
    # Scotland flag: emoji base U+1F3F4 + tag chars ending in U+E007F.
    raw = "\U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f"
    cleaned, _ = clean_text(raw)
    assert cleaned == raw


def test_clean_strips_incomplete_flag_tag_sequence():
    raw = "\U0001f3f4\U000e0067\U000e0062"
    cleaned, stats = clean_text(raw)
    assert cleaned == "\U0001f3f4"
    assert stats["removed_count"] == 2


def test_clean_strips_joiner_between_unrelated_scripts():
    raw = "\u845b\u200cA"
    cleaned, stats = clean_text(raw)
    assert cleaned == "\u845bA"
    assert stats["removed_count"] == 1


def test_clean_preserves_orthographic_arabic_cf():
    raw = "x\u0600y\u06ddz"  # ARABIC NUMBER SIGN, END OF AYAH
    cleaned, _ = clean_text(raw)
    assert cleaned == raw


def test_clean_still_strips_joiners_between_latin():
    # ZWJ/ZWNJ next to ASCII is a carrier, not orthography \u2014 still removed.
    for raw in ("a\u200db", "a\u200cb", "ab\u200c"):
        cleaned, _ = clean_text(raw)
        assert "\u200c" not in cleaned and "\u200d" not in cleaned


def test_strip_emoji_glue_flag_restores_blanket_strip():
    cleaned, _ = clean_text("\u0645\u06cc\u200c\u0631", strip_emoji_glue=True)
    assert "\u200c" not in cleaned
    assert clean_text("x\u0600y", strip_emoji_glue=True)[0] == "xy"


def test_nfkc_change_is_in_replacement_count():
    cleaned, stats = clean_text("Ａ", nfkc=True)
    assert cleaned == "A"
    assert stats["nfkc_changed"] is True
    assert stats["replaced_count"] == 1


def test_nfkc_counts_changed_input_codepoints():
    cleaned, stats = clean_text("ＡＢ ﬃ", nfkc=True)
    assert cleaned == "AB ffi"
    assert stats["replaced"]["NFKC_normalize"] == 3
    assert stats["replaced_count"] == 3


def test_nfkc_counts_contextual_composition_input_codepoints():
    raw = "A\u030a A\u030a"
    cleaned, stats = clean_text(raw, nfkc=True)
    assert cleaned == "\u00c5 \u00c5"
    assert stats["replaced"]["NFKC_normalize"] == 4


def test_clean_preserves_mongolian_fvs():
    # Mongolian letter + FVS1/2/3 selects a positional glyph variant.
    for raw in ("\u1820\u180b\u1821", "\u1820\u180c\u1821", "\u1820\u180d\u1821"):
        cleaned, _ = clean_text(raw)
        assert cleaned == raw
    # FVS can chain after a single letter; both must stay bound to the base.
    raw = "\u1820\u180b\u180c\u1821"
    cleaned, _ = clean_text(raw)
    assert cleaned == raw


def test_clean_preserves_mongolian_fvs4():
    # FVS4 (U+180F, added in Unicode 14) selects a glyph variant just like
    # FVS1-3 and must stay bound to its Mongolian base.
    raw = "ᠠ᠏ᠡ"
    cleaned, stats = clean_text(raw)
    assert cleaned == raw
    assert stats["removed_count"] == 0


def test_clean_strips_missed_default_ignorable_carriers():
    # Mongolian FVS4, Hangul Filler, and Halfwidth Hangul Filler are
    # blank-rendering Default_Ignorable carriers (Mn/Lo, so the Cf catch-all
    # never saw them). Between Latin they are contraband and must be stripped.
    for cp in (0x180F, 0x3164, 0xFFA0):
        raw = "word" + chr(cp) + "word"
        cleaned, stats = clean_text(raw)
        assert cleaned == "wordword"
        assert stats["removed_count"] == 1


def test_inspect_flags_missed_default_ignorable_carriers():
    for cp in (0x180F, 0x3164, 0xFFA0):
        report = inspect_text("word" + chr(cp) + "word")
        assert report.suspicious_total >= 1


# Unassigned code points carrying Other_Default_Ignorable_Code_Point=Yes:
# reserved for future default-ignorables, so conformant renderers display
# them invisibly, and normalisation preserves them. Perfect covert carriers.
# Ranges transcribed from Unicode PropList.txt independently of the
# implementation table, so a typo in either shows up as a mismatch.
RESERVED_IGNORABLE_CPS = [
    0x2065,
    0xE0000,
    *range(0xFFF0, 0xFFF9),
    *range(0xE0080, 0xE0100),
    *range(0xE01F0, 0xE1000),
]


def test_clean_strips_reserved_default_ignorables():
    for cp in RESERVED_IGNORABLE_CPS:
        raw = "word" + chr(cp) + "word"
        cleaned, stats = clean_text(raw)
        assert cleaned == "wordword", f"U+{cp:04X} not stripped"
        assert stats["removed_count"] == 1


def test_inspect_reports_reserved_ignorable_kind():
    for cp in RESERVED_IGNORABLE_CPS:
        report = inspect_text("word" + chr(cp) + "word")
        assert any(h.kind == "reserved_ignorable" for h in report.hits), (
            f"U+{cp:04X} not reported as reserved_ignorable"
        )


def test_reserved_ignorable_does_not_claim_assigned_neighbours():
    # Boundary check: U+2064 (invisible plus, Cf) and U+FFF9 (interlinear
    # annotation anchor) are assigned and already handled under other kinds;
    # U+E0001 is a tag character. None of them may report reserved_ignorable.
    for cp in (0x2064, 0xFFF9, 0xE0001, 0xE0100):
        report = inspect_text("word" + chr(cp) + "word")
        assert not any(h.kind == "reserved_ignorable" for h in report.hits), (
            f"U+{cp:04X} wrongly reported as reserved_ignorable"
        )


def test_clean_preserves_khmer_inherent_vowels():
    # Invisible but phonemic inherent vowels after a Khmer consonant.
    for raw in ("\u1780\u17b4\u1781", "\u1780\u17b5\u1781"):
        cleaned, _ = clean_text(raw)
        assert cleaned == raw


def test_clean_preserves_hangul_fillers():
    # Fillers hold jamo slots in a partial syllable; removing them lets the
    # jamo compose into a different syllable (ᄀᅟᅡ vs 가).
    for raw in ("\u1100\u115f\u1161", "\u1100\u1160\u1161"):
        cleaned, _ = clean_text(raw)
        assert cleaned == raw


def test_clean_preserves_compatibility_and_halfwidth_hangul_fillers():
    # U+3164 after a compatibility jamo and U+FFA0 after a halfwidth jamo are
    # fillers in their own presentation form, preserved exactly like
    # U+115F/U+1160 after conjoining jamo.
    for raw in ("\u3131\u3164\u314f", "\uffa1\uffa0\uffc2"):
        cleaned, stats = clean_text(raw)
        assert cleaned == raw
        assert stats["removed_count"] == 0


def test_clean_still_strips_floating_compatibility_and_halfwidth_fillers():
    # Isolated or between Latin they stay contraband (see also
    # test_clean_strips_missed_default_ignorable_carriers).
    for raw in ("a\u3164b", "a\uffa0b", "\u3164", "\uffa0"):
        cleaned, _ = clean_text(raw)
        assert "\u3164" not in cleaned and "\uffa0" not in cleaned


def test_clean_still_strips_floating_script_glue():
    # Isolated between Latin these are contraband, not orthography.
    for raw in ("a\u180bb", "a\u17b4b", "a\u115fb", "\u180b", "\u1160"):
        cleaned, _ = clean_text(raw)
        assert cleaned == raw.replace("\u180b", "").replace("\u17b4", "").replace(
            "\u115f", ""
        ).replace("\u1160", "")


def test_clean_strip_emoji_glue_flag_strips_script_glue():
    for raw in ("\u1820\u180b\u1821", "\u1780\u17b4\u1781", "\u1100\u115f\u1161"):
        cleaned, _ = clean_text(raw, strip_emoji_glue=True)
        assert "\u180b" not in cleaned and "\u17b4" not in cleaned and "\u115f" not in cleaned


def test_clean_strips_private_use():
    # BMP + both supplementary PUA planes: no portable meaning, so stripped.
    raw = "a\ue000b\U000f0000c\U0010fffd"
    cleaned, stats = clean_text(raw)
    assert cleaned == "abc"
    assert stats["removed_count"] >= 3


def test_inspect_script_glue_not_suspicious_by_default():
    raw = "\u1820\u180b\u1821\u1780\u17b4\u1781\u1100\u115f\u1161"
    report = inspect_text(raw)
    assert report.suspicious_total == 0


def test_inspect_floating_script_glue_is_suspicious():
    for raw in ("a\u180b", "a\u17b4", "a\u115f"):
        report = inspect_text(raw)
        assert report.suspicious_total >= 1


def test_inspect_private_use():
    report = inspect_text("a\ue000b")
    assert any(h.kind == "private_use" for h in report.hits)


# The 66 Unicode noncharacters: U+FDD0..U+FDEF plus U+nFFFE/U+nFFFF at the
# end of every plane. Permanently reserved for internal use and prohibited in
# interchange (TUS 23.7), rendered as nothing or tofu, preserved by
# normalisation: covert carriers with no future-assignment risk at all.
# Transcribed independently of the implementation table.
NONCHARACTER_CPS = list(range(0xFDD0, 0xFDF0)) + [
    plane << 16 | low for plane in range(0x11) for low in (0xFFFE, 0xFFFF)
]


def test_noncharacter_list_is_complete():
    assert len(NONCHARACTER_CPS) == 66


def test_clean_strips_noncharacters():
    for cp in NONCHARACTER_CPS:
        raw = "word" + chr(cp) + "word"
        cleaned, stats = clean_text(raw)
        assert cleaned == "wordword", f"U+{cp:04X} not stripped"
        assert stats["removed_count"] == 1


def test_inspect_reports_noncharacter_kind():
    for cp in NONCHARACTER_CPS:
        report = inspect_text("word" + chr(cp) + "word")
        assert any(h.kind == "noncharacter" for h in report.hits), (
            f"U+{cp:04X} not reported as noncharacter"
        )


def test_noncharacter_does_not_claim_assigned_neighbours():
    # U+FDF0 (Arabic ligature) and U+FFFD (replacement character) sit right
    # next to noncharacter ranges and must stay untouched and unclaimed.
    for cp in (0xFDF0, 0xFFFD):
        raw = "word" + chr(cp) + "word"
        cleaned, _ = clean_text(raw)
        assert cleaned == raw, f"U+{cp:04X} wrongly altered"
        report = inspect_text(raw)
        assert not any(h.kind == "noncharacter" for h in report.hits), (
            f"U+{cp:04X} wrongly reported as noncharacter"
        )


def test_clean_preserves_egyptian_format_controls():
    # Quadrat layout controls (joiners, insert/segment pairs) visibly govern
    # how hieroglyphic text renders: body, not carriers, next to hieroglyphs.
    for raw in (
        "\U00013079\U00013430\U000130a7",  # glyph VERTICAL-JOINER glyph
        "\U00013437\U00013079\U000130a7\U00013438",  # BEGIN/END SEGMENT pair
    ):
        cleaned, stats = clean_text(raw)
        assert cleaned == raw
        assert stats["removed_count"] == 0


def test_clean_preserves_duployan_shorthand_controls():
    # LETTER OVERLAP between two Duployan letters, UP STEP after one.
    for raw in ("\U0001bc02\U0001bca0\U0001bc03", "\U0001bc02\U0001bca3"):
        cleaned, stats = clean_text(raw)
        assert cleaned == raw
        assert stats["removed_count"] == 0


def test_clean_preserves_musical_beam_controls():
    # BEGIN/END BEAM around two stemmed notes.
    raw = "\U0001d158\U0001d165\U0001d173\U0001d158\U0001d165\U0001d174"
    cleaned, stats = clean_text(raw)
    assert cleaned == raw
    assert stats["removed_count"] == 0


def test_clean_strips_floating_layout_format_controls():
    # Between unrelated text the same controls stay strip-class carriers.
    for cp in (0x13430, 0x13438, 0x1BCA0, 0x1BCA3, 0x1D173, 0x1D17A):
        raw = "word" + chr(cp) + "word"
        cleaned, stats = clean_text(raw)
        assert cleaned == "wordword", f"U+{cp:04X} not stripped when floating"
        assert stats["removed_count"] == 1


def test_inspect_layout_controls_in_context_not_suspicious():
    raw = "\U00013079\U00013430\U000130a7\U0001bc02\U0001bca0\U0001bc03"
    report = inspect_text(raw)
    assert report.suspicious_total == 0


def test_inspect_floating_layout_controls_suspicious():
    for cp in (0x13430, 0x1BCA0, 0x1D173):
        report = inspect_text("a" + chr(cp) + "b")
        assert report.suspicious_total >= 1


def test_strip_emoji_glue_flag_strips_layout_controls():
    # Paranoid mode keeps its blanket-strip semantics.
    cleaned, _ = clean_text("\U00013079\U00013430\U000130a7", strip_emoji_glue=True)
    assert "\U00013430" not in cleaned
