"""Tests for the cleaners security hardening (safe argv, resource caps, safe writes)."""

from __future__ import annotations

import io
import os
import re
import struct
import sys
import time
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import common
import container_meta
from common import (
    backup_path,
    read_text_input,
    safe_arg,
    safe_write_bytes,
    safe_write_text,
)
from container_meta import (
    _SVG_METADATA_CLOSE_RE,
    _SVG_METADATA_OPEN_RE,
    MAX_ZIP_DECOMPRESSED_BYTES,
    ZipBudgetExceeded,
    _check_zip_budget,
    _drop_tag_blocks,
    _pdf_structured_blob,
    _read_zip_member,
    _script_tag_is_jsonld,
    clean_html,
    clean_odt,
    clean_svg,
    inspect_docx,
)


def test_safe_arg_prefixes_leading_dash():
    assert safe_arg("-@evil") == "./-@evil"
    assert safe_arg("--argfile") == "./--argfile"


def test_safe_arg_leaves_normal_paths_alone():
    assert safe_arg("photo.png") == "photo.png"
    assert safe_arg("dir/file.svg") == "dir/file.svg"
    assert safe_arg("/abs/path.pdf") == "/abs/path.pdf"
    assert safe_arg(".") == "."


def test_zip_budget_rejects_oversized_member():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "<w:document/>")
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        info = zf.infolist()[0]
    info.file_size = MAX_ZIP_DECOMPRESSED_BYTES + 1
    raised = False
    try:
        _check_zip_budget(info, [0])
    except ZipBudgetExceeded:
        raised = True
    assert raised


def _patch_zip_declared_size(data: bytes, new_size: int) -> bytes:
    """Rewrite the uncompressed-size field of every central-directory entry.

    Produces the crafted-archive shape zip-bomb guards must defend
    against: the central directory (which backs ZipInfo.file_size) claims
    ``new_size`` decompressed bytes while the real deflate payload and its
    CRC stay intact. The local headers are left untouched.
    """
    out = bytearray(data)
    pos = 0
    while True:
        pos = out.find(b"PK\x01\x02", pos)
        if pos == -1:
            break
        out[pos + 24 : pos + 28] = struct.pack("<I", new_size)
        pos += 4
    return bytes(out)


def test_zip_budget_charges_real_bytes_not_declared(monkeypatch):
    # The budget must be charged on bytes actually produced, not on the
    # central directory claim (ZipInfo.file_size is attacker-controlled).
    # Here the archive declares 150 KiB for a member that really carries
    # 128 KiB; the accounting must follow reality.
    monkeypatch.setattr(container_meta, "MAX_ZIP_DECOMPRESSED_BYTES", 1 << 20)
    payload = b"y" * (1 << 17)  # 128 KiB
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("word/document.xml", payload)
    crafted = _patch_zip_declared_size(buf.getvalue(), (1 << 17) + (1 << 16))
    with zipfile.ZipFile(io.BytesIO(crafted)) as zf:
        info = zf.infolist()[0]
        assert info.file_size == (1 << 17) + (1 << 16)  # declared claim
        budget = [0]
        out = _read_zip_member(zf, info, budget)
    assert out == payload
    assert budget[0] == len(payload)  # real bytes, not the declared claim


def test_zip_budget_accumulates_real_bytes_across_members(monkeypatch):
    # The cap is cumulative across members: several members under the cap
    # individually can still exhaust it together.
    monkeypatch.setattr(container_meta, "MAX_ZIP_DECOMPRESSED_BYTES", 1 << 20)
    payload = b"a" * (1 << 19)  # 512 KiB
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.xml", payload)
        zf.writestr("b.xml", payload)
        zf.writestr("c.xml", payload)  # 3 x 512 KiB > 1 MiB cap
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        budget = [0]
        infos = zf.infolist()
        _read_zip_member(zf, infos[0], budget)
        _read_zip_member(zf, infos[1], budget)
        with pytest.raises(ZipBudgetExceeded):
            _read_zip_member(zf, infos[2], budget)


def test_inspect_docx_with_ai_markers_does_not_crash():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "<w:document/>")
        zf.writestr(
            "docProps/app.xml",
            "<Properties><Application>Claude AI Writer</Application></Properties>",
        )
        zf.writestr(
            "docProps/core.xml",
            "<cp:coreProperties><dc:creator>Anthropic</dc:creator></cp:coreProperties>",
        )
    has_c2pa, has_ai, findings, _ = inspect_docx(buf.getvalue())
    assert has_ai
    assert findings
    assert not has_c2pa or has_ai


# ---------------------------------------------------------------------------
# Safe (atomic, symlink-safe) writes
# ---------------------------------------------------------------------------


def _make_symlink(dest: Path, target: Path) -> None:
    """Create a symlink, skipping where the platform denies the privilege."""
    try:
        dest.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


def test_safe_write_refuses_symlink_destination(tmp_path: Path):
    victim = tmp_path / "victim.txt"
    victim.write_text("PRECIOUS DATA")
    dest = tmp_path / "out.txt"
    _make_symlink(dest, victim)
    with pytest.raises(OSError):
        safe_write_text(dest, "cleaned content")
    # The victim must be untouched and no temp litter may remain.
    assert victim.read_text() == "PRECIOUS DATA"
    assert not list(tmp_path.glob("*.tmp"))


def test_safe_write_atomically_replaces_existing_file(tmp_path: Path):
    dest = tmp_path / "out.txt"
    safe_write_text(dest, "first")
    safe_write_text(dest, "second")
    assert dest.read_text() == "second"
    # No stray temp files after a successful write.
    assert not list(tmp_path.glob("*.tmp"))


def test_safe_write_bytes_creates_parent_dirs(tmp_path: Path):
    dest = tmp_path / "a" / "b" / "out.bin"
    safe_write_bytes(dest, b"\x00\x01")
    assert dest.read_bytes() == b"\x00\x01"


def test_safe_write_bytes_without_fchmod(tmp_path: Path, monkeypatch):
    # os.fchmod is POSIX-only; on Windows the write must still go through.
    monkeypatch.delattr(os, "fchmod", raising=False)
    dest = tmp_path / "out.bin"
    safe_write_bytes(dest, b"payload")
    assert dest.read_bytes() == b"payload"
    assert not list(tmp_path.glob("*.tmp"))


def test_backup_path_creates_bak_copy(tmp_path: Path):
    src = tmp_path / "doc.md"
    src.write_text("body")
    bak, created = backup_path(src)
    assert bak.name == "doc.md.bak"
    assert created is True
    assert bak.read_text() == "body"
    assert src.read_text() == "body"


def test_backup_path_refuses_symlinked_bak(tmp_path: Path):
    # An existing .bak is now PRESERVED rather than written through (#172), so
    # a symlinked .bak can no longer trigger the write path at all: the call
    # succeeds without touching the symlink target.
    src = tmp_path / "doc.md"
    src.write_text("body")
    bak = tmp_path / "doc.md.bak"
    victim = tmp_path / "victim.txt"
    victim.write_text("PRECIOUS")
    _make_symlink(bak, victim)
    result, created = backup_path(src)
    assert created is False
    assert result == bak
    assert victim.read_text() == "PRECIOUS"


# ---------------------------------------------------------------------------
# Input size caps (stdin + file)
# ---------------------------------------------------------------------------


def test_read_text_input_refuses_oversized_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(common, "MAX_INPUT_BYTES", 8)
    big = tmp_path / "big.txt"
    big.write_text("x" * 64)
    with pytest.raises(SystemExit):
        read_text_input(str(big))


def test_read_stdin_capped(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(common, "MAX_STDIN_BYTES", 16)
    monkeypatch.setattr(sys, "stdin", io.StringIO("x" * 64))
    with pytest.raises(SystemExit):
        read_text_input(None)


def test_read_stdin_under_cap_ok(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(common, "MAX_STDIN_BYTES", 1024)
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello stdin"))
    assert read_text_input(None) == "hello stdin"


# ---------------------------------------------------------------------------
# Quadratic-regex DoS (GHSA-7vpp-96qp-j9wh): lazy .*? block stripping
# ---------------------------------------------------------------------------
#
# clean_svg / clean_odt (and the other tag-block scans below) used lazy
# ".*?</close>" regexes. With many opening tags and no closing tag, CPython
# rescans to end-of-input from every candidate start - quadratic, and the
# GIL stalls the whole single-process service. The scans are now linear;
# these tests assert both the timing fix and the preserved behavior.


def _assert_completes_fast(fn, *args, budget_s: float = 5.0):
    start = time.perf_counter()
    result = fn(*args)
    elapsed = time.perf_counter() - start
    assert elapsed < budget_s, f"took {elapsed:.1f}s (linear scan expected < {budget_s}s)"
    return result


def test_clean_svg_metadata_flood_is_linear():
    # ~256 KiB of unclosed <metadata> tags. The old lazy regex measured
    # ~25 s at this size; the linear scan is milliseconds.
    flood = b"<metadata>" * (256 * 1024 // 11)
    assert b"</metadata>" not in flood
    cleaned, actions = _assert_completes_fast(clean_svg, flood)
    # No closing tag exists, so nothing is stripped - the block is preserved.
    assert cleaned == flood
    assert not any(a.startswith("drop <metadata>") for a in actions)


def test_clean_svg_mixed_closed_and_unclosed_metadata():
    svg = b"<svg><metadata>c2pa Anthropic</metadata><metadata>open</svg>"
    cleaned, actions = clean_svg(svg)
    # The closed block is stripped; the unclosed one is preserved verbatim.
    assert b"<metadata>c2pa" not in cleaned
    assert b"<metadata>open</svg>" in cleaned
    assert any(a.startswith("drop <metadata>") for a in actions)


def test_clean_odt_generator_flood_is_linear():
    # GHSA-7vpp-96qp-j9wh PoC shape: an ODT whose meta.xml is ~512 KiB of
    # unclosed <meta:generator> tags (about 1.4 KiB on the wire after DEFLATE).
    payload = b"<meta:generator>" * (512 * 1024 // 16)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", b"application/vnd.oasis.opendocument.text")
        zf.writestr("content.xml", b"<x/>")
        zf.writestr("meta.xml", payload)
    cleaned, _actions = _assert_completes_fast(clean_odt, buf.getvalue())
    # Unclosed blocks are preserved: the payload round-trips untouched.
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        assert zf.read("meta.xml") == payload


def test_clean_svg_comment_end_forms():
    # HTML comment end tags are "-->" or "--!>" (CodeQL bad-html-filtering
    # regexp). Comments in either form must be stripped when they carry AI
    # provenance markers and kept otherwise.
    svg = b"<svg><!-- Anthropic --!><!-- plain --></svg>"
    cleaned, actions = clean_svg(svg)
    assert b"Anthropic" not in cleaned
    assert b"plain" in cleaned
    assert any("comment" in a for a in actions)


def test_clean_html_jsonld_flood_is_linear():
    flood = '<script type="application/ld+json">' * (128 * 1024 // 39)
    assert "</script>" not in flood
    cleaned, _actions = _assert_completes_fast(clean_html, flood)
    assert cleaned == flood


def test_clean_html_jsonld_attr_flood_no_close_is_linear():
    # CodeQL py/polynomial-redos worst case for the tag scanner: the literal
    # 'type="application/ld+json"' repeats many times but there is never a '>'.
    # A single "<script[^>]*type...[^>]*>" regex backtracks quadratically here;
    # the linear tag scan + separate type check must not.
    flood = '<script type="application/ld+json"' * (128 * 1024 // 39)
    assert ">" not in flood
    cleaned, _actions = _assert_completes_fast(clean_html, flood)
    assert cleaned == flood


def test_clean_html_data_uri_flood_is_linear():
    # CodeQL py/polynomial-redos worst case for the data-URI matcher: many
    # 'data:image/+;' prefixes and no comma. The old params class could match
    # the comma (and ';'), so every prefix rescanned the tail - quadratic. The
    # bounded, unambiguous header must keep the whole pass linear.
    flood = "data:image/+;" * (256 * 1024 // 13)
    assert "," not in flood
    cleaned, actions = _assert_completes_fast(clean_html, flood)
    assert cleaned == flood
    assert not any("embedded data:image" in a for a in actions)


def test_jsonld_type_parser_is_linear_on_whitespace():
    # A whitespace-heavy tag without a type must be rejected quickly (a "\s+"
    # or search-based matcher here would rescan the run from every position).
    flood = "<script" + "  " * (512 * 1024) + ">"
    start = time.perf_counter()
    assert _script_tag_is_jsonld(flood) is False
    assert time.perf_counter() - start < 5.0


def test_jsonld_type_parser_ignores_decoy_attributes():
    # A 'type=' inside another attribute's quoted value, and a name that merely
    # ends in "type" (data-type), are NOT the JSON-LD type attribute.
    assert _script_tag_is_jsonld('<script data-type="application/ld+json">') is False
    assert _script_tag_is_jsonld("<script title='x type=\"application/ld+json\" y'>") is False
    assert _script_tag_is_jsonld('<script id="type" type="application/ld+json">') is True


def test_pdf_stream_and_xpacket_floods_are_linear():
    stream_flood = b"stream\n" * (256 * 1024 // 8)
    assert b"endstream" not in stream_flood
    blob = _assert_completes_fast(_pdf_structured_blob, stream_flood)
    assert blob == stream_flood + b"\n"

    xpacket_flood = b"<?xpacket begin" * (256 * 1024 // 15)
    assert b"<?xpacket end" not in xpacket_flood
    blob = _assert_completes_fast(_pdf_structured_blob, xpacket_flood)
    assert blob == xpacket_flood + b"\n"


def test_tag_block_scan_matches_lazy_regex_semantics():
    # The linear scan must strip exactly what the old lazy regex stripped,
    # on well- and malformed inputs alike (closed, unclosed, mixed, nested,
    # case variants, and a closing tag smuggled inside an attribute value).
    lazy = re.compile(r"<metadata\b[^>]*>.*?</metadata\s*>", re.I | re.DOTALL)
    samples = [
        "",
        "<metadata>a</metadata>",
        "<metadata>a</metadata><metadata>b</metadata>",
        "<metadata>unclosed",
        "<metadata>a</metadata><metadata>unclosed",
        "x<metadata>a</metadata>y",
        "<metadata foo='x'>a</metadata >",
        "<METADATA>a</METADATA>",
        "<metadata><metadata>nested</metadata>",
        "<metadata>a</metadata><metadata>b</metadata><metadata>c",
        "<metadata foo='" + "</metadata>" + "'>tail</metadata>",
    ]
    for sample in samples:
        expected, n_expected = re.subn(lazy, "", sample)
        got, n_got = _drop_tag_blocks(sample, _SVG_METADATA_OPEN_RE, _SVG_METADATA_CLOSE_RE)
        assert got == expected, (sample, expected, got)
        assert n_got == n_expected, (sample, n_expected, n_got)


def test_scrub_text_runs_is_linear_on_unclosed_runs():
    # (w:t)-style body scrubs share the same lazy pattern class; a run of
    # unclosed <w:t> tags must not blow up either.
    from container_meta import _scrub_docx_text

    flood = "<w:t>" * (128 * 1024 // 5)
    assert "</w:t>" not in flood
    start = time.perf_counter()
    out, removed, replaced = _scrub_docx_text(flood)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"took {elapsed:.1f}s"
    assert out == flood
    assert removed == 0 and replaced == 0


def test_reconfigure_stream_writes_utf8():
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1252")
    common._reconfigure_stream(stream, "backslashreplace")
    stream.write("\u200b")
    stream.flush()
    assert buf.getvalue() == "\u200b".encode("utf-8")
