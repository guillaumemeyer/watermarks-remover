"""Tests for finding confidence and aggregate audit scripts."""

from __future__ import annotations

import gzip
import struct
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_website
from audit_lib import aggregate, is_actionable, scan_file
from audit_website import guess_kind, inspect_remote, parse_sitemap
from common import classify_finding_confidence
from container_meta import inspect_container
from text_unicode import inspect_text


def test_classify_finding_confidence_buckets():
    cases = {
        "c2patool reports a C2PA-related manifest": "confirmed",
        "PNG chunk caBX (possible C2PA container)": "confirmed",
        "JPEG APP11 segment (JUMBF/C2PA common)": "confirmed",
        "pdf-structured:ai:digitalSourceType": "confirmed",
        "pdf-structured:ai:AIGC": "probable",
        "PNG tEXt: c2pa, contentcredentials": "probable",
        "frontmatter key: generator": "probable",
        'info: cms generator: <meta name="generator" content="WordPress">': "informational",
        "customXml parts: 1": "informational",
        "unsupported container: woff2": "informational",
        "svg <metadata> present": "informational",
        "byte-scan C2PA markers: c2pa": "likely_false_positive",
    }
    for finding, expected in cases.items():
        assert classify_finding_confidence(finding) == expected, finding


def test_text_report_hit_confidence():
    report = inspect_text("a\u200bb")
    assert report.to_dict()["hits"][0]["confidence"] == "probable"

    # Exotic space homoglyphs are weaker context.
    report = inspect_text("a\u2003b")
    assert report.to_dict()["hits"][0]["confidence"] == "informational"


def test_container_report_findings_confidence(tmp_path: Path):
    src = tmp_path / "cms.html"
    src.write_text(
        '<html><head><meta name="generator" content="WordPress 6.0"></head></html>',
        encoding="utf-8",
    )
    report = inspect_container(src)
    assert report.to_dict()["findings_confidence"] == ["informational"]


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _minimal_png_with_text() -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00")
    text = b"Comment\x00c2pa contentcredentials"
    return (
        sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"tEXt", text)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def test_image_report_findings_confidence(tmp_path: Path):
    from image_meta import inspect_image

    src = tmp_path / "t.png"
    src.write_bytes(_minimal_png_with_text())
    report = inspect_image(src)
    d = report.to_dict()
    assert "findings_confidence" in d
    assert any(c in ("probable", "confirmed") for c in d["findings_confidence"])


def test_scan_file_text_and_html(tmp_path: Path):
    text = tmp_path / "a.txt"
    text.write_text("Hello\u200bWorld\n", encoding="utf-8")
    item = scan_file(text)
    assert item["kind"] == "text"
    assert is_actionable(item)

    html = tmp_path / "b.html"
    html.write_text(
        '<html><head><meta name="generator" content="WordPress"></head><body>ok</body></html>',
        encoding="utf-8",
    )
    item = scan_file(html)
    assert item["kind"] == "html"
    assert not is_actionable(item)


def test_scan_file_container_layer_a_reported_once(tmp_path: Path):
    """Layer A body findings come from inspect_container() exactly once."""
    md = tmp_path / "post.md"
    md.write_text("# Title\n\nHello\u200bWorld\n", encoding="utf-8")
    item = scan_file(md)
    layer_a = [f for f in item["findings"] if "layer-a" in f]
    assert len(layer_a) == 1
    assert item["suspicious_total"] == 1
    assert is_actionable(item)


def test_aggregate_summary():
    files = [
        {
            "path": "a.txt",
            "kind": "text",
            "has_c2pa": False,
            "has_ai_metadata": False,
            "suspicious_total": 1,
            "findings": ["layer-a [zwj_family] U+200B ZERO WIDTH SPACE (Cf) x1"],
            "confidence": ["probable"],
        },
        {
            "path": "b.html",
            "kind": "html",
            "has_c2pa": False,
            "has_ai_metadata": False,
            "suspicious_total": 0,
            "findings": ["info: cms generator: <meta>"],
            "confidence": ["informational"],
        },
    ]
    summary = aggregate(files)
    assert summary["total"] == 2
    assert summary["actionable_files"] == 1
    assert summary["findings_by_confidence"]["probable"] == 1
    assert summary["findings_by_confidence"]["informational"] == 1
    assert summary["with_suspicious_text"] == 1


def test_parse_sitemap_urlset_and_index():
    data = (
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<url><loc>https://x.test/a</loc></url></urlset>"
    )
    kind, urls = parse_sitemap(data)
    assert kind == "urlset"
    assert urls == ["https://x.test/a"]

    idx = b"<sitemapindex><sitemap><loc>https://x.test/s1.xml</loc></sitemap></sitemapindex>"
    assert parse_sitemap(idx) == ("sitemapindex", ["https://x.test/s1.xml"])

    assert parse_sitemap(gzip.compress(data))[0] == "urlset"


def test_guess_kind():
    assert guess_kind("https://x.test/a", b"<html>x</html>", "text/html") == "html"
    assert guess_kind("https://x.test/a.png", b"\x89PNG\r\n\x1a\n", None) == "png"
    assert guess_kind("https://x.test/a", b"%PDF-1.4", None) == "pdf"


def test_inspect_remote_html_cms_informational():
    result = inspect_remote(
        "https://x.test/page",
        b'<html><head><meta name="generator" content="WordPress"></head><body>hi</body></html>',
        "text/html",
    )
    assert result["kind"] == "html"
    assert result["confidence"] == ["informational"]
    assert not is_actionable(result)


def test_parse_sitemap_rejects_dtd_and_entity_bomb():
    """Regression: billion-laughs via internal DTD entities must be refused.

    parse_sitemap rejects any document declaring a DTD or entities before
    parsing (audit_website.py:75). A hostile sitemap of a few hundred bytes
    would otherwise expand to gigabytes during ET.fromstring, OOM-ing the
    audit host. See GHSA-pjg6-92pm-mmcf.
    """
    bomb = (
        b'<?xml version="1.0"?>'
        b"<!DOCTYPE sitemapindex ["
        b' <!ENTITY a "lol">'
        b' <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">'
        b' <!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">'
        b' <!ENTITY d "&c;&c;&c;&c;&c;&c;&c;&c;&c;&c;">'
        b' <!ENTITY e "&d;&d;&d;&d;&d;&d;&d;&d;&d;&d;">'
        b' <!ENTITY f "&e;&e;&e;&e;&e;&e;&e;&e;&e;&e;">'
        b' <!ENTITY g "&f;&f;&f;&f;&f;&f;&f;&f;&f;&f;">'
        b"]>"
        b"<sitemapindex><loc>&g;</loc></sitemapindex>"
    )
    assert b"<!DOCTYPE" in bomb  # sanity: this is the entity-bomb payload

    with pytest.raises(ValueError, match="declares a DTD / entities"):
        parse_sitemap(bomb)

    # Same refusal for the gzip-compressed form (the guard runs on the
    # decompressed bytes, after the size cap).
    with pytest.raises(ValueError, match="declares a DTD / entities"):
        parse_sitemap(gzip.compress(bomb))

    # A bare DOCTYPE with no entities is also refused, so the rejection does
    # not depend on recognizing a particular bomb shape.
    with pytest.raises(ValueError, match="declares a DTD / entities"):
        parse_sitemap(b"<!DOCTYPE sitemapindex><sitemapindex></sitemapindex>")


def test_parse_sitemap_rejects_oversized_gzip(monkeypatch):
    monkeypatch.setattr(
        audit_website,
        "MAX_SITEMAP_DECOMPRESSED_BYTES",
        128,
    )

    payload = b"<urlset>" + (b" " * 256) + b"</urlset>"

    with pytest.raises(
        ValueError,
        match="decompressed size exceeds cap",
    ):
        audit_website.parse_sitemap(gzip.compress(payload))


def test_validate_public_url_rejects_private_literal():
    with pytest.raises(ValueError, match="non-public address"):
        audit_website._validate_public_http_url("http://127.0.0.1/secret")


def test_validate_public_url_rejects_private_dns(monkeypatch):
    def fake_getaddrinfo(host, port, *, type):
        return [
            (
                audit_website.socket.AF_INET,
                audit_website.socket.SOCK_STREAM,
                6,
                "",
                ("10.0.0.8", port),
            )
        ]

    monkeypatch.setattr(
        audit_website.socket,
        "getaddrinfo",
        fake_getaddrinfo,
    )

    with pytest.raises(ValueError, match="non-public address"):
        audit_website._validate_public_http_url("https://example.com/page")


def test_validate_public_url_accepts_public_dns(monkeypatch):
    def fake_getaddrinfo(host, port, *, type):
        return [
            (
                audit_website.socket.AF_INET,
                audit_website.socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", port),
            )
        ]

    monkeypatch.setattr(
        audit_website.socket,
        "getaddrinfo",
        fake_getaddrinfo,
    )

    assert audit_website._validate_public_http_url("https://example.com/page") == (
        "https",
        "example.com",
        443,
    )


def test_collect_urls_rejects_cross_origin_entry(monkeypatch):
    def fake_getaddrinfo(host, port, *, type):
        return [
            (
                audit_website.socket.AF_INET,
                audit_website.socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", port),
            )
        ]

    def fake_fetch(
        url,
        timeout,
        max_bytes,
        *,
        allowed_origin=None,
    ):
        return (
            b"<urlset><url><loc>http://127.0.0.1/secret</loc></url></urlset>",
            "application/xml",
        )

    monkeypatch.setattr(
        audit_website.socket,
        "getaddrinfo",
        fake_getaddrinfo,
    )
    monkeypatch.setattr(
        audit_website,
        "fetch",
        fake_fetch,
    )

    with pytest.raises(
        ValueError,
        match="cross-origin sitemap URL",
    ):
        audit_website.collect_urls(
            "https://example.com/sitemap.xml",
            1,
            10,
        )


def test_pinned_connection_uses_validated_numeric_ip(monkeypatch):
    seen = []

    class FakeSocket:
        def close(self):
            pass

    def fake_create_connection(address, timeout=None):
        seen.append(address)
        return FakeSocket()

    monkeypatch.setattr(
        audit_website.socket,
        "create_connection",
        fake_create_connection,
    )

    conn = audit_website._open_pinned_connection(
        ("http", "example.com", 80),
        ("93.184.216.34",),
        1,
    )

    try:
        assert seen == [("93.184.216.34", 80)]
        assert conn.host == "example.com"
    finally:
        conn.close()


def test_fetch_revalidates_redirect_before_second_connection(monkeypatch):
    dns_calls = 0
    connections = 0

    def fake_getaddrinfo(host, port, *, type):
        nonlocal dns_calls
        dns_calls += 1
        address = "93.184.216.34" if dns_calls == 1 else "10.0.0.9"
        return [
            (
                audit_website.socket.AF_INET,
                audit_website.socket.SOCK_STREAM,
                6,
                "",
                (address, port),
            )
        ]

    class RedirectResponse:
        status = 302

        def getheader(self, name, default=None):
            if name == "Location":
                return "https://example.com/private"
            return default

        def close(self):
            pass

    class RedirectConnection:
        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return RedirectResponse()

        def close(self):
            pass

    def fake_open(origin, addresses, timeout):
        nonlocal connections
        connections += 1
        return RedirectConnection()

    monkeypatch.setattr(
        audit_website.socket,
        "getaddrinfo",
        fake_getaddrinfo,
    )
    monkeypatch.setattr(
        audit_website,
        "_open_pinned_connection",
        fake_open,
    )

    with pytest.raises(ValueError, match="non-public address"):
        audit_website.fetch(
            "https://example.com/start",
            1,
            1024,
        )

    assert connections == 1


def test_audit_website_partial_scan_exit_3(monkeypatch):
    """A URL that fails to fetch makes the audit partial: exit 3, distinct
    from both "clean" (0) and "actionable findings" (1)."""

    def _fetch(*_args, **_kwargs):
        raise ValueError("connection refused")

    monkeypatch.setattr(
        audit_website,
        "collect_urls",
        lambda *_a, **_k: ["https://example.com/1", "https://example.com/2"],
    )
    monkeypatch.setattr(audit_website, "fetch", _fetch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_website.py", "--sitemap", "https://example.com/sitemap.xml"],
    )
    assert audit_website.main() == 3


def test_main_rejects_private_base_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_website.py",
            "--base",
            "http://127.0.0.1",
        ],
    )

    assert audit_website.main() == 2

    err = capsys.readouterr().err
    assert "invalid base URL" in err
    assert "non-public address" in err


def test_unreadable_text_file_is_a_failed_scan_not_clean(tmp_path, monkeypatch):
    """A text file the scanner cannot open must surface as skipped/partial,
    never as scanned-and-clean (#158)."""
    from pathlib import Path

    import audit_lib

    target = tmp_path / "notes.txt"
    target.write_text("hello", encoding="utf-8")

    def _deny(self, *args, **kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(Path, "read_bytes", _deny)

    # scan_file raises instead of returning a confidence-less "clean" item.
    try:
        audit_lib.scan_file(target)
    except OSError:
        pass
    else:
        raise AssertionError("scan_file must propagate the read failure")

    # The old swallowed-error item shape must never count as actionable.
    assert audit_lib.is_actionable({"path": "x.txt", "kind": "text", "error": "denied"}) is False

    # audit_dir's worker converts the raise into a skipped entry.
    import audit_dir

    item, skipped = audit_dir._scan_worker(target, False)
    assert item is None
    assert skipped is not None and "Permission denied" in skipped.get("reason", "")
