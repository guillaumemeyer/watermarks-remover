"""Tests for EPUB container metadata, XHTML, embedded media, and encryption."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from container_meta import (
    clean_container,
    clean_epub,
    detect_container_format,
    inspect_container,
    inspect_epub,
)
from format_dispatch import classify, classify_bytes

from tests.test_clean_image import _minimal_png_with_text


def _build_epub(
    *,
    creator: str = "OpenAI",
    generator: str = "ChatGPT",
    body_text: str = "Chapter one\u200b with a hidden mark",
    with_c2pa_png: bool = True,
    with_meta_part: bool = True,
    encrypted_parts: tuple[str, ...] = (),
) -> bytes:
    generator_meta = f'    <meta name="generator" content="{generator}"/>\n' if generator else ""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "mimetype",
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" '
            'media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        if encrypted_parts:
            zf.writestr(
                "META-INF/encryption.xml",
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
                'xmlns:enc="http://www.w3.org/2001/04/xmlenc#">'
                + "".join(
                    f"<enc:EncryptedData><enc:CipherData>"
                    f'<enc:CipherReference URI="../{p}"/></enc:CipherData></enc:EncryptedData>'
                    for p in encrypted_parts
                )
                + "</encryption>",
            )
        zf.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
            '<dc:identifier id="uid">urn:uuid:1234</dc:identifier>'
            "<dc:title>Test Book</dc:title>"
            f"<dc:creator>{creator}</dc:creator>"
            "<dc:publisher>Acme Press</dc:publisher>"
            f"{generator_meta}  </metadata>"
            "<manifest>"
            '<item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="img" href="images/cover.png" media-type="image/png"/>'
            "</manifest>"
            '<spine><itemref idref="c1"/></spine>'
            "</package>",
        )
        zf.writestr(
            "OEBPS/chapter1.xhtml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<html xmlns="http://www.w3.org/1999/xhtml">'
            "<head><title>Chapter 1</title>"
            f"{generator_meta}  </head>"
            f"<body><p>{body_text}</p></body></html>",
        )
        if with_c2pa_png:
            zf.writestr("OEBPS/images/cover.png", _minimal_png_with_text())
        if with_meta_part:
            zf.writestr(
                "META-INF/metadata.xml",
                '<metadata xmlns="http://www.idpf.org/2013/metadata">c2pa contentcredentials</metadata>',
            )
    return buf.getvalue()


def test_detect_epub_by_extension_and_sniff(tmp_path: Path):
    epub = _build_epub()
    path = tmp_path / "book.epub"
    path.write_bytes(epub)
    assert detect_container_format(path, epub) == "epub"
    assert detect_container_format(Path("book.bin"), epub) == "epub"
    assert classify_bytes(epub, ".epub") == "container"
    assert classify_bytes(epub, "") == "container"
    assert classify(path) == "container"


def test_inspect_epub_detects_ai_metadata():
    has_c2pa, has_ai, findings, details = inspect_epub(_build_epub())
    assert has_ai is True
    assert has_c2pa is True
    assert any("content.opf" in f for f in findings)
    assert any("chapter1.xhtml" in f for f in findings)
    assert any("cover.png" in f for f in findings)
    assert details["parts"] >= 5


def test_inspect_epub_clean_book_has_no_flags():
    has_c2pa, has_ai, _findings, _details = inspect_epub(
        _build_epub(creator="Jane Doe", generator="", with_c2pa_png=False, with_meta_part=False)
    )
    assert has_c2pa is False
    assert has_ai is False


def test_clean_epub_strips_metadata_and_layer_a():
    epub = _build_epub(body_text="Chapter one\u200b with a hidden mark")
    cleaned, actions = clean_epub(epub, also_layer_a_text=True)
    assert any("creator" in a for a in actions)
    assert any("OPF meta" in a for a in actions)
    assert any("layer A text" in a for a in actions)
    assert any("cover.png" in a for a in actions)

    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        names = zf.namelist()
        opf = zf.read("OEBPS/content.opf").decode("utf-8")
        xhtml = zf.read("OEBPS/chapter1.xhtml").decode("utf-8")
        png = zf.read("OEBPS/images/cover.png")
    assert "META-INF/metadata.xml" not in names
    assert "OpenAI" not in opf
    assert "ChatGPT" not in opf
    assert "<dc:creator/>" in opf
    assert 'name="generator"' not in opf
    assert "\u200b" not in xhtml
    assert b"c2pa" not in png.lower()
    assert "mimetype" in names and "META-INF/container.xml" in names


def test_clean_epub_skips_encrypted_parts():
    # The "encrypted" XHTML is opaque ciphertext: cleaning it as text would
    # mangle bytes, so clean_epub must copy it verbatim.
    ciphertext = b"\x00\x01\x02\x03AIGC\xff\xfe\xfd binary blob"
    epub = _build_epub(
        body_text="",
        encrypted_parts=("OEBPS/chapter1.xhtml",),
    )
    # overwrite chapter1.xhtml with ciphertext inside a fresh zip
    buf = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(epub)) as zin,
        zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zout,
    ):
        for info in zin.infolist():
            raw = zin.read(info.filename)
            if info.filename == "OEBPS/chapter1.xhtml":
                raw = ciphertext
            zout.writestr(info, raw)
    epub = buf.getvalue()

    cleaned, _actions = clean_epub(epub, also_layer_a_text=True)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        assert zf.read("OEBPS/chapter1.xhtml") == ciphertext  # untouched


def test_clean_epub_roundtrip(tmp_path: Path):
    src = tmp_path / "book.epub"
    src.write_bytes(_build_epub())
    dest = tmp_path / "book.cleaned.epub"
    result = clean_container(src, dest)
    assert result["format"] == "epub"
    assert result["still_has_c2pa"] is False
    assert result["still_has_ai_metadata"] is False
    with zipfile.ZipFile(dest) as zf:
        assert zf.read("mimetype") == b"application/epub+zip"
    rep = inspect_container(dest)
    assert rep.format == "epub"
    assert rep.has_c2pa is False
    assert rep.has_ai_metadata is False


def test_clean_epub_keeps_plain_creator():
    epub = _build_epub(creator="Jane Doe", generator="", with_c2pa_png=False, with_meta_part=False)
    cleaned, actions = clean_epub(epub)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        opf = zf.read("OEBPS/content.opf").decode("utf-8")
    assert "Jane Doe" in opf
    assert not any("creator" in a for a in actions)


def test_clean_epub_prunes_opf_manifest_for_dropped_part():
    """A dropped non-content part must also lose its <item> entry (and any
    spine itemref) in the OPF manifest, or the book fails EPUB validation."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        zf.writestr(
            "OEBPS/content.opf",
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
            "<manifest>"
            '<item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="m1" href="../META-INF/custommeta.xml" media-type="application/xml"/>'
            "</manifest>"
            '<spine><itemref idref="c1"/></spine>'
            "</package>",
        )
        zf.writestr("OEBPS/chapter1.xhtml", "<html><body><p>hi</p></body></html>")
        zf.writestr(
            "META-INF/custommeta.xml",
            '<metadata xmlns="http://www.idpf.org/2013/metadata">Anthropic Claude</metadata>',
        )
    data = buf.getvalue()

    cleaned, actions = clean_epub(data)
    assert any("drop part META-INF/custommeta.xml" in a for a in actions)
    assert any("prune OPF manifest entries" in a for a in actions)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        assert "META-INF/custommeta.xml" not in zf.namelist()
        opf = zf.read("OEBPS/content.opf").decode("utf-8")
        assert "custommeta.xml" not in opf
        assert 'id="c1"' in opf


def _make_epub_with_marked_part(opf: str, part_name: str) -> bytes:
    """EPUB whose OPF manifest is `opf` verbatim, with one AI-marked binary part."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "OEBPS/content.opf",
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">' + opf + "</package>",
        )
        zf.writestr("OEBPS/ch1.xhtml", "<html><body><p>hi</p></body></html>")
        zf.writestr(f"OEBPS/{part_name}", b"generated by claude.ai watermark payload")
    return buf.getvalue()


def test_clean_epub_prunes_single_quoted_manifest_entries():
    """XML permits single-quoted attributes; a single-quoted <item>/<itemref>
    for a dropped part must prune like the double-quoted form (#130's class,
    in the OPF pruner)."""
    data = _make_epub_with_marked_part(
        "<manifest>"
        "<item id='c1' href='ch1.xhtml' media-type='application/xhtml+xml'/>"
        "<item id='m1' href='misc/notes.bin' media-type='application/octet-stream'/>"
        "</manifest>"
        "<spine><itemref idref='c1'/><itemref idref='m1'/></spine>",
        "misc/notes.bin",
    )

    cleaned, actions = clean_epub(data)
    assert any("drop part OEBPS/misc/notes.bin" in a for a in actions)
    assert any("prune OPF manifest entries" in a for a in actions)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        assert "OEBPS/misc/notes.bin" not in zf.namelist()
        opf = zf.read("OEBPS/content.opf").decode("utf-8")
    assert "notes.bin" not in opf
    assert "idref='m1'" not in opf
    # the kept chapter entry survives, quotes and all
    assert "ch1.xhtml" in opf


def test_clean_epub_prunes_entity_escaped_href():
    """An XML-escaped href (misc/a&amp;b.bin) names the same part as the
    archive member (misc/a&b.bin): it must resolve decoded, like the
    relationship pruner's targets (#394's class, in the OPF pruner)."""
    data = _make_epub_with_marked_part(
        "<manifest>"
        '<item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="m1" href="misc/a&amp;b.bin" media-type="application/octet-stream"/>'
        "</manifest>"
        '<spine><itemref idref="c1"/><itemref idref="m1"/></spine>',
        "misc/a&b.bin",
    )

    cleaned, actions = clean_epub(data)
    assert any("drop part OEBPS/misc/a&b.bin" in a for a in actions)
    assert any("prune OPF manifest entries" in a for a in actions)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        assert "OEBPS/misc/a&b.bin" not in zf.namelist()
        opf = zf.read("OEBPS/content.opf").decode("utf-8")
    assert "b.bin" not in opf
    assert 'idref="m1"' not in opf
    assert "ch1.xhtml" in opf
