"""Tests for SVG/HTML/MD/DOCX/ODT container cleaners."""

from __future__ import annotations

import io
import json
import posixpath
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from container_meta import (
    clean_container,
    clean_docx,
    clean_html,
    clean_markdown,
    clean_odt,
    clean_svg,
    detect_container_format,
    inspect_container,
    inspect_docx,
    inspect_html,
    inspect_markdown,
    inspect_odt,
    inspect_svg,
)


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_markdown_frontmatter():
    text = """---
title: Hello
generator: Claude
ai_generated: true
---
Body\u200b text.
"""
    _c2, has_ai, findings, _d = inspect_markdown(text)
    assert has_ai
    assert any("generator" in f or "ai" in f.lower() for f in findings)
    cleaned, actions = clean_markdown(text)
    assert "generator:" not in cleaned
    assert "ai_generated:" not in cleaned
    assert "title: Hello" in cleaned
    assert any("drop" in a for a in actions)


def test_markdown_frontmatter_with_blank_line_does_not_crash():
    """Regression: a blank line inside frontmatter used to raise IndexError."""
    text = "---\ntitle: Demo\n\nauthor: you\n---\nBody\n"
    cleaned, actions = clean_markdown(text)
    assert "title: Demo" in cleaned
    assert "author: you" in cleaned
    assert actions


def test_markdown_drops_nested_children_of_dropped_key():
    """Regression: nested values under a dropped AI key survived the clean."""
    text = "---\ntitle: Demo\nmodel:\n  name: claude-opus\n  version: 4\nauthor: you\n---\nBody\n"
    cleaned, actions = clean_markdown(text)
    assert "claude-opus" not in cleaned  # the leak
    assert "version: 4" not in cleaned
    assert "title: Demo" in cleaned  # siblings untouched
    assert "author: you" in cleaned
    assert any("drop frontmatter key: model" in a for a in actions)


def test_markdown_clean_output_is_no_longer_flagged():
    """Round-trip: re-inspecting a cleaned document reports nothing AI-ish."""
    text = "---\ntitle: Demo\nmodel:\n  name: claude-opus\ngenerator: Claude\n---\nBody\n"
    cleaned, _ = clean_markdown(text)
    _c2, has_ai, findings, _d = inspect_markdown(cleaned)
    assert not has_ai, findings


def test_markdown_preserves_comments_and_non_ai_keys():
    text = "---\n# editorial notes\ntitle: Demo\ntags:\n  - one\n  - two\n---\nBody\n"
    cleaned, _ = clean_markdown(text)
    assert "# editorial notes" in cleaned
    assert "- one" in cleaned and "- two" in cleaned


def test_html_meta_strip():
    html = """<html><head>
<meta name="generator" content="ChatGPT">
<meta name="viewport" content="width=device-width">
<meta name="description" content="ok">
</head><body data-ai-model="gpt">Hi</body></html>"""
    _c2, has_ai, _findings, _ = inspect_html(html)
    assert has_ai
    cleaned, actions = clean_html(html)
    assert "ChatGPT" not in cleaned
    assert "viewport" in cleaned
    assert "data-ai-model" not in cleaned
    assert any("drop" in a for a in actions)


def test_html_cms_generator_not_ai():
    html = '<meta name="generator" content="WordPress 6.0">'
    has_c2pa, has_ai, findings, _ = inspect_html(html)
    assert not has_c2pa
    assert not has_ai
    assert any("cms" in f for f in findings)


def test_html_cms_generator_preserved_by_clean():
    html = '<html><head><meta name="generator" content="WordPress 6.0"><meta name="viewport" content="width=device-width"></head></html>'
    cleaned, _actions = clean_html(html)
    assert "WordPress" in cleaned
    assert "viewport" in cleaned


def test_html_cms_generator_attribute_names_are_case_insensitive():
    for html in (
        '<META NAME="generator" CONTENT="WordPress 6.0">',
        '<meta Name="generator" Content="WordPress 6.0">',
    ):
        has_c2pa, has_ai, findings, _ = inspect_html(html)
        assert not has_c2pa
        assert not has_ai
        assert any("cms" in finding for finding in findings)
        assert clean_html(html)[0] == html

    ai_html = '<META NAME="generator" CONTENT="Claude">'
    assert inspect_html(ai_html)[1]
    assert clean_html(ai_html)[0] == ""


def test_html_ai_generator_still_dropped():
    html = '<meta name="generator" content="Claude">'
    cleaned, actions = clean_html(html)
    assert "Claude" not in cleaned
    assert any("drop" in a for a in actions)


def test_pdf_stream_byte_collision_not_ai(tmp_path: Path):
    from container_meta import inspect_pdf

    pdf = b"%PDF-1.4\n1 0 obj<< /Length 4 >>stream\nAIGC\nendstream\nendobj\n%%EOF\n"
    src = tmp_path / "collision.pdf"
    src.write_bytes(pdf)
    has_c2pa, has_ai, _findings, _ = inspect_pdf(src, pdf)
    assert not has_c2pa
    assert not has_ai


def test_svg_metadata():
    svg = b"""<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <metadata>c2pa contentcredentials Anthropic</metadata>
  <circle cx="1" cy="1" r="1"/>
</svg>"""
    has_c2pa, has_ai, _findings, _ = inspect_svg(svg)
    assert has_c2pa or has_ai
    cleaned, actions = clean_svg(svg)
    assert b"<metadata" not in cleaned.lower() or b"c2pa" not in cleaned.lower()
    assert b"<circle" in cleaned
    assert any("metadata" in a or "drop" in a for a in actions)


def _make_docx_with_app(
    app_name: str = "Microsoft Macintosh Word", company: str = "Acme AI Corp"
) -> bytes:
    """Generate a synthetic DOCX bytes payload containing extended app properties."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/customXml/item1.xml" ContentType="application/xml"/>
</Types>""",
        )
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Hello</w:t></w:r></w:p></w:body></w:document>',
        )
        zf.writestr(
            "docProps/app.xml",
            f'<?xml version="1.0"?><Properties><Application>{app_name}</Application><Company>{company}</Company><AppVersion>14.0000</AppVersion></Properties>',
        )
        zf.writestr(
            "customXml/item1.xml",
            '<?xml version="1.0"?><root>c2pa contentcredentials</root>',
        )
    return buf.getvalue()


def test_docx_strips_app_and_customxml(tmp_path: Path):
    """Verify DOCX cleans Application/Company and customXml while keeping AppVersion."""
    data = _make_docx_with_app()
    cleaned, actions = clean_docx(data)
    assert any(
        "customXml" in a or "Application" in a or "Company" in a or "drop" in a for a in actions
    )
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        names = zf.namelist()
        assert "word/document.xml" in names
        assert not any(n.startswith("customXml/") for n in names)
        app = zf.read("docProps/app.xml").decode()
        assert "Acme AI Corp" not in app
        assert "<Application></Application>" in app or "<Application/>" in app
        # AppVersion is preserved to avoid ECMA-376 schema corruption (#283)
        assert "<AppVersion>14.0000</AppVersion>" in app


def _dangling_rels(zip_bytes: bytes) -> list[str]:
    """Return every internal relationship whose target part is missing."""
    bad: list[str] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
        for rels in (n for n in names if n.endswith(".rels")):
            base = posixpath.dirname(posixpath.dirname(rels))
            text = zf.read(rels).decode()
            for m in re.finditer(r'<Relationship\b[^>]*Target="([^"]*)"[^>]*/>', text, re.I):
                target, tag = m.group(1), m.group(0)
                if re.search(r"\bTargetMode\s*=", tag, re.I):
                    continue  # external
                if target.startswith("/"):
                    resolved = posixpath.normpath(target.lstrip("/"))
                else:
                    resolved = posixpath.normpath(posixpath.join(base, target))
                if resolved not in ("", ".") and resolved not in names:
                    bad.append(f"{rels} -> {target}")
    return bad


def _make_docx_with_rels() -> bytes:
    """DOCX whose document rels reference customXml, a kept part and a URL."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/customXml/item1.xml" ContentType="application/xml"/>
</Types>""",
        )
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Hello</w:t></w:r></w:p></w:body></w:document>',
        )
        zf.writestr(
            "customXml/item1.xml",
            '<?xml version="1.0"?><root>c2pa contentcredentials</root>',
        )
        zf.writestr(
            "word/_rels/document.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/customXml" Target="../customXml/item1.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.com/" TargetMode="External"/>
</Relationships>""",
        )
    return buf.getvalue()


def test_docx_dropped_customxml_prunes_dangling_relationships():
    data = _make_docx_with_rels()
    assert _dangling_rels(data) == []
    cleaned, actions = clean_docx(data)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        names = zf.namelist()
        assert not any(n.startswith("customXml/") for n in names)
        rels = zf.read("word/_rels/document.xml.rels").decode()
        assert "../customXml/item1.xml" not in rels
        assert 'Target="document.xml"' in rels
        assert 'TargetMode="External"' in rels
    assert _dangling_rels(cleaned) == []
    assert any("prune dangling relationships" in a for a in actions)


def _make_docx_with_body_text(body_text: str = "Claude wrote this.") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>""",
        )
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>'
            + body_text
            + "</w:t></w:r></w:p></w:body></w:document>",
        )
        zf.writestr(
            "docProps/core.xml",
            '<?xml version="1.0"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"></cp:coreProperties>',
        )
    return buf.getvalue()


def test_docx_body_vendor_word_is_not_ai_metadata():
    from container_meta import inspect_docx

    data = _make_docx_with_body_text()
    has_c2pa, has_ai, findings, _ = inspect_docx(data)
    assert not has_c2pa
    assert not has_ai
    assert not any("Claude" in f for f in findings)


def test_docx_metadata_vendor_word_is_still_flagged():
    from container_meta import inspect_docx

    data = _make_docx_with_app("Claude AI Writer")
    _has_c2pa, has_ai, findings, _ = inspect_docx(data)
    assert has_ai
    assert any("Claude" in f for f in findings)


def _make_docx_with_docprops() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/custom.xml" ContentType="application/vnd.openxmlformats-officedocument.custom-properties+xml"/>
</Types>""",
        )
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Hello</w:t></w:r></w:p></w:body></w:document>',
        )
        zf.writestr(
            "docProps/core.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/">
  <dc:title>My Document</dc:title>
  <dc:creator>ChatGPT</dc:creator>
  <cp:lastModifiedBy>Claude</cp:lastModifiedBy>
  <dc:description>Generated by AI</dc:description>
  <cp:keywords>ai, model</cp:keywords>
  <dc:subject>artificial intelligence</dc:subject>
  <cp:category>report</cp:category>
</cp:coreProperties>""",
        )
        zf.writestr(
            "docProps/app.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
  <Application>ChatGPT</Application>
  <AppVersion>16.0</AppVersion>
  <Company>OpenAI</Company>
  <Manager>Someone</Manager>
</Properties>""",
        )
        zf.writestr(
            "docProps/custom.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <property fmtid="{D5CDD505-2E9C-101B-9397-08002B2CF9AE}" pid="2" name="Client"><vt:lpwstr>Acme</vt:lpwstr></property>
</Properties>""",
        )
    return buf.getvalue()


def _make_docx_with_invisible_body() -> bytes:
    buf = io.BytesIO()
    document = (
        '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        "<w:p><w:r><w:t>Hello\u200b world\u2060 with\u00a0space </w:t></w:r></w:p>"
        '<w:p><w:r><w:instrText xml:space="preserve"> FIELD \u200b KEEP </w:instrText></w:r></w:p>'
        '<w:p><w:r><w:t xml:space="preserve">keep\u200bme </w:t></w:r></w:p>'
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>""",
        )
        zf.writestr("word/document.xml", document)
        zf.writestr(
            "docProps/core.xml",
            '<?xml version="1.0"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"></cp:coreProperties>',
        )
    return buf.getvalue()


def test_docx_scrubs_docprops_provenance_fields_unconditionally():
    """Verify unconditional scrubbing of docProps provenance fields in DOCX."""
    import xml.etree.ElementTree as ET

    data = _make_docx_with_docprops()
    cleaned, actions = clean_docx(data)

    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        names = zf.namelist()
        assert "docProps/core.xml" in names
        assert "docProps/app.xml" in names
        assert "docProps/custom.xml" not in names
        ct = zf.read("[Content_Types].xml").decode()
        assert 'PartName="/docProps/custom.xml"' not in ct

        core = zf.read("docProps/core.xml").decode()
        app = zf.read("docProps/app.xml").decode()

    # Every provenance field is emptied...
    for field in (
        "dc:creator",
        "cp:lastModifiedBy",
        "dc:description",
        "cp:keywords",
        "dc:subject",
        "cp:category",
    ):
        assert f"<{field}></{field}>" in core or f"<{field}/>" in core
    for field in ("Application", "Company", "Manager"):
        assert f"<{field}></{field}>" in app or f"<{field}/>" in app
    # ...while AppVersion and dc:title survive (issue #283)
    assert "<AppVersion>16.0</AppVersion>" in app
    assert "<dc:title>My Document</dc:title>" in core
    assert "ChatGPT" not in core
    assert "Generated by AI" not in core
    assert "OpenAI" not in app

    # The output docProps remain well-formed XML (trusted in-memory test data).
    ET.fromstring(core)  # noqa: S314
    ET.fromstring(app)  # noqa: S314

    assert any("scrub docProps/core.xml field dc:creator" in a for a in actions)
    assert any("drop part docProps/custom.xml" in a for a in actions)


def test_docx_docprops_scrub_clears_residual_warning(tmp_path: Path):
    src = tmp_path / "in.docx"
    src.write_bytes(_make_docx_with_docprops())
    dest = tmp_path / "out.docx"
    result = clean_container(src, dest)
    assert result["format"] == "docx"
    assert not result["still_has_ai_metadata"]
    assert not any("docProps/core.xml" in f for f in result["post_findings"])


def test_docx_layer_a_strips_invisible_body_chars():
    data = _make_docx_with_invisible_body()
    cleaned, actions = clean_docx(data)
    assert any(a.startswith("layer A text: removed=") for a in actions)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        doc = zf.read("word/document.xml").decode()
        # ZWSP / word joiner / NBSP removed from w:t runs...
        assert "Hello\u200b" not in doc
        assert "world\u2060" not in doc
        assert "with\u00a0space" not in doc
        # ...NBSP was replaced with a regular space, trailing space keeps preserve
        assert '<w:t xml:space="preserve">Hello world with space </w:t>' in doc
        # field codes are never touched
        assert " FIELD \u200b KEEP " in doc
        # existing xml:space is retained on cleaned runs
        assert '<w:t xml:space="preserve">keepme </w:t>' in doc


def test_docx_layer_a_can_be_disabled():
    data = _make_docx_with_invisible_body()
    cleaned, actions = clean_docx(data, also_layer_a_text=False)
    assert not any(a.startswith("layer A text:") for a in actions)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        assert "Hello\u200b" in zf.read("word/document.xml").decode()


def test_docx_layer_a_via_clean_container(tmp_path: Path):
    src = tmp_path / "in.docx"
    src.write_bytes(_make_docx_with_invisible_body())
    dest = tmp_path / "out.docx"
    result = clean_container(src, dest)
    assert any(a.startswith("layer A text: removed=") for a in result["actions"])
    with zipfile.ZipFile(dest) as zf:
        doc = zf.read("word/document.xml").decode()
        assert "Hello\u200b" not in doc
        assert " FIELD \u200b KEEP " in doc


def _make_odt_with_invisible_text() -> bytes:
    buf = io.BytesIO()
    content = (
        '<?xml version="1.0"?><office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
        "<office:body><office:text>"
        '<text:p text:style-name="P1">Hello\u200b <text:span>world\u2060</text:span>!</text:p>'
        "</office:text></office:body></office:document-content>"
    )
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        zf.writestr("meta.xml", '<?xml version="1.0"?><office:document-meta/>')
        zf.writestr("content.xml", content)
        zf.writestr("META-INF/manifest.xml", '<?xml version="1.0"?><manifest:manifest/>')
    return buf.getvalue()


def test_odt_layer_a_strips_invisible_text():
    data = _make_odt_with_invisible_text()
    cleaned, actions = clean_odt(data)
    assert any(a.startswith("layer A text: removed=") for a in actions)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        content = zf.read("content.xml").decode()
        assert "\u200b" not in content
        assert "\u2060" not in content
        assert "<text:span>world</text:span>" in content


def _make_odt(generator: str = "Anthropic Claude") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        zf.writestr(
            "meta.xml",
            f'<?xml version="1.0"?><office:document-meta xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0"><meta:generator>{generator}</meta:generator></office:document-meta>',
        )
        zf.writestr(
            "content.xml",
            '<?xml version="1.0"?><office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"/>',
        )
        zf.writestr(
            "META-INF/manifest.xml",
            '<?xml version="1.0"?><manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"/>',
        )
    return buf.getvalue()


def test_odt_drops_generator(tmp_path: Path):
    data = _make_odt()
    cleaned, actions = clean_odt(data)
    assert any("generator" in a for a in actions)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        meta = zf.read("meta.xml").decode()
        assert "Claude" not in meta
        assert "meta:generator" not in meta or "Anthropic" not in meta


def _make_odt_with_manifest(manifest_entries: list[str], extra_parts: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    manifest = (
        '<?xml version="1.0"?><manifest:manifest '
        'xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0">'
        + "".join(manifest_entries)
        + "</manifest:manifest>"
    )
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        zf.writestr("content.xml", b'<?xml version="1.0"?><office:document-content/>')
        zf.writestr("META-INF/manifest.xml", manifest)
        for name, payload in extra_parts.items():
            zf.writestr(name, payload)
    return buf.getvalue()


def test_odt_dropped_part_removes_manifest_entry():
    """Dropping a marker-bearing part must also drop its manifest entry,
    or readers flag the package as damaged."""
    data = _make_odt_with_manifest(
        [
            '<manifest:file-entry manifest:media-type="application/vnd.oasis.opendocument.text" manifest:full-path="/"/>',
            '<manifest:file-entry manifest:media-type="text/xml" manifest:full-path="content.xml"/>',
            '<manifest:file-entry manifest:media-type="application/octet-stream" manifest:full-path="custommeta.xml"/>',
        ],
        extra_parts={"custommeta.xml": b"<meta><creator>Anthropic Claude</creator></meta>"},
    )
    cleaned, actions = clean_odt(data)
    assert any("drop part custommeta.xml" in a for a in actions)
    assert any("drop manifest entries" in a for a in actions)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        assert zf.namelist().count("META-INF/manifest.xml") == 1
        manifest = zf.read("META-INF/manifest.xml").decode()
        assert "custommeta.xml" not in manifest
        assert 'full-path="content.xml"' in manifest
        assert 'full-path="/"' in manifest


def test_odt_manifest_rewrite_is_attribute_order_independent():
    """The dropped entry has full-path before media-type; matching must not
    depend on attribute order."""
    data = _make_odt_with_manifest(
        [
            '<manifest:file-entry manifest:full-path="/"/>',
            '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>',
            '<manifest:file-entry manifest:full-path="custommeta.xml" manifest:media-type="application/octet-stream"/>',
        ],
        extra_parts={"custommeta.xml": b"<meta><creator>Anthropic Claude</creator></meta>"},
    )
    cleaned, _actions = clean_odt(data)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        manifest = zf.read("META-INF/manifest.xml").decode()
        assert "custommeta.xml" not in manifest
        assert 'full-path="content.xml"' in manifest


def test_odt_manifest_untouched_when_nothing_dropped():
    data = _make_odt_with_manifest(
        [
            '<manifest:file-entry manifest:media-type="application/vnd.oasis.opendocument.text" manifest:full-path="/"/>',
            '<manifest:file-entry manifest:media-type="text/xml" manifest:full-path="content.xml"/>',
        ],
        extra_parts={},
    )
    cleaned, actions = clean_odt(data)
    assert not any("manifest" in a for a in actions)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        manifest = zf.read("META-INF/manifest.xml").decode()
        assert 'full-path="content.xml"' in manifest


def test_clean_container_markdown_file(tmp_path: Path):
    src = tmp_path / "x.md"
    src.write_text("---\ngenerator: OpenAI\n---\nHi\u200b\n", encoding="utf-8")
    dest = tmp_path / "x.cleaned.md"
    result = clean_container(src, dest)
    assert dest.is_file()
    body = dest.read_text(encoding="utf-8")
    assert "generator" not in body
    assert "\u200b" not in body
    assert result["format"] == "markdown"


def test_inspect_container_reports_layer_a_body_text(tmp_path: Path):
    """inspect must flag invisible carriers that clean would strip.

    Regression: markdown/html routed to the container inspector, which never
    ran the Layer A scan, so identical bytes were reported suspicious as .txt
    and clean as .md while clean_container() went on to remove them.
    """
    body = "Hello​world‌ test⁠end.\n"
    for name in ("x.md", "x.html"):
        src = tmp_path / name
        src.write_text(body, encoding="utf-8")
        report = inspect_container(src)
        assert report.layer_a_total == 3, name
        assert report.to_dict()["suspicious_total"] == 3, name
        codepoints = {h["codepoint"] for h in report.layer_a_hits}
        assert {"U+200B", "U+200C", "U+2060"} <= codepoints, name
        assert any(f.startswith("layer-a:") for f in report.findings), name


def test_inspect_container_clean_leaves_no_layer_a(tmp_path: Path):
    """The post-clean re-inspect must come back with nothing left."""
    src = tmp_path / "x.md"
    src.write_text("Hi​‌there\n", encoding="utf-8")
    dest = tmp_path / "x.cleaned.md"
    clean_container(src, dest)
    assert inspect_container(dest).layer_a_total == 0


def test_inspect_container_clean_file_stays_clean(tmp_path: Path):
    """No false positives on ordinary prose."""
    src = tmp_path / "x.md"
    src.write_text("# Title\n\nOrdinary prose, nothing hidden.\n", encoding="utf-8")
    report = inspect_container(src)
    assert report.layer_a_total == 0
    assert report.layer_a_hits == []


def test_inspect_container_svg(tmp_path: Path):
    src = tmp_path / "a.svg"
    src.write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg"><metadata>c2pa</metadata></svg>')
    report = inspect_container(src)
    assert report.format == "svg"
    assert report.has_c2pa or report.has_ai_metadata


def test_fixtures_md_html_svg_roundtrip(tmp_path: Path):
    root = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    for name in ("sample_ai.md", "sample_ai.html", "sample_meta.svg"):
        src = root / name
        dest = tmp_path / f"{name}.cleaned{src.suffix}"
        result = clean_container(src, dest)
        assert dest.is_file()
        assert result["format"] in ("markdown", "html", "svg")
        # AI-ish keys/tags should be reduced
        body = dest.read_bytes().lower()
        assert b"chatgpt" not in body
        assert b"generator: claude" not in body


def test_pdf_degraded_clean_without_crash(tmp_path: Path):
    """Minimal PDF with an XMP packet; clean should not raise (may be degraded)."""
    from container_meta import clean_pdf, inspect_pdf

    xmp = (
        b"<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>"
        b"<x:xmpmeta xmlns:x='adobe:ns:meta/'>"
        b"<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>"
        b"<rdf:Description>"
        b"<digitalSourceType>trainedAlgorithmicMedia</digitalSourceType>"
        b"</rdf:Description></rdf:RDF></x:xmpmeta>"
        b"<?xpacket end='w'?>"
    )
    # Minimal-ish PDF skeleton (not renderable; enough for byte-level tools)
    pdf = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n" + xmp + b"\n%%EOF\n"
    src = tmp_path / "t.pdf"
    dest = tmp_path / "t.cleaned.pdf"
    src.write_bytes(pdf)
    has_c2pa, has_ai, findings, _ = inspect_pdf(src, pdf)
    assert has_ai or has_c2pa or findings
    actions, meta = clean_pdf(src, dest)
    assert dest.is_file()
    assert actions
    assert meta.get("mode") in ("exiftool", "stdlib-xmp", "copy")


def test_clean_container_accepts_explicit_fmt(tmp_path: Path):
    """A .bak source with no detectable magic bytes still cleans when fmt is pinned."""
    src = tmp_path / "backup.md.bak"
    src.write_text("---\ngenerator: OpenAI\n---\nHi\u200b\n", encoding="utf-8")
    dest = tmp_path / "out.md"
    result = clean_container(src, dest, fmt="markdown")
    assert result["format"] == "markdown"
    body = dest.read_text(encoding="utf-8")
    assert "generator" not in body
    assert "\u200b" not in body


@pytest.mark.parametrize(
    "ext,make_bytes",
    [
        (
            "md",
            (Path(__file__).resolve().parent / "fixtures" / "sample_ai.md").read_bytes,
        ),
        (
            "html",
            (Path(__file__).resolve().parent / "fixtures" / "sample_ai.html").read_bytes,
        ),
        (
            "svg",
            (Path(__file__).resolve().parent / "fixtures" / "sample_meta.svg").read_bytes,
        ),
        ("docx", _make_docx_with_app),
        ("odt", _make_odt),
    ],
)
def test_clean_file_in_place_for_every_container_ext(tmp_path: Path, ext: str, make_bytes):
    path = tmp_path / f"a.{ext}"
    path.write_bytes(make_bytes())
    r = _run("clean_file.py", str(path), "--in-place", "--json")
    assert r.returncode == 0, r.stderr
    assert path.with_suffix(path.suffix + ".bak").is_file()
    data = json.loads(r.stdout)
    assert data["kind"] == "container"
    assert (
        data["format"]
        == {"docx": "docx", "odt": "odt", "svg": "svg", "md": "markdown", "html": "html"}[ext]
    )


def test_container_inspectors_survive_malformed_zip():
    # A truncated or garbage container degrades to a clear finding instead of
    # raising, for docx, odt, and format detection.
    truncated = bytes([0x50, 0x4B, 0x03, 0x04]) + bytes(8)
    garbage = b"not a zip at all"
    for data in (truncated, garbage):
        assert inspect_docx(data) == (False, False, ["not a valid DOCX zip"], {})
        assert inspect_odt(data) == (False, False, ["not a valid ODT zip"], {})
        assert detect_container_format(Path("x.bin"), data) == "unknown"


def test_zip_budget_rejection_propagates_from_inspect(monkeypatch):
    # A refused zip bomb must propagate the same way it does out of clean_*,
    # not be reported as an unparseable container.
    import container_meta

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "<w:document/>")
    monkeypatch.setattr(container_meta, "MAX_ZIP_DECOMPRESSED_BYTES", 1)
    with pytest.raises(container_meta.ZipBudgetExceeded):
        inspect_docx(buf.getvalue())


def test_zip_budget_rejection_propagates_from_detect_container_format_mimetype(monkeypatch):
    import container_meta

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
    monkeypatch.setattr(container_meta, "MAX_ZIP_DECOMPRESSED_BYTES", 5)
    with pytest.raises(container_meta.ZipBudgetExceeded):
        detect_container_format(Path("x.bin"), buf.getvalue())


def test_docx_clean_preserves_ooxml_appversion_schema_validity():
    """clean_docx preserves AppVersion while scrubbing application provenance (#283)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>""",
        )
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Hello world</w:t></w:r></w:p></w:body></w:document>',
        )
        zf.writestr(
            "docProps/core.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:creator>Test Author</dc:creator>
  <dc:description>Generated Document</dc:description>
</cp:coreProperties>""",
        )
        zf.writestr(
            "docProps/app.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
  <Application>Microsoft Macintosh Word</Application>
  <AppVersion>16.0000</AppVersion>
  <Company>Test Corp</Company>
  <Manager>Test Manager</Manager>
</Properties>""",
        )
    data = buf.getvalue()
    cleaned, actions = clean_docx(data)
    assert any("scrub docProps/core.xml field dc:creator" in a for a in actions)
    assert any("scrub docProps/app.xml field Application" in a for a in actions)
    assert any("scrub docProps/app.xml field Company" in a for a in actions)
    assert any("scrub docProps/app.xml field Manager" in a for a in actions)
    assert not any("AppVersion" in a for a in actions)

    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        app = zf.read("docProps/app.xml").decode("utf-8")
        assert "<Application></Application>" in app or "<Application/>" in app
        assert "<AppVersion>16.0000</AppVersion>" in app
        assert "<Company></Company>" in app or "<Company/>" in app
        assert "<Manager></Manager>" in app or "<Manager/>" in app
