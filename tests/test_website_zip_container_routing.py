import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_website


def _make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_extensionless_docx_classified_as_docx():
    docx_data = _make_zip(
        {
            "word/document.xml": b"<w:document></w:document>",
            "[Content_Types].xml": b"<Types></Types>",
        }
    )
    assert audit_website.guess_kind("https://example.com/download/12345", docx_data, None) == "docx"
    assert (
        audit_website.guess_kind(
            "https://example.com/download/12345", docx_data, "application/octet-stream"
        )
        == "docx"
    )


def test_extensionless_xlsx_classified_as_xlsx():
    xlsx_data = _make_zip(
        {"xl/workbook.xml": b"<workbook></workbook>", "[Content_Types].xml": b"<Types></Types>"}
    )
    assert audit_website.guess_kind("https://example.com/file", xlsx_data, None) == "xlsx"


def test_extensionless_pptx_classified_as_pptx():
    pptx_data = _make_zip(
        {
            "ppt/presentation.xml": b"<presentation></presentation>",
            "[Content_Types].xml": b"<Types></Types>",
        }
    )
    assert audit_website.guess_kind("https://example.com/file", pptx_data, None) == "pptx"


def test_extensionless_odt_classified_as_odt():
    odt_data = _make_zip(
        {
            "content.xml": b"<office:document-content></office:document-content>",
            "meta.xml": b"<office:document-meta></office:document-meta>",
        }
    )
    assert audit_website.guess_kind("https://example.com/file", odt_data, None) == "odt"


def test_extensionless_epub_classified_as_epub():
    epub_data = _make_zip(
        {
            "META-INF/container.xml": b"<container></container>",
            "content.opf": b"<package></package>",
        }
    )
    assert audit_website.guess_kind("https://example.com/file", epub_data, None) == "epub"


def test_ambiguous_zip_with_lone_manifest_not_misrouted():
    zip_data = _make_zip({"META-INF/manifest.xml": b"<manifest></manifest>"})
    assert audit_website.guess_kind("https://example.com/file", zip_data, None) not in (
        "odt",
        "epub",
    )


def test_ambiguous_zip_with_lone_opf_not_misrouted():
    zip_data = _make_zip({"package.opf": b"<package></package>"})
    assert audit_website.guess_kind("https://example.com/file", zip_data, None) not in (
        "odt",
        "epub",
    )
