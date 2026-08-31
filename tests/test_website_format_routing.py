"""audit_website routes binary formats to their real scanners (#166).

webp/avif/heic/gif/bmp/tiff/xlsx/pptx/epub/mp4/wav/mp3 arrived over HTTP as
"text" — the bytes fell to the Unicode scanner and C2PA-carrying assets
reported clean with no failure signal.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_website


def _webp_with_c2pa() -> bytes:
    payload = b"c2pa contentcredentials"
    # WebP chunks are padded to even length; 23 is odd so append one pad byte.
    chunk = b"c2pa" + struct.pack("<I", len(payload)) + payload + bytes([0])
    riff = b"WEBP" + chunk
    return b"RIFF" + struct.pack("<I", len(riff)) + riff


def _mp4_with_c2pa() -> bytes:
    payload = b"c2pa contentcredentials"
    box = b"jumb" + struct.pack(">I", 8 + len(payload)) + payload
    return b"\x00\x00\x00\x14ftypisom" + b"\x00\x00\x00\x00" + box


def test_webp_classified_by_content_type_suffix_and_magic():
    data = _webp_with_c2pa()
    assert audit_website.guess_kind("/img.webp", data, "image/webp") == "webp"
    assert audit_website.guess_kind("/img.webp", data, None) == "webp"
    assert audit_website.guess_kind("/asset", data, None) == "webp"


def test_mp4_classified_by_content_type_and_magic():
    data = _mp4_with_c2pa()
    assert audit_website.guess_kind("/v.mp4", data, "video/mp4") == "mp4"
    assert audit_website.guess_kind("/asset", data, None) == "mp4"


def test_extended_suffix_table():
    # Every handled format's suffix classifies to its kind, not "text".
    for suffix, kind in [
        (".avif", "avif"),
        (".heic", "heic"),
        (".gif", "gif"),
        (".bmp", "bmp"),
        (".tiff", "tiff"),
        (".xlsx", "xlsx"),
        (".pptx", "pptx"),
        (".epub", "epub"),
        (".wav", "wav"),
        (".mp3", "mp3"),
    ]:
        assert audit_website.guess_kind(f"/a{suffix}", b"", None) == kind, suffix


def test_webp_marker_reaches_image_scanner_not_text():
    # The distinguishing assertion of #166: a C2PA-carrying WebP that the
    # image inspector sees must no longer scan as a clean "text" page.
    import image_meta

    data = _webp_with_c2pa()
    has_c2pa, _has_ai, _ = image_meta.inspect_webp(data)
    assert has_c2pa is True
    # And the site classifier routes it to that scanner's kind.
    assert audit_website.guess_kind("/img.webp", data, "image/webp") == "webp"


def test_html_still_text_friendly():
    assert audit_website.guess_kind("/x", b"<html><body>hi</body></html>", "text/html") == "html"
    assert audit_website.guess_kind("/page.htm", b"<html>", None) == "html"


def test_extensionless_zip_classified_by_container_magic():
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
    data = buf.getvalue()
    # URL has no extension, must route to epub based on container magic/mimetype
    assert audit_website.guess_kind("https://example.com/download/asset", data, None) == "epub"
