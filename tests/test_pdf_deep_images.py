"""Deep-image pass: metadata that lives inside an embedded image.

`exiftool -all=` and the qpdf rewrite only reach document-level metadata, so
EXIF or a C2PA manifest carried by an image XObject survives both. clean_pdf
re-distills through Ghostscript to reach it, losslessly first and recompressing
only when a marker demonstrably rode through the passed-through stream.
"""

from __future__ import annotations

import base64
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service" / "scripts"))

from common import which
from container_meta import (
    clean_pdf,
    embedded_image_metadata_present,
    which_ghostscript,
)

needs_exiftool = pytest.mark.skipif(not which("exiftool"), reason="exiftool not installed")
needs_ghostscript = pytest.mark.skipif(not which_ghostscript(), reason="ghostscript not installed")


def _assemble_pdf(objs: list[bytes], trailer_extra: bytes = b"") -> bytes:
    """Wrap numbered objects in a header, xref table and trailer.

    The xref offsets are byte-exact, so this lives in one place: a correction
    applied to only one copy would leave the other quietly malformed.
    """
    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R%s >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1,
        trailer_extra,
        xref,
    )
    return bytes(out)


def _minimal_pdf(info: bytes = b"") -> bytes:
    """A one-page PDF, optionally carrying an Info dictionary."""
    content = b"BT /F1 12 Tf 72 720 Td (deep image test) Tj ET"
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    trailer_extra = b""
    if info:
        objs.append(info)
        trailer_extra = b" /Info %d 0 R" % len(objs)
    return _assemble_pdf(objs, trailer_extra)


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    i = 2
    while i < len(data) - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        # Fill bytes may repeat before a marker -- the same detail the walker in
        # container_meta has to honour.
        while i + 1 < len(data) and data[i + 1] == 0xFF:
            i += 1
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            height, width = struct.unpack(">HH", data[i + 5 : i + 9])
            return width, height
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        i += 2 + struct.unpack(">H", data[i + 2 : i + 4])[0]
    raise AssertionError("no JPEG SOF marker")


def _pdf_with_image(jpeg: bytes) -> bytes:
    width, height = _jpeg_dimensions(jpeg)
    content = b"q %d 0 0 %d 0 0 cm /Im0 Do Q" % (width, height)
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] "
        b"/Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>" % (width, height),
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB "
        b"/BitsPerComponent 8 /Filter /DCTDecode /Length %d >>\nstream\n"
        % (width, height, len(jpeg))
        + jpeg
        + b"\nendstream",
    ]
    return _assemble_pdf(objs)


# A C2PA-signed image carries the manifest in APP11 (JUMBF) and, alongside it,
# an XMP packet naming the provenance. The byte scan skips PDF stream payloads,
# so the XMP packet is what actually raises the flag on an embedded image --
# reproduce both or the fixture will not behave like a real signed file.
_PROVENANCE_XMP = (
    b'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
    b'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b'<rdf:Description rdf:about="" xmlns:dcterms="http://purl.org/dc/terms/">'
    b"<dcterms:provenance>self#jumbf=c2pa/contentauth</dcterms:provenance>"
    b'</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
)


# A 16x16 solid-colour JPEG, inlined rather than generated: another module in
# this suite installs a PIL stub into sys.modules, and importing PIL here picks
# up whichever version ran first.
_TINY_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYI"
    "DAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkF"
    "BQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQU"
    "FBQUFBT/wAARCAAQABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQF"
    "BgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEI"
    "I0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNk"
    "ZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLD"
    "xMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEB"
    "AQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJB"
    "UQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZH"
    "SElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaan"
    "qKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oA"
    "DAMBAAIRAxEAPwDxSiiivzc/tU//2Q=="
)


def _exif_only_jpeg() -> bytes:
    """A JPEG carrying camera-style EXIF and nothing an AI scan would flag."""
    exif = b"Exif\x00\x00MM\x00*\x00\x00\x00\x08 Camera Model X"
    app1 = b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
    return _TINY_JPEG[:2] + app1 + _TINY_JPEG[2:]


def _c2pa_like_jpeg() -> bytes:
    """A tiny JPEG marked the way a C2PA-signed one is: APP11 + XMP in APP1."""
    jumbf = b"JP\x00\x00jumbc2pa contentauth manifest"
    app11 = b"\xff\xeb" + struct.pack(">H", len(jumbf) + 2) + jumbf
    xmp_payload = b"http://ns.adobe.com/xap/1.0/\x00" + _PROVENANCE_XMP
    app1 = b"\xff\xe1" + struct.pack(">H", len(xmp_payload) + 2) + xmp_payload
    # Both segments sit right after SOI, so they travel with the image bytes.
    return _TINY_JPEG[:2] + app11 + app1 + _TINY_JPEG[2:]


@needs_exiftool
def test_never_skips_the_deep_pass(tmp_path):
    src = tmp_path / "in.pdf"
    src.write_bytes(_minimal_pdf(b"<< /Producer (Claude) >>"))
    actions, meta = clean_pdf(src, tmp_path / "out.pdf", deep_images="never")

    assert meta["deep_images"] == "never"
    assert meta["deep_image_pass"] is False
    assert not any("ghostscript" in a for a in actions)


@needs_exiftool
def test_auto_leaves_a_marker_free_pdf_alone(tmp_path):
    """No AI/C2PA markers left means no reason to spend a re-distill."""
    src = tmp_path / "in.pdf"
    src.write_bytes(_minimal_pdf(b"<< /Producer (Claude) >>"))
    actions, meta = clean_pdf(src, tmp_path / "out.pdf", deep_images="auto")

    assert meta["deep_image_pass"] is False
    assert meta["images_reencoded"] is False
    assert any("deep image pass not needed" in a for a in actions)


@needs_exiftool
@needs_ghostscript
def test_always_runs_the_lossless_pass(tmp_path):
    src = tmp_path / "in.pdf"
    src.write_bytes(_minimal_pdf(b"<< /Producer (Claude) >>"))
    dest = tmp_path / "out.pdf"
    actions, meta = clean_pdf(src, dest, deep_images="always")

    assert meta["deep_image_pass"] is True
    assert meta["images_reencoded"] is False, "nothing survived, so nothing to recompress"
    assert any("passed through" in a for a in actions)
    assert dest.read_bytes().startswith(b"%PDF")


@needs_exiftool
@needs_ghostscript
def test_marker_inside_the_image_is_removed(tmp_path):
    """A mark the document-level strip cannot reach must not survive the clean.

    Which rung of the ladder clears it depends on the file: pdfwrite usually
    rebuilds the JPEG container and drops APPn on the way, but it copies some
    streams verbatim, and then only a re-encode shifts the marker. Assert the
    outcome, not the rung.
    """
    src = tmp_path / "in.pdf"
    src.write_bytes(_pdf_with_image(_c2pa_like_jpeg()))
    dest = tmp_path / "out.pdf"

    _actions, meta = clean_pdf(src, dest, deep_images="auto")

    assert meta["deep_image_pass"] is True
    assert b"contentauth" not in dest.read_bytes()


@needs_exiftool
def test_auto_escalates_when_the_lossless_pass_leaves_the_marker(tmp_path, monkeypatch):
    """The ladder's second rung, with the pass stubbed so nothing is removed."""
    import container_meta

    calls: list[bool] = []

    def _stub(dest, actions, *, reencode=False, deadline=None):
        calls.append(reencode)
        actions.append(f"stub deep pass (reencode={reencode})")
        return True

    monkeypatch.setattr(container_meta, "_pdf_deep_image_clean", _stub)

    src = tmp_path / "in.pdf"
    src.write_bytes(_pdf_with_image(_c2pa_like_jpeg()))
    actions, meta = clean_pdf(src, tmp_path / "out.pdf", deep_images="auto")

    assert calls == [False, True], "lossless first, then the re-encoding escalation"
    assert meta["images_reencoded"] is True
    assert any("escalating to a re-encoding pass" in a for a in actions)


@needs_exiftool
def test_lossless_refuses_to_escalate(tmp_path, monkeypatch):
    """Same stub, but `lossless` must stop after the first rung."""
    import container_meta

    calls: list[bool] = []

    def _stub(dest, actions, *, reencode=False, deadline=None):
        calls.append(reencode)
        return True

    monkeypatch.setattr(container_meta, "_pdf_deep_image_clean", _stub)

    src = tmp_path / "in.pdf"
    src.write_bytes(_pdf_with_image(_c2pa_like_jpeg()))
    _actions, meta = clean_pdf(src, tmp_path / "out.pdf", deep_images="lossless")

    assert calls == [False]
    assert meta["images_reencoded"] is False


@needs_exiftool
def test_rejects_an_unknown_deep_images_value(tmp_path):
    """A typo must not silently downgrade to a mode that may recompress."""
    src = tmp_path / "in.pdf"
    src.write_bytes(_minimal_pdf(b"<< /Producer (Claude) >>"))

    with pytest.raises(ValueError, match="deep_images"):
        clean_pdf(src, tmp_path / "out.pdf", deep_images="lossles")


def test_metadata_detector_ignores_structural_segments():
    """APP0 (JFIF) is structural and APP2 (ICC) carries colour, not provenance."""
    jfif_and_icc = (
        _TINY_JPEG[:2]
        + b"\xff\xe0"
        + struct.pack(">H", 7)
        + b"JFIF\x00"
        + b"\xff\xe2"
        + struct.pack(">H", 14)
        + b"ICC_PROFILE\x00"
        + _TINY_JPEG[2:]
    )
    assert embedded_image_metadata_present(jfif_and_icc) is False
    assert embedded_image_metadata_present(_exif_only_jpeg()) is True


@needs_exiftool
@needs_ghostscript
def test_always_reaches_exif_inside_the_image(tmp_path):
    """`always` promises camera and editor traces go too, so they must go.

    Which rung achieves it depends on the file: pdfwrite usually rebuilds the
    JPEG container and drops APPn while passing the compressed data through,
    but it copies some streams verbatim, and then only a re-encode shifts them.
    Assert the guarantee, not the rung.
    """
    src = tmp_path / "in.pdf"
    src.write_bytes(_pdf_with_image(_exif_only_jpeg()))
    dest = tmp_path / "out.pdf"
    assert embedded_image_metadata_present(src.read_bytes()) is True

    _actions, meta = clean_pdf(src, dest, deep_images="always")

    assert meta["deep_image_pass"] is True
    assert embedded_image_metadata_present(dest.read_bytes()) is False


@needs_exiftool
def test_auto_leaves_plain_exif_and_the_pixels_alone(tmp_path):
    """Ordinary EXIF is not an AI mark, so `auto` has no reason to chase it."""
    src = tmp_path / "in.pdf"
    src.write_bytes(_pdf_with_image(_exif_only_jpeg()))
    dest = tmp_path / "out.pdf"

    actions, meta = clean_pdf(src, dest, deep_images="auto")

    assert meta["deep_image_pass"] is False
    assert meta["images_reencoded"] is False
    assert any("deep image pass not needed" in a for a in actions)


@needs_exiftool
def test_always_escalates_when_the_lossless_pass_keeps_exif(tmp_path, monkeypatch):
    """The safety net, with the pass stubbed so the EXIF cannot be removed."""
    import container_meta

    calls: list[bool] = []

    def _stub(dest, actions, *, reencode=False, deadline=None):
        calls.append(reencode)
        return True

    monkeypatch.setattr(container_meta, "_pdf_deep_image_clean", _stub)

    src = tmp_path / "in.pdf"
    src.write_bytes(_pdf_with_image(_exif_only_jpeg()))
    _actions, meta = clean_pdf(src, tmp_path / "out.pdf", deep_images="always")

    assert calls == [False, True], "lossless first, then the re-encoding escalation"
    assert meta["images_reencoded"] is True


@needs_exiftool
def test_no_escalation_when_the_deep_pass_cannot_run(tmp_path, monkeypatch):
    """A missing Ghostscript must not be mistaken for a surviving marker.

    Rung 1 returning False means nothing was attempted, so rung 2 cannot help
    either -- asking would only spend an inspect_pdf and append the same
    "install ghostscript" warning a second time.
    """
    import container_meta

    calls: list[bool] = []

    def _unavailable(dest, actions, *, reencode=False, deadline=None):
        calls.append(reencode)
        actions.append(
            "warning: metadata inside embedded images left in place; "
            "install ghostscript for the deep image pass"
        )
        return False

    monkeypatch.setattr(container_meta, "_pdf_deep_image_clean", _unavailable)

    src = tmp_path / "in.pdf"
    src.write_bytes(_pdf_with_image(_c2pa_like_jpeg()))
    actions, meta = clean_pdf(src, tmp_path / "out.pdf", deep_images="always")

    assert calls == [False], "rung 2 must not be attempted when rung 1 never ran"
    assert meta["images_reencoded"] is False
    assert not any("escalating" in a for a in actions)
    assert sum("install ghostscript" in a for a in actions) == 1


@needs_ghostscript
def test_the_deep_pass_runs_without_exiftool(tmp_path, monkeypatch):
    """Ghostscript reaches image XObjects on its own; exiftool is not its gate.

    The ladder used to live inside `if exiftool:`, so a machine with Ghostscript
    but no exiftool fell through to the stdlib path and never touched metadata
    inside images -- the one job exiftool cannot do anyway.
    """
    import container_meta

    real_which = container_meta.which
    monkeypatch.setattr(
        container_meta,
        "which",
        lambda name: None if name == "exiftool" else real_which(name),
    )

    src = tmp_path / "in.pdf"
    src.write_bytes(_pdf_with_image(_c2pa_like_jpeg()))
    actions, meta = clean_pdf(src, tmp_path / "out.pdf", deep_images="always")

    assert meta["deep_image_pass"] is True
    # Running is not the point; the fixture provenance being gone is.
    assert b"contentauth" not in (tmp_path / "out.pdf").read_bytes()
    assert meta["degraded"] is True, "the document-level strip was still the stdlib one"
    assert meta["mode"] in ("stdlib-xmp", "copy")
    # The swap of one provenance stamp for another must not be silent.
    assert any("stamped its own /Producer" in a for a in actions)


@needs_ghostscript
def test_auto_sees_provenance_that_only_lives_in_a_stream(tmp_path, monkeypatch):
    """`auto` must not need c2patool to notice a manifest inside an image.

    inspect_pdf drops stream payloads before its marker scan, so with c2patool
    absent the only signal left is the stream itself -- and `auto` decides
    whether to re-distill from exactly that scan.
    """
    import container_meta

    real_which = container_meta.which
    monkeypatch.setattr(
        container_meta,
        "which",
        lambda name: None if name == "c2patool" else real_which(name),
    )

    src = tmp_path / "in.pdf"
    src.write_bytes(_pdf_with_image(_c2pa_like_jpeg()))
    dest = tmp_path / "out.pdf"
    _actions, meta = clean_pdf(src, dest, deep_images="auto")

    assert meta["deep_image_pass"] is True
    assert b"contentauth" not in dest.read_bytes()


def test_provenance_detector_ignores_ordinary_exif():
    """`auto` promises not to spend a re-distill on camera traces."""
    import struct

    import container_meta

    exif = b"Exif\x00\x00MM\x00*" + b"\x00" * 32
    app1 = b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
    jpeg = _TINY_JPEG[:2] + app1 + _TINY_JPEG[2:]
    pdf = _pdf_with_image(jpeg)

    assert container_meta.embedded_image_metadata_present(pdf) is True
    assert container_meta.embedded_provenance_present(pdf) is False


def test_blanked_xmp_keeps_the_pdf_parseable(tmp_path, monkeypatch):
    """The stdlib path must not shift byte offsets out from under the xref."""
    import container_meta

    real_which = container_meta.which
    monkeypatch.setattr(
        container_meta,
        "which",
        lambda name: None if name == "exiftool" else real_which(name),
    )

    xmp = (
        b'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"/><?xpacket end="w"?>'
    )
    src = tmp_path / "in.pdf"
    src.write_bytes(
        _minimal_pdf(b"<< /Producer (Claude) >>").replace(
            b"deep image test", b"deep image test" + b" " * len(xmp)
        )
    )
    # Put the packet in a place the xref does not describe, so only offsets
    # after it would break: enough to catch a length-changing edit.
    src.write_bytes(src.read_bytes().replace(b" " * len(xmp), xmp, 1))
    dest = tmp_path / "out.pdf"

    _actions, _meta = clean_pdf(src, dest, deep_images="never")

    cleaned = dest.read_bytes()
    assert len(cleaned) == src.stat().st_size, "blanking must not resize the file"
    assert b"<?xpacket" not in cleaned


def test_provenance_survives_marker_fill_bytes():
    """A marker may be preceded by 0xFF padding; the walker must skip it.

    Reading the first fill byte as the marker shifts the length by one, and the
    walk then misses everything after it -- including the APP11 that decides
    whether `auto` runs the deep pass at all.
    """
    import container_meta

    padded = _c2pa_like_jpeg().replace(b"\xff\xeb", b"\xff\xff\xff\xeb", 1)
    assert container_meta.embedded_provenance_present(_pdf_with_image(padded)) is True
