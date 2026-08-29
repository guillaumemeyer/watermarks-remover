"""Route a file or byte stream to the text, image or container pipeline.

The routers (inspect_file, clean_file) and the audits (audit_lib) all need the
same answer: given a path or bytes, which pipeline owns it? That decision used
to live in three copies with subtly different extension tables and sniffing.
This module is the single interface for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from av_meta import AV_EXTS, detect_av_format
from container_meta import detect_container_format
from image_meta import detect_format as detect_image_format

Kind = Literal["text", "image", "container", "av", "unknown"]

#: Bytes read for header-only sniffing. Every supported image/container
#: magic lives in the prefix; zip-based containers (docx/odt/...) need the
#: full central directory, which sits at the end of the archive, so only a
#: PK header triggers a whole-file read.
CLASSIFY_HEADER_BYTES = 4096

IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".avif",
    ".heic",
    ".heif",
    ".bmp",
    ".gif",
    ".tiff",
    ".tif",
}
CONTAINER_EXTS = {
    ".svg",
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".odt",
    ".epub",
    ".html",
    ".htm",
    ".md",
    ".markdown",
    ".mdx",
}
TEXT_EXTS = {
    ".txt",
    ".text",
    ".css",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".gd",
    ".gdshader",
    ".py",
    ".rs",
    ".go",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
}


def classify_bytes(data: bytes, suffix: str | None = None) -> Kind:
    """Classify *data* by extension first, then by magic bytes.

    The extension wins when it names a known format; otherwise the bytes are
    sniffed for image/container signatures. Unrecognized bytes classify as
    "unknown" — callers that must not mangle unknown binaries refuse unless
    the user explicitly forces a kind (--as / --force-text).

    *data* must cover the whole file: zip-based containers (docx/odt) are
    detected from their central directory, which sits at the end of the bytes.
    """
    ext = (suffix or "").lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in CONTAINER_EXTS:
        return "container"
    if ext in TEXT_EXTS:
        return "text"
    if ext in AV_EXTS:
        return "av"
    if detect_image_format(data) in ("png", "jpeg", "webp", "avif", "heic", "bmp", "gif", "tiff"):
        return "image"
    if detect_av_format(data) != "unknown":
        return "av"
    if data:
        sniff_path = Path("input") if not ext else Path(f"input{ext}")
        if detect_container_format(sniff_path, data) != "unknown":
            return "container"
    return "unknown"


def classify(path: Path) -> Kind:
    """Classify a file on disk by extension, then by its bytes.

    Known extensions are routed without reading the file. For unknown
    extensions a 4096-byte header is sniffed once; only when the header is a
    zip local header (PK) is the whole file read, because the container
    signature (docx/xlsx/pptx/odt/epub) lives in the central directory at
    the end of the archive.
    """
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in CONTAINER_EXTS:
        return "container"
    if ext in TEXT_EXTS:
        return "text"
    if ext in AV_EXTS:
        return "av"
    with path.open("rb") as fh:
        head = fh.read(CLASSIFY_HEADER_BYTES)
    if detect_image_format(head) in ("png", "jpeg", "webp", "avif", "heic", "bmp", "gif", "tiff"):
        return "image"
    if detect_av_format(head) != "unknown":
        return "av"
    if head:
        data = path.read_bytes() if head[:4] == b"PK\x03\x04" else head
        sniff_path = Path("input") if not ext else Path(f"input{ext}")
        if detect_container_format(sniff_path, data) != "unknown":
            return "container"
    return "unknown"
