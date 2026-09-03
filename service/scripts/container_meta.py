"""Inspect/clean AI provenance metadata in non-raster containers.

Formats: SVG, PDF (best-effort), DOCX, ODT, HTML, Markdown frontmatter, EPUB.
Stdlib-first; PDF prefers optional exiftool/c2patool when present.
"""

import base64
import io
import os
import posixpath
import re
import subprocess
import tempfile
import time
import urllib.parse
import zipfile
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common import (
    c2patool_probe_note,
    classify_finding_confidence,
    safe_arg,
    safe_write_bytes,
    safe_write_text,
    subprocess_creationflags,
    subprocess_preexec_fn,
    which,
)
from image_meta import (
    AI_META_HINTS,
    C2PA_MARKERS,
    inspect_bmp,
    inspect_gif,
    inspect_isobmff,
    inspect_jpeg,
    inspect_png,
    inspect_tiff,
    inspect_webp,
    run_optional_tools,
    strip_bmp,
    strip_gif,
    strip_isobmff,
    strip_jpeg,
    strip_png,
    strip_tiff,
    strip_webp,
)
from image_meta import (
    detect_format as detect_image_format,
)


class ZipBudgetExceeded(Exception):
    """A zip's declared decompressed size exceeds the processing cap.

    Kept separate from the parse errors below so a refused zip bomb keeps
    propagating (as it already does out of the clean_* helpers) instead of
    being reported as an unparseable container.
    """


# A corrupt, truncated, encrypted, or unsupported-compression zip surfaces as
# more than just BadZipFile: reading a member can raise NotImplementedError
# (unknown compression/version), RuntimeError (encrypted), zlib.error (bad
# deflate stream), EOFError (truncated), OSError (invalid stream), or
# ValueError (e.g. a negative seek).
_ZIP_PARSE_ERRORS = (
    zipfile.BadZipFile,
    zipfile.LargeZipFile,
    NotImplementedError,
    RuntimeError,
    EOFError,
    OSError,
    ValueError,
    zlib.error,
)

MAX_ZIP_DECOMPRESSED_BYTES = 128 * 1024 * 1024


def _check_zip_budget(info: zipfile.ZipInfo, budget: list[int]) -> None:
    """Fast-path zip-bomb guard on the declared member size.

    A single member whose *declared* size already exceeds the cap is
    rejected before any decompression. The authoritative accounting lives in
    _read_zip_member, which charges **actual** decompressed bytes to the
    shared budget: ZipInfo.file_size comes from the archive central
    directory and is attacker-controlled, so trusting it for the cumulative
    limit would let a crafted archive declare a tiny size and still expand
    to gigabytes.
    """
    if info.file_size > MAX_ZIP_DECOMPRESSED_BYTES:
        raise ZipBudgetExceeded(
            "zip decompressed size exceeds cap "
            f"({MAX_ZIP_DECOMPRESSED_BYTES} bytes); refusing to process"
        )


def _read_zip_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, budget: list[int]) -> bytes:
    """Read one zip member, charging real decompressed bytes to the budget.

    Streams the member in chunks so the cumulative cap is enforced on bytes
    actually produced, not on the declared ``file_size``; raises
    ``ZipBudgetExceeded`` the moment the cap is crossed, before the whole
    member is buffered.
    """
    _check_zip_budget(info, budget)
    with zf.open(info) as stream:
        chunks: list[bytes] = []
        while True:
            chunk = stream.read(1 << 16)
            if not chunk:
                break
            budget[0] += len(chunk)
            if budget[0] > MAX_ZIP_DECOMPRESSED_BYTES:
                raise ZipBudgetExceeded(
                    "zip decompressed size exceeds cap "
                    f"({MAX_ZIP_DECOMPRESSED_BYTES} bytes); refusing to process"
                )
            chunks.append(chunk)
    return b"".join(chunks)


# Frontmatter / meta keys that often carry AI provenance
AI_FRONTMATTER_KEYS = frozenset(
    {
        "generator",
        "ai",
        "ai_generated",
        "ai-generated",
        "claude",
        "anthropic",
        "openai",
        "gemini",
        "synthid",
        "c2pa",
        "content_credentials",
        "contentcredentials",
        "provenance",
        "digital_source_type",
        "digitalsourcetype",
        "created_with",
        "createdwith",
        "model",
        "llm",
    }
)

AI_META_NAME_RE = re.compile(
    r"generator|ai[-_ ]?generated|claude|anthropic|openai|gemini|synthid|"
    r"c2pa|content.?credential|provenance|digital.?source|aigc",
    re.I,
)

SVG_DROP_TAGS = frozenset(
    {
        "{http://www.w3.org/2000/svg}metadata",
        "metadata",
        "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF",
        "{adobe:ns:meta/}xmpmeta",
    }
)


@dataclass
class ContainerInspectReport:
    path: str
    format: str
    has_c2pa: bool
    has_ai_metadata: bool
    findings: list[str] = field(default_factory=list)
    tools: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # Layer A (invisible/format Unicode) scan of the text body, populated only
    # for the formats clean_container() actually scrubs. Without it, inspect
    # reported a markdown/html file carrying invisible carriers as clean while
    # clean then went on to remove them.
    layer_a_total: int = 0
    layer_a_hits: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "format": self.format,
            "has_c2pa": self.has_c2pa,
            "has_ai_metadata": self.has_ai_metadata,
            "findings": self.findings,
            "findings_confidence": [classify_finding_confidence(f) for f in self.findings],
            "tools": self.tools,
            "details": self.details,
            "notes": self.notes,
            # Same key as TextInspectReport so every caller — including the
            # HTTP server's `suspicious` flag — reads both report kinds alike.
            "suspicious_total": self.layer_a_total,
            "layer_a_hits": self.layer_a_hits,
        }


def detect_container_format(path: Path, data: bytes | None = None) -> str:
    ext = path.suffix.lower()
    if ext in (".svg",):
        return "svg"
    if ext in (".pdf",):
        return "pdf"
    if ext in (".docx",):
        return "docx"
    if ext in (".xlsx",):
        return "xlsx"
    if ext in (".pptx",):
        return "pptx"
    if ext in (".odt",):
        return "odt"
    if ext in (".epub",):
        return "epub"
    if ext in (".html", ".htm"):
        return "html"
    if ext in (".md", ".markdown", ".mdx"):
        return "markdown"
    if data is not None:
        if data[:4] == b"%PDF":
            return "pdf"
        if data[:100].lstrip().startswith(b"<") and b"svg" in data[:500].lower():
            return "svg"
        if data[:2] == b"PK":
            # zip-based; sniff
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    names = set(zf.namelist())
                    if "mimetype" in names:
                        info = zf.getinfo("mimetype")
                        budget = [0]
                        raw = _read_zip_member(zf, info, budget)
                        try:
                            mt = raw.decode("ascii", errors="ignore").strip()
                            if "epub" in mt:
                                return "epub"
                            if "opendocument" in mt or "oasis" in mt:
                                return "odt"
                        except (UnicodeDecodeError, AttributeError):
                            pass
                    if "word/document.xml" in names or any(n.startswith("word/") for n in names):
                        return "docx"
                    if "xl/workbook.xml" in names or any(n.startswith("xl/") for n in names):
                        return "xlsx"
                    if "ppt/presentation.xml" in names or any(n.startswith("ppt/") for n in names):
                        return "pptx"
                    if "content.xml" in names and (
                        "meta.xml" in names or "META-INF/manifest.xml" in names
                    ):
                        return "odt"
                    if "META-INF/container.xml" in names and any(n.endswith(".opf") for n in names):
                        return "epub"
            except _ZIP_PARSE_ERRORS:
                pass
    return "unknown"


def _blob_hits(blob: bytes) -> tuple[bool, bool, list[str]]:
    lower = blob.lower()
    findings: list[str] = []
    has_c2pa = False
    has_ai = False
    for n in C2PA_MARKERS:
        if n.lower() in lower:
            has_c2pa = True
            findings.append(f"marker:{n.decode('ascii', errors='replace')}")
    for n in AI_META_HINTS:
        if n.lower() in lower:
            has_ai = True
            label = n.decode("ascii", errors="replace")
            if label not in {f.split(":", 1)[-1] for f in findings}:
                findings.append(f"ai:{label}")
    return has_c2pa, has_ai or has_c2pa, findings[:30]


# Charsets for the hand-written data-URI parser below. Parsing the URI
# structure with plain string scans (instead of a regex) keeps the pass linear
# and accepts any MIME parameter sequence with no fixed counts or length limits,
# while also never feeding a user-controlled string to a regex engine.
_ASCII_ALNUM = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
_DATA_URI_MIME_CHARS = _ASCII_ALNUM | frozenset("+-.")
_DATA_URI_PAYLOAD_CHARS = _ASCII_ALNUM | frozenset("+/=%")
# Characters that can never appear in an unquoted data-URI body; they terminate
# a candidate span and cannot be part of its header or payload.
_DATA_URI_BREAK = frozenset("\"'<>()")
# Characters that end a parameter value: the ';' and ',' separators plus the
# boundary set above.
_DATA_URI_PARAM_BREAK = _DATA_URI_BREAK | frozenset(",;")


def _iter_data_uris(text: str) -> Iterator[tuple[int, int, str, str, str]]:
    """Yield (start, end, mime, params, payload) for each data:image URI.

    Linear regardless of input shape: each candidate ('data:image/' up to the
    next quote/angle/paren) is parsed once, and a candidate that does not form a
    URI is skipped across, so an adversarial "data:image/+;" flood costs one
    pass, not a rescan per prefix.
    """
    n = len(text)
    pos = 0
    low = text.lower()
    while True:
        i = low.find("data:image/", pos)
        if i < 0:
            return
        k = i + len("data:image/")
        mime_start = k
        while k < n and text[k] in _DATA_URI_MIME_CHARS:
            k += 1
        mime = text[mime_start:k]
        if not mime:
            pos = i + 1
            continue
        params_start = k
        while k < n and text[k] == ";":
            k += 1
            while k < n and text[k] not in _DATA_URI_PARAM_BREAK and not text[k].isspace():
                k += 1
        params = text[params_start:k]
        if k >= n or text[k] != ",":
            pos = _skip_data_uri_candidate(text, i)
            continue
        k += 1  # comma
        payload_start = k
        while k < n and (text[k] in _DATA_URI_PAYLOAD_CHARS or text[k].isspace()):
            k += 1
        payload = text[payload_start:k]
        if not payload:
            pos = _skip_data_uri_candidate(text, i)
            continue
        yield i, k, mime, params, payload
        pos = k


def _skip_data_uri_candidate(text: str, start: int) -> int:
    """Return the position to resume scanning after a non-URI candidate.

    Skips past the current 'data:image/' candidate (to its terminating
    quote/angle/paren or end of text), so a document full of 'data:image/+;'
    with no comma costs one pass rather than a rescan per prefix.
    """
    n = len(text)
    j = start
    while j < n and text[j] not in _DATA_URI_BREAK:
        j += 1
    return j if j > start + 1 else start + 1


def _inspect_embedded_data_uris(text: str) -> tuple[bool, bool, list[str]]:
    has_c2pa = False
    has_ai = False
    findings: list[str] = []

    for _start, _end, mime, params, payload in _iter_data_uris(text):
        mime_l = mime.lower()
        params_l = params.lower()
        is_b64 = "base64" in params_l

        try:
            if is_b64:
                raw_b64 = re.sub(r"\s+", "", payload)
                pad = len(raw_b64) % 4
                if pad:
                    raw_b64 += "=" * (4 - pad)
                data = base64.b64decode(raw_b64)
            else:
                data = urllib.parse.unquote_to_bytes(payload)
        except Exception:  # noqa: S112 - malformed data URI; skip to next match
            continue

        if not data:
            continue

        fmt = detect_image_format(data)
        if fmt == "png":
            sub_c2pa, sub_ai, sub_findings = inspect_png(data)
        elif fmt == "jpeg":
            sub_c2pa, sub_ai, sub_findings = inspect_jpeg(data)
        elif fmt == "webp":
            sub_c2pa, sub_ai, sub_findings = inspect_webp(data)
        elif fmt in ("avif", "heic"):
            sub_c2pa, sub_ai, sub_findings = inspect_isobmff(data, fmt)
        elif "svg" in mime_l or data.lstrip().startswith(b"<"):
            sub_c2pa, sub_ai, sub_findings, _ = inspect_svg(data)
        else:
            sub_c2pa, sub_ai, sub_findings = _blob_hits(data)

        if sub_c2pa:
            has_c2pa = True
        if sub_ai or sub_c2pa:
            has_ai = True
        for f in sub_findings:
            findings.append(f"embedded data:image/{mime_l}: {f}")

    return has_c2pa, has_ai, findings


def _clean_embedded_data_uris(
    text: str, *, strip_all_metadata: bool = True
) -> tuple[str, list[str]]:
    actions: list[str] = []

    def _clean_one(mime: str, params: str, payload: str) -> str | None:
        """Return the rebuilt data URI, or None when nothing changed."""
        is_b64 = "base64" in params.lower()

        try:
            if is_b64:
                raw_b64 = re.sub(r"\s+", "", payload)
                pad = len(raw_b64) % 4
                if pad:
                    raw_b64 += "=" * (4 - pad)
                data = base64.b64decode(raw_b64)
            else:
                data = urllib.parse.unquote_to_bytes(payload)
        except Exception:
            return None

        if not data:
            return None

        fmt = detect_image_format(data)
        sub_actions: list[str] = []
        cleaned_bytes = data

        try:
            if fmt == "png":
                cleaned_bytes, sub_actions = strip_png(data, strip_all_text=strip_all_metadata)
            elif fmt == "jpeg":
                cleaned_bytes, sub_actions = strip_jpeg(data, strip_all_app=strip_all_metadata)
            elif fmt == "webp":
                cleaned_bytes, sub_actions = strip_webp(data, strip_all_metadata=strip_all_metadata)
            elif fmt in ("avif", "heic"):
                cleaned_bytes, sub_actions = strip_isobmff(
                    data, fmt, strip_all_metadata=strip_all_metadata
                )
            elif "svg" in mime.lower() or data.lstrip().startswith(b"<"):
                cleaned_bytes, sub_actions = clean_svg(data)
        except Exception:
            return None

        if not any("drop" in a.lower() for a in sub_actions) or cleaned_bytes == data:
            return None

        actions.append(f"cleaned embedded data:image/{mime} ({', '.join(sub_actions[:2])})")

        if is_b64:
            new_b64 = base64.b64encode(cleaned_bytes).decode("ascii")
            return f"data:image/{mime}{params},{new_b64}"
        return f"data:image/{mime}{params},{urllib.parse.quote_from_bytes(cleaned_bytes)}"

    out: list[str] = []
    last = 0
    for start, end, mime, params, payload in _iter_data_uris(text):
        out.append(text[last:start])
        rebuilt = _clean_one(mime, params, payload)
        out.append(text[start:end] if rebuilt is None else rebuilt)
        last = end
    out.append(text[last:])
    return "".join(out), actions


# ---------------------------------------------------------------------------
# Linear tag-block scanning
# ---------------------------------------------------------------------------
#
# The lazy ".*?</close>" idiom is quadratic on adversarial input: with many
# opening tags and no closing tag, the engine rescans to end-of-input from
# every candidate start position (CPython's "re" holds the GIL, so one such
# request stalls the whole single-process service for its duration). The
# helpers below locate every opening and closing tag once and pair them with a
# forward pointer - O(n) total regardless of input shape - while preserving
# the exact match semantics of re.subn with a lazy ".*?" (the first closing
# tag at/after each opening tag's end, blocks never overlapping).


def _iter_tag_blocks(text, open_re, close_re):
    """Yield (open_start, open_end, close_start, close_end) for each block.

    Works on str and bytes alike. Linear in len(text): closing tags are
    collected once with finditer and consumed by a forward pointer, so an
    unbounded run of unclosed opening tags costs one pass, not one rescan per
    opening tag.
    """
    closes = list(close_re.finditer(text))
    ci = 0
    last_end = 0
    for om in open_re.finditer(text):
        if om.start() < last_end:
            continue
        while ci < len(closes) and closes[ci].start() < om.end():
            ci += 1
        if ci >= len(closes):
            return
        cm = closes[ci]
        yield om.start(), om.end(), cm.start(), cm.end()
        last_end = cm.end()


def _drop_tag_blocks(text, open_re, close_re):
    """Remove every open...close block. Return (text, count). Linear time."""
    out = []
    last = 0
    count = 0
    for os_, _oe, _cs, ce in _iter_tag_blocks(text, open_re, close_re):
        out.append(text[last:os_])
        last = ce
        count += 1
    if not count:
        return text, 0
    out.append(text[last:])
    return text[:0].join(out), count


def _drop_blocks_if(text, open_re, close_re, predicate):
    """Drop blocks whose full text satisfies predicate. Return (text, count).

    Blocks failing the predicate are kept verbatim; predicate is called with
    the whole block (open tag through closing tag), matching the
    callback-based re.sub sites this replaces.
    """
    out = []
    last = 0
    count = 0
    for os_, _oe, _cs, ce in _iter_tag_blocks(text, open_re, close_re):
        if not predicate(text[os_:ce]):
            continue
        out.append(text[last:os_])
        last = ce
        count += 1
    if not count:
        return text, 0
    out.append(text[last:])
    return text[:0].join(out), count


# ---------------------------------------------------------------------------
# Markdown frontmatter
# ---------------------------------------------------------------------------

_FM_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)


def _parse_simple_yaml_keys(block: str) -> list[tuple[str, str, int]]:
    """Return list of (key, full_line, line_index) for top-level keys only."""
    rows: list[tuple[str, str, int]] = []
    for i, line in enumerate(block.splitlines()):
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line[0] in (" ", "\t", "-"):
            continue  # nested / list — leave alone
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*:", line)
        if m:
            rows.append((m.group(1), line, i))
    return rows


def inspect_markdown(text: str) -> tuple[bool, bool, list[str], dict]:
    findings: list[str] = []
    has_ai = False
    has_fm = False
    keys = []
    m = _FM_RE.match(text)
    if m:
        has_fm = True
        block = m.group(1)
        for key, _line, _i in _parse_simple_yaml_keys(block):
            keys.append(key)
            if key.lower() in AI_FRONTMATTER_KEYS or AI_META_NAME_RE.search(key):
                has_ai = True
                findings.append(f"frontmatter key: {key}")
            # also check value
            val = _line.split(":", 1)[1] if ":" in _line else ""
            if AI_META_NAME_RE.search(val):
                has_ai = True
                findings.append(f"frontmatter value hit on {key}")

    uri_c2pa, uri_ai, uri_findings = _inspect_embedded_data_uris(text)
    if uri_c2pa:
        has_ai = True
    if uri_ai:
        has_ai = True
    findings.extend(uri_findings)

    c2pa = uri_c2pa or any("c2pa" in f.lower() or "content" in f.lower() for f in findings)
    return c2pa, has_ai, findings, {"has_frontmatter": has_fm, "keys": keys}


def clean_markdown(text: str) -> tuple[str, list[str]]:
    actions: list[str] = []
    m = _FM_RE.match(text)
    if m:
        block = m.group(1)
        body = text[m.end() :]
        kept: list[str] = []
        dropping = False  # inside the nested block of a dropped top-level key
        for line in block.splitlines():
            stripped = line.strip()

            # Blank lines and comments belong to whichever block we are inside.
            if not stripped or stripped.startswith("#"):
                if not dropping:
                    kept.append(line)
                continue

            # Continuation lines (nested mappings, list items) follow their parent.
            if line[0] in (" ", "\t", "-"):
                if not dropping:
                    kept.append(line)
                continue

            km = re.match(r"^([A-Za-z0-9_.-]+)\s*:", line)
            if not km:
                dropping = False
                kept.append(line)
                continue

            key = km.group(1)
            val = line.split(":", 1)[1] if ":" in line else ""
            if key.lower() in AI_FRONTMATTER_KEYS or AI_META_NAME_RE.search(key):
                actions.append(f"drop frontmatter key: {key}")
                dropping = True
                continue
            if AI_META_NAME_RE.search(val):
                actions.append(f"drop frontmatter key (value hit): {key}")
                dropping = True
                continue

            dropping = False
            kept.append(line)
        new_block = "\n".join(kept).strip("\n")
        if new_block:
            out = f"---\n{new_block}\n---\n{body}"
        else:
            out = body.lstrip("\n")
            actions.append("removed empty frontmatter block")
    else:
        out = text

    out, uri_actions = _clean_embedded_data_uris(out)
    if uri_actions:
        actions.extend(uri_actions)

    if not actions:
        actions.append("no AI frontmatter keys or embedded data URIs removed")
    return out, actions


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_META_TAG_RE = re.compile(
    r"<meta\b[^>]*>",
    re.I,
)
_META_ATTR_RE = re.compile(
    r"""(name|property|content|generator)\s*=\s*["']([^"']*)["']""",
    re.I,
)

# Known AI vendor names for the "generator" meta tag. A plain CMS generator
# (WordPress, Elementor) is CMS provenance, not AI-generator metadata.
_GENERATOR_AI_RE = re.compile(
    r"claude|anthropic|openai|chatgpt|gemini|synthid|copilot|midjourney|dall.?e|stable.?diffusion",
    re.I,
)


def _meta_attrs(tag: str) -> dict[str, str]:
    return {name.lower(): value for name, value in _META_ATTR_RE.findall(tag)}


def _is_cms_generator_meta(tag: str) -> bool:
    """Return True for a generator meta tag that is CMS provenance, not AI."""
    attrs = _meta_attrs(tag)
    name_or_prop = (
        attrs.get("name") or attrs.get("property") or attrs.get("generator") or ""
    ).lower()
    if name_or_prop != "generator":
        return False
    return not (_GENERATOR_AI_RE.search(attrs.get("content", "")) or _GENERATOR_AI_RE.search(tag))


_JSONLD_CLOSE_RE = re.compile(r"</script>", re.I)
# HTML treats form feed as whitespace; include it wherever attribute separators
# are checked.
_HTML_SPACE = " \t\r\n\f"


def _find_tag_end(text: str, start: int) -> int:
    """Return the index just after the closing '>' of the tag at 'start'.

    Quote-aware: a '>' inside a quoted attribute value does not end the tag.
    Linear; returns len(text) when the tag is unterminated.
    """
    n = len(text)
    i = start + 1
    while i < n:
        c = text[i]
        if c == ">":
            return i + 1
        if c in "\"'":
            quote = c
            i += 1
            while i < n and text[i] != quote:
                i += 1
            i += 1  # skip closing quote
        else:
            i += 1
    return n


def _iter_script_blocks(
    text: str,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield (open_start, open_end, close_start, close_end) for script blocks.

    Linear and quote-aware, with no length cap on the opening tag: each opening
    tag is scanned to its true boundary and closing tags are advanced by a
    forward pointer, so an unterminated run of '<script' costs one pass, not a
    rescan per prefix. Yields the same 4-tuple shape as _iter_tag_blocks.
    """
    closes = [m.start() for m in _JSONLD_CLOSE_RE.finditer(text)]
    ci = 0
    last_end = 0
    pos = 0
    n = len(text)
    low = text.lower()
    while True:
        i = low.find("<script", pos)
        if i < 0:
            return
        after = i + 7
        if after >= n or text[after] not in ">" + _HTML_SPACE + "/":
            pos = i + 1
            continue
        open_end = _find_tag_end(text, i)
        if i < last_end:
            pos = max(open_end, i + 1)
            continue
        while ci < len(closes) and closes[ci] < open_end:
            ci += 1
        if ci >= len(closes):
            return
        close_start = closes[ci]
        close_end = close_start + len("</script>")
        yield i, open_end, close_start, close_end
        last_end = close_end
        pos = open_end


def _script_tag_is_jsonld(open_tag: str) -> bool:
    """True iff the opening tag has a top-level type="application/ld+json".

    Single-pass and quote-aware: quoted attribute values are skipped as a unit,
    so only an actual top-level attribute named 'type' with the JSON-LD value
    is matched. Linear in the tag length.
    """
    i, n = 0, len(open_tag)
    if i < n and open_tag[i] == "<":
        i += 1
    while i < n and open_tag[i] not in _HTML_SPACE + "/>":  # tag name
        i += 1
    while i < n:
        while i < n and open_tag[i] in _HTML_SPACE:
            i += 1
        if i >= n or open_tag[i] == ">":
            return False
        name_start = i
        while i < n and open_tag[i] not in "=" + _HTML_SPACE + "/>":
            i += 1
        name = open_tag[name_start:i]
        while i < n and open_tag[i] in _HTML_SPACE:
            i += 1
        value = ""
        if i < n and open_tag[i] == "=":
            i += 1
            while i < n and open_tag[i] in _HTML_SPACE:
                i += 1
            if i < n and open_tag[i] in "\"'":
                quote = open_tag[i]
                i += 1
                value_start = i
                while i < n and open_tag[i] != quote:
                    i += 1
                value = open_tag[value_start:i]
                i += 1  # skip closing quote
            else:
                value_start = i
                while i < n and open_tag[i] not in _HTML_SPACE + ">":
                    i += 1
                value = open_tag[value_start:i]
        if name.lower() == "type" and value.lower() == "application/ld+json":
            return True
    return False


def inspect_html(text: str) -> tuple[bool, bool, list[str], dict]:
    findings: list[str] = []
    has_ai = False
    has_c2pa = False
    for tag in _META_TAG_RE.findall(text):
        if re.search(r"c2pa|content.?credential", tag, re.I):
            has_c2pa = True
        if _is_cms_generator_meta(tag):
            findings.append(f"info: cms generator: {tag[:120]}")
            continue
        if AI_META_NAME_RE.search(tag) or any(
            h.decode("ascii", "ignore").lower() in tag.lower() for h in AI_META_HINTS[:12]
        ):
            has_ai = True
            findings.append(f"meta: {tag[:120]}")
    for os_, oe, _cs, ce in _iter_script_blocks(text):
        if not _script_tag_is_jsonld(text[os_:oe]):
            continue
        blob = text[os_:ce]
        if AI_META_NAME_RE.search(blob) or re.search(
            r"DigitalSourceType|trainedAlgorithmicMedia|SoftwareAgent", blob, re.I
        ):
            has_ai = True
            findings.append("json-ld provenance-like block")
            if re.search(r"c2pa|contentcredential", blob, re.I):
                has_c2pa = True
    # data-ai* attributes
    for m in re.finditer(r"\bdata-ai[\w-]*\s*=\s*[\"'][^\"']*[\"']", text, re.I):
        has_ai = True
        findings.append(f"attr: {m.group(0)[:80]}")

    uri_c2pa, uri_ai, uri_findings = _inspect_embedded_data_uris(text)
    if uri_c2pa:
        has_c2pa = True
    if uri_ai:
        has_ai = True
    findings.extend(uri_findings)

    return has_c2pa, has_ai, findings, {}


def clean_html(text: str) -> tuple[str, list[str]]:
    actions: list[str] = []

    def _meta_sub(m: re.Match[str]) -> str:
        tag = m.group(0)
        if _is_cms_generator_meta(tag):
            return tag
        if AI_META_NAME_RE.search(tag) or re.search(
            r"generator|claude|anthropic|openai|gemini|synthid|c2pa|aigc", tag, re.I
        ):
            actions.append(f"drop meta: {tag[:80]}")
            return ""
        return tag

    out = _META_TAG_RE.sub(_meta_sub, text)

    def _block_is_ai(open_tag: str, blob: str) -> bool:
        if not _script_tag_is_jsonld(open_tag):
            return False
        return AI_META_NAME_RE.search(blob) or re.search(
            r"DigitalSourceType|trainedAlgorithmicMedia|SoftwareAgent", blob, re.I
        )

    kept = []
    last = 0
    n = 0
    for os_, oe, _cs, ce in _iter_script_blocks(out):
        blob = out[os_:ce]
        if not _block_is_ai(out[os_:oe], blob):
            continue
        kept.append(out[last:os_])
        last = ce
        n += 1
    if n:
        kept.append(out[last:])
        out = "".join(kept)
        actions.extend(["drop json-ld provenance-like script"] * n)
    out2, n = re.subn(r"\sdata-ai[\w-]*\s*=\s*[\"'][^\"']*[\"']", "", out, flags=re.I)
    if n:
        actions.append(f"drop data-ai* attributes x{n}")
        out = out2

    out, uri_actions = _clean_embedded_data_uris(out)
    if uri_actions:
        actions.extend(uri_actions)

    if not actions:
        actions.append("no HTML AI meta removed")
    return out, actions


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------

# Opening/closing tag patterns for the linear block scans below. Kept as
# separate open/close halves: the lazy ".*?</close>" form is quadratic when
# many opening tags have no closing tag (see _iter_tag_blocks).
_SVG_METADATA_OPEN_RE = re.compile(r"<metadata\b[^>]*>", re.I)
_SVG_METADATA_CLOSE_RE = re.compile(r"</metadata\s*>", re.I)
_SVG_XMPMETA_OPEN_RE = re.compile(r"<x:xmpmeta\b[^>]*>", re.I)
_SVG_XMPMETA_CLOSE_RE = re.compile(r"</x:xmpmeta\s*>", re.I)
_SVG_COMMENT_OPEN_RE = re.compile(r"<!--")
# HTML comment end tags are "-->" or "--!>" (CodeQL: bad HTML filtering
# regexp otherwise). XML/SVG only emits "-->", but accepting both keeps the
# scrub effective on HTML-flavoured inputs.
_SVG_COMMENT_CLOSE_RE = re.compile(r"--!?>")


def inspect_svg(data: bytes) -> tuple[bool, bool, list[str], dict]:
    findings: list[str] = []
    has_c2pa, has_ai, hits = _blob_hits(data)
    findings.extend(hits)
    try:
        text = data.decode("utf-8", errors="replace")
        if re.search(r"<metadata[\s>]", text, re.I):
            findings.append("svg <metadata> present")
            has_ai = True  # often XMP; treat as inspect signal
        if re.search(r"xmpmeta|rdf:RDF|contentcredentials", text, re.I):
            has_ai = True
            findings.append("XMP/RDF-like content in SVG")
        if re.search(r"c2pa|jumbf", text, re.I):
            has_c2pa = True

        uri_c2pa, uri_ai, uri_findings = _inspect_embedded_data_uris(text)
        if uri_c2pa:
            has_c2pa = True
        if uri_ai:
            has_ai = True
        findings.extend(uri_findings)
    except Exception as e:
        findings.append(f"svg decode note: {e}")
    return has_c2pa, has_ai or has_c2pa, findings, {}


_XML_DECL_KEYWORDS = ("doctype", "entity")


def _xml_decl_keyword(text: str, i: int) -> str | None:
    """Return "doctype"/"entity" if text[i:] opens that declaration, else None."""
    if text[i : i + 2] != "<!":
        return None
    rest = text[i + 2 :]
    for kw in _XML_DECL_KEYWORDS:
        if rest.lower().startswith(kw):
            # Require a word boundary so names like <!DOCTYPEfoo> aren't matched.
            nxt = rest[len(kw) : len(kw) + 1]
            if nxt and (nxt.isalnum() or nxt in "_:"):
                return None
            return kw
    return None


def _xml_decl_end(text: str, i: int, keyword: str) -> int:
    """Return the index just past a declaration's closing '>', or -1 if unterminated.

    Quotes and the internal subset are respected, so a '>' inside a quoted
    external identifier or a nested internal subset does not end the
    declaration early.
    """
    j = i + 2 + len(keyword)
    n = len(text)
    quote: str | None = None
    subset_depth = 0
    while j < n:
        c = text[j]
        if quote is not None:
            # Line breaks are legal inside quoted system/public literals and
            # entity values, so they don't terminate the declaration; the first
            # unquoted '>' ends it.
            if c == quote:
                quote = None
            j += 1
            continue
        # DTD comments inside the internal subset must not affect subset_depth
        # or look like a close, so skip them whole.
        if text.startswith("<!--", j):
            end = text.find("-->", j)
            if end == -1:
                return -1
            j = end + 3
            continue
        if c in "\"'":
            quote = c
            j += 1
            continue
        if c == "[":
            subset_depth += 1
            j += 1
            continue
        if c == "]":
            if subset_depth:
                subset_depth -= 1
            j += 1
            continue
        if c == ">" and subset_depth == 0:
            return j + 1
        j += 1
    return -1


def _strip_xml_declarations(text: str) -> tuple[str, int]:
    """Remove top-level <!DOCTYPE ...> and <!ENTITY ...> declarations.

    Only declarations in markup context are stripped, so matching text inside
    CDATA sections, comments, and quoted attribute values (which do not start a
    declaration) is preserved. A declaration is consumed through its true
    terminator: quoted external identifiers and the internal subset, including
    nested brackets and embedded '>' characters, are respected. An unterminated
    declaration is left intact rather than partially deleted.
    """
    i = 0
    n = len(text)
    out: list[str] = []
    removed = 0
    in_tag = False
    quote: str | None = None
    while i < n:
        c = text[i]
        if in_tag:
            if quote is not None:
                out.append(c)
                if c == quote:
                    quote = None
            elif c in "\"'":
                quote = c
                out.append(c)
            elif c == ">":
                in_tag = False
                out.append(c)
            else:
                out.append(c)
            i += 1
            continue
        if text.startswith("<![CDATA[", i):
            end = text.find("]]>", i)
            if end == -1:
                out.append(text[i:])
                break
            out.append(text[i : end + 3])
            i = end + 3
            continue
        if text.startswith("<!--", i):
            end = text.find("-->", i)
            if end == -1:
                out.append(text[i:])
                break
            out.append(text[i : end + 3])
            i = end + 3
            continue
        keyword = _xml_decl_keyword(text, i)
        if keyword:
            end = _xml_decl_end(text, i, keyword)
            if end == -1:
                out.append(c)
                i += 1
                continue
            removed += 1
            i = end
            continue
        out.append(c)
        if c == "<":
            in_tag = True
        i += 1
    return "".join(out), removed


def _strip_root_svg_attrs(text: str) -> tuple[str, int]:
    """Remove provenance attributes from the root <svg ...> start tag only.

    The attribute substitution runs on the parsed root start tag so matching
    text in CDATA, comments, or text content is untouched. Both single- and
    double-quoted attribute values are supported.
    """
    n = len(text)
    i = 0
    start = -1
    while i < n:
        if text.startswith("<![CDATA[", i):
            end = text.find("]]>", i)
            if end == -1:
                return text, 0
            i = end + 3
            continue
        if text.startswith("<!--", i):
            end = text.find("-->", i)
            if end == -1:
                return text, 0
            i = end + 3
            continue
        if text.startswith("<?", i):
            # A processing instruction may contain "<svg"; skip it so the root
            # element start tag is what gets cleaned.
            end = text.find("?>", i)
            if end == -1:
                return text, 0
            i = end + 2
            continue
        if text[i : i + 4].lower() == "<svg":
            nxt = text[i + 4 : i + 5]
            if not (nxt and (nxt.isalnum() or nxt in "_:.-")):
                start = i
                break
        i += 1
    if start == -1:
        return text, 0
    # Find the quote-aware '>' that closes the start tag.
    j = start + 4
    quote: str | None = None
    while j < n:
        c = text[j]
        if quote is not None:
            if c == quote:
                quote = None
            j += 1
            continue
        if c in "\"'":
            quote = c
            j += 1
            continue
        if c == ">":
            break
        j += 1
    else:
        return text, 0  # unclosed start tag; leave as-is
    tag = text[start : j + 1]
    new_tag, cnt = re.subn(
        r'\s(?:inkscape:version|sodipodi:docname|generator)\s*=\s*("[^"]*"|\'[^\']*\')',
        "",
        tag,
        flags=re.I,
    )
    if not cnt:
        return text, 0
    return text[:start] + new_tag + text[j + 1 :], cnt


def clean_svg(data: bytes) -> tuple[bytes, list[str]]:
    """Strip AI provenance metadata from SVG content, returning (bytes, actions).

    Removes <metadata>/<xmpmeta> blocks, XML DOCTYPE and ENTITY declarations,
    AI-marker comments, and embedded data URIs, and drops generator-like
    attributes on the root element. Declaration stripping is context-aware so
    the document body, CDATA sections, comments, and quoted attribute values are
    preserved.
    """
    actions: list[str] = []
    text = data.decode("utf-8", errors="surrogateescape")
    # Drop metadata blocks (linear scan - lazy .*? is quadratic on unclosed tags)
    new, n = _drop_tag_blocks(text, _SVG_METADATA_OPEN_RE, _SVG_METADATA_CLOSE_RE)
    if n:
        actions.append(f"drop <metadata> x{n}")
        text = new
    # Drop adobe xmp packets
    new, n = _drop_tag_blocks(text, _SVG_XMPMETA_OPEN_RE, _SVG_XMPMETA_CLOSE_RE)
    if n:
        actions.append(f"drop xmpmeta x{n}")
        text = new

    # Drop XML DOCTYPE and ENTITY declarations (context-aware)
    text, decl_count = _strip_xml_declarations(text)
    if decl_count:
        actions.append(f"drop DOCTYPE/entity declarations x{decl_count}")

    # Drop comments that look like provenance (linear scan)
    def _cmt(block: str) -> bool:
        return bool(AI_META_NAME_RE.search(block))

    new, n = _drop_blocks_if(text, _SVG_COMMENT_OPEN_RE, _SVG_COMMENT_CLOSE_RE, _cmt)
    if n:
        actions.extend(["drop SVG comment with AI markers"] * n)
        text = new

    # Clean embedded data URIs
    text, uri_actions = _clean_embedded_data_uris(text)
    if uri_actions:
        actions.extend(uri_actions)

    # Strip generator-like attributes on the root <svg> start tag independently
    # of the actions above, so they are removed whenever present (not only when
    # nothing else needed cleaning).
    text, n = _strip_root_svg_attrs(text)
    if n:
        actions.append(f"drop generator-like attrs x{n}")
    if not actions:
        actions.append("no SVG metadata removed")
    return text.encode("utf-8", errors="surrogateescape"), actions


# ---------------------------------------------------------------------------
# DOCX / ODT (zip + XML)
# ---------------------------------------------------------------------------

# Open/close tag halves for the linear metadata-block scans in clean_odt.
_ODT_GENERATOR_OPEN_RE = re.compile(r"<meta:generator\b[^>]*>", re.I)
_ODT_GENERATOR_CLOSE_RE = re.compile(r"</meta:generator\s*>", re.I)
_DC_CREATOR_OPEN_RE = re.compile(r"<dc:creator\b[^>]*>", re.I)
_DC_CREATOR_CLOSE_RE = re.compile(r"</dc:creator\s*>", re.I)

DOCX_META_PARTS = (
    "docProps/core.xml",
    "docProps/app.xml",
    "docProps/custom.xml",
)
DOCX_CUSTOM_PREFIXES = (
    "customXml/",
    "docProps/",
)

# Provenance fields in docProps/core.xml and docProps/app.xml that always come
# out empty. dc:title is deliberately not listed: it is the document's own
# heading, not provenance. AppVersion is also excluded: ECMA-376 Part 1
# §15.2.12.1 requires AppVersion to match \d+\.\d{4} when present, so blanking it
# produces schema-invalid XML that Word/Office rejects with unreadable content (#283).
DOCX_SCRUB_FIELDS = (
    ("dc:creator", "dc:creator"),
    ("cp:lastModifiedBy", "cp:lastModifiedBy"),
    ("dc:description", "dc:description"),
    ("cp:keywords", "cp:keywords"),
    ("dc:subject", "dc:subject"),
    ("cp:category", "cp:category"),
    ("Application", "Application"),
    ("Company", "Company"),
    ("Manager", "Manager"),
)


def _zip_namelist(data: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return zf.namelist()


def _is_docx_meta_part(name: str) -> bool:
    """Return True for DOCX/XLSX/PPTX parts that carry provenance, not visible content."""
    return name.startswith(("docProps/", "customXml/"))


def _inspect_ooxml_zip(
    data: bytes, fmt: str, budget: list[int] | None = None
) -> tuple[bool, bool, list[str], dict]:
    findings: list[str] = []
    has_c2pa = False
    has_ai = False
    parts: list[str] = []
    if budget is None:
        budget = [0]
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            parts = zf.namelist()
            for info in zf.infolist():
                _check_zip_budget(info, budget)
                name = info.filename
                # Check media parts for C2PA/AI metadata
                if re.search(
                    r"^(?:word|xl|ppt)/media/.+\.(png|jpe?g|webp|avif|heic|gif|bmp|tiff?|svg)$",
                    name,
                    re.I,
                ):
                    raw = _read_zip_member(zf, info, budget)
                    img_fmt = detect_image_format(raw)
                    sub_c2pa, sub_ai, sub_findings = False, False, []
                    if img_fmt == "png":
                        sub_c2pa, sub_ai, sub_findings = inspect_png(raw)
                    elif img_fmt == "jpeg":
                        sub_c2pa, sub_ai, sub_findings = inspect_jpeg(raw)
                    elif img_fmt == "webp":
                        sub_c2pa, sub_ai, sub_findings = inspect_webp(raw)
                    elif img_fmt in ("avif", "heic"):
                        sub_c2pa, sub_ai, sub_findings = inspect_isobmff(raw, img_fmt)
                    elif img_fmt == "gif":
                        sub_c2pa, sub_ai, sub_findings = inspect_gif(raw)
                    elif img_fmt == "tiff":
                        sub_c2pa, sub_ai, sub_findings = inspect_tiff(raw)
                    elif img_fmt == "bmp":
                        sub_c2pa, sub_ai, sub_findings = inspect_bmp(raw)
                    elif name.lower().endswith(".svg") or raw.lstrip().startswith(b"<"):
                        sub_c2pa, sub_ai, sub_findings, _ = inspect_svg(raw)
                    if sub_c2pa:
                        has_c2pa = True
                    if sub_ai or sub_c2pa:
                        has_ai = True
                    for sf in sub_findings:
                        findings.append(f"{name}: {sf}")
                    continue

                # Only metadata/provenance parts carry AI markers. The visible
                # body (word/*.xml, xl/*.xml, ppt/*.xml) may legitimately mention
                # vendor names such as "Claude" without being AI-generated metadata.
                if not _is_docx_meta_part(name):
                    continue
                raw = _read_zip_member(zf, info, budget)
                c2, ai, hits = _blob_hits(raw)
                if c2 or ai:
                    if c2:
                        has_c2pa = True
                    if ai:
                        has_ai = True
                    findings.append(f"{name}: {', '.join(hits[:6])}")
            # always flag customXml presence lightly
            custom = [n for n in parts if n.startswith("customXml/")]
            if custom:
                findings.append(f"customXml parts: {len(custom)}")
    except _ZIP_PARSE_ERRORS as exc:
        # A member that failed to read must not discard the evidence already
        # collected from earlier members, nor read as "opened and found
        # clean": keep the flags/findings, append a partial-read note, and
        # only a wholly-garbage container (nothing accumulated) keeps the old
        # not-a-valid-zip shape (#164).
        if findings:
            findings.append(
                f"partial read of {fmt.upper()} zip ({exc.__class__.__name__}); "
                "evidence above survives, later members were not scanned"
            )
            return has_c2pa, has_ai or has_c2pa, findings, {"parts": len(parts)}
        return False, False, [f"not a valid {fmt.upper()} zip"], {}
    return has_c2pa, has_ai or has_c2pa, findings, {"parts": len(parts)}


def inspect_docx(
    data: bytes, budget: list[int] | None = None
) -> tuple[bool, bool, list[str], dict]:
    return _inspect_ooxml_zip(data, "docx", budget)


def inspect_xlsx(
    data: bytes, budget: list[int] | None = None
) -> tuple[bool, bool, list[str], dict]:
    return _inspect_ooxml_zip(data, "xlsx", budget)


def inspect_pptx(
    data: bytes, budget: list[int] | None = None
) -> tuple[bool, bool, list[str], dict]:
    return _inspect_ooxml_zip(data, "pptx", budget)


_XML_CHAR_REF_RE = re.compile(r"&#(?:x([0-9A-Fa-f]+)|([0-9]+));|&(amp|lt|gt|quot|apos);")
_XML_NAMED_ENTITIES = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'"}


def _decode_xml_entities(s: str) -> str:
    """Decode what a conforming XML parser would resolve: numeric character
    references and the five predefined entities. A watermark carrier encoded as
    ``&#x200B;`` otherwise reaches Layer A as nine ASCII characters and survives
    cleaning, because the parser in Word/Writer decodes the entity afterwards
    (#129). Anything a parser would leave literal stays literal here.
    """

    def _sub(m: re.Match[str]) -> str:
        hex_digits, decimal, named = m.group(1), m.group(2), m.group(3)
        if named:
            return _XML_NAMED_ENTITIES[named]
        codepoint = int(hex_digits, 16) if hex_digits else int(decimal)
        if codepoint == 0 or codepoint > 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
            return m.group(0)  # not a character; leave for the real parser to reject
        try:
            return chr(codepoint)
        except ValueError:
            return m.group(0)

    return _XML_CHAR_REF_RE.sub(_sub, s)


def _reencode_xml_text(s: str) -> str:
    """Escape text content the way a serializer would: ``&``, ``<`` and ``>``.

    ``>`` is legal literally in text content but re-escaping it is a semantic
    no-op, and ``<``/``&`` must be escaped anyway — so a decode/clean/re-encode
    round trip is stable for any well-formed input.
    """
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _scrub_text_runs(
    xml_text: str,
    open_re: re.Pattern[str],
    close_re: re.Pattern[str],
    *,
    normalize_spaces: bool = True,
) -> tuple[str, int, int]:
    """Run Layer A over the text runs delimited by open_re/close_re.

    XML character references are decoded first (a carrier written as
    ``&#x200B;`` otherwise reaches Layer A as nine ASCII characters and
    survives cleaning, only to be resurrected by the parser in Word/Writer —
    #129) and the surviving text is re-encoded on the way out, so the round
    trip is stable for any well-formed input.

    Shared by the DOCX/XLSX/PPTX body scrubs. Linear scan (see
    _iter_tag_blocks) - the previous "(<tag>)(.*?)(</tag>)" lazy pattern was
    quadratic on a run of unclosed opening tags.
    """
    from text_unicode import clean_text  # local import to avoid cycles

    removed = 0
    replaced = 0
    out = []
    last = 0
    for os_, oe, cs_, ce in _iter_tag_blocks(xml_text, open_re, close_re):
        open_tag = xml_text[os_:oe]
        inner = xml_text[oe:cs_]
        close_tag = xml_text[cs_:ce]
        new_inner, stats = clean_text(
            _decode_xml_entities(inner), normalize_spaces=normalize_spaces
        )
        if not (stats["removed_count"] or stats["replaced_count"]):
            continue
        removed += stats["removed_count"]
        replaced += stats["replaced_count"]
        if (new_inner[:1].isspace() or new_inner[-1:].isspace()) and "xml:space" not in open_tag:
            open_tag = open_tag[:-1] + ' xml:space="preserve">'
        out.append(xml_text[last:os_])
        out.append(open_tag + _reencode_xml_text(new_inner) + close_tag)
        last = ce
    out.append(xml_text[last:])
    return "".join(out), removed, replaced


def _scrub_docx_text(xml_text: str, *, normalize_spaces: bool = True) -> tuple[str, int, int]:
    """Run Layer A over the ``<w:t>`` text runs of a DOCX part.

    Only ``w:t`` nodes are touched: field codes (``w:instrText``), run/paragraph
    properties and the surrounding XML are left byte-identical. If leading or
    trailing whitespace survives the clean, the node keeps
    ``xml:space="preserve"`` so Word does not trim it.
    """
    return _scrub_text_runs(
        xml_text,
        re.compile(r"<w:t\b[^>]*>"),
        re.compile(r"</w:t>"),
        normalize_spaces=normalize_spaces,
    )


def _scrub_xlsx_text(xml_text: str, *, normalize_spaces: bool = True) -> tuple[str, int, int]:
    """Run Layer A over the ``<t>`` text elements of an XLSX part."""
    return _scrub_text_runs(
        xml_text, re.compile(r"<t\b[^>]*>"), re.compile(r"</t>"), normalize_spaces=normalize_spaces
    )


def _scrub_pptx_text(xml_text: str, *, normalize_spaces: bool = True) -> tuple[str, int, int]:
    """Run Layer A over the ``<a:t>`` text elements of a PPTX part."""
    return _scrub_text_runs(
        xml_text,
        re.compile(r"<a:t\b[^>]*>"),
        re.compile(r"</a:t>"),
        normalize_spaces=normalize_spaces,
    )


def _scrub_odt_text(xml_text: str, *, normalize_spaces: bool = True) -> tuple[str, int, int]:
    """Run Layer A over ODF paragraph text (``text:p`` content, incl. spans).

    ``text:span``/``text:tab``/``text:s`` children live inside the paragraph,
    so cleaning the paragraph content covers the visible text. The markup
    itself is untouched: entities are decoded and re-encoded per text segment
    only — a whole-paragraph round trip would escape the nested markup.
    Linear scan (see _iter_tag_blocks).
    """
    from text_unicode import clean_text  # local import to avoid cycles

    removed = 0
    replaced = 0
    out = []
    last = 0
    for os_, oe, cs_, ce in _iter_tag_blocks(
        xml_text, re.compile(r"<text:p\b[^>]*>"), re.compile(r"</text:p>")
    ):
        open_tag = xml_text[os_:oe]
        inner = xml_text[oe:cs_]
        close_tag = xml_text[cs_:ce]
        parts = re.split(r"(<[^>]+>)", inner)
        new_parts: list[str] = []
        changed = False
        for segment in parts:
            if not segment or segment.startswith("<"):
                new_parts.append(segment)
                continue
            new_segment, stats = clean_text(
                _decode_xml_entities(segment), normalize_spaces=normalize_spaces
            )
            if stats["removed_count"] or stats["replaced_count"]:
                removed += stats["removed_count"]
                replaced += stats["replaced_count"]
                changed = True
                new_parts.append(_reencode_xml_text(new_segment))
            else:
                new_parts.append(segment)
        if not changed:
            continue
        new_para = "".join(new_parts)
        if (new_para[:1].isspace() or new_para[-1:].isspace()) and "xml:space" not in open_tag:
            open_tag = open_tag[:-1] + ' xml:space="preserve">'
        out.append(xml_text[last:os_])
        out.append(open_tag + new_para + close_tag)
        last = ce
    out.append(xml_text[last:])
    return "".join(out), removed, replaced


def _inspect_text_runs(
    xml_text: str, open_re: re.Pattern[str], close_re: re.Pattern[str]
) -> tuple[int, list[dict]]:
    """Layer A count/hits over text runs, mirroring _scrub_text_runs().

    Character references are decoded exactly as the scrub decodes them (a
    carrier written as ``&#x200B;`` is counted too), so /inspect reports the
    same carriers that /clean will remove for the format (#312).
    """
    from text_unicode import inspect_text  # local import to avoid cycles

    total = 0
    hits: list[dict] = []
    for _, oe, cs_, _ in _iter_tag_blocks(xml_text, open_re, close_re):
        ta = inspect_text(_decode_xml_entities(xml_text[oe:cs_])).to_dict()
        if ta["suspicious_total"]:
            total += ta["suspicious_total"]
            hits.extend(ta["hits"])
    return total, hits


def _inspect_odt_text(xml_text: str) -> tuple[int, list[dict]]:
    """Layer A count/hits over ODF paragraph text, mirroring _scrub_odt_text()."""
    from text_unicode import inspect_text  # local import to avoid cycles

    total = 0
    hits: list[dict] = []
    for _, oe, cs_, _ in _iter_tag_blocks(
        xml_text, re.compile(r"<text:p\b[^>]*>"), re.compile(r"</text:p>")
    ):
        for segment in re.split(r"(<[^>]+>)", xml_text[oe:cs_]):
            if not segment or segment.startswith("<"):
                continue
            ta = inspect_text(_decode_xml_entities(segment)).to_dict()
            if ta["suspicious_total"]:
                total += ta["suspicious_total"]
                hits.extend(ta["hits"])
    return total, hits


def _layer_a_body_part(name: str, fmt: str) -> bool:
    """True for the body parts a format's cleaner Layer-A scrubs.

    Mirrors the case-sensitive ``startswith``/``endswith`` check used by the
    OOXML/ODT cleaners, so /inspect selects exactly the archive entries /clean
    touches (a case-variant name such as ``WORD/document.XML`` is left alone by
    both) (#312).
    """
    if fmt == "odt":
        return name == "content.xml"
    if fmt == "docx":
        return name.startswith("word/") and name.endswith(".xml")
    if fmt == "xlsx":
        return name.startswith("xl/") and name.endswith(".xml")
    if fmt == "pptx":
        return name.startswith("ppt/") and name.endswith(".xml")
    return False


def _inspect_container_body_layer_a(
    data: bytes, fmt: str, budget: list[int] | None = None
) -> list[tuple[str, dict]]:
    """Per-part Layer A findings for the text runs clean_container() scrubs.

    Scans the same body parts the cleaner touches (``word``/``xl``/``ppt`` XML
    parts, or ``content.xml`` for ODT), so /inspect predicts /clean rather than
    contradicting it (#312).
    """
    out: list[tuple[str, dict]] = []
    if budget is None:
        budget = [0]
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if not _layer_a_body_part(info.filename, fmt):
                    continue
                text = _read_zip_member(zf, info, budget).decode("utf-8", errors="replace")
                if fmt == "docx":
                    total, hits = _inspect_text_runs(
                        text, re.compile(r"<w:t\b[^>]*>"), re.compile(r"</w:t>")
                    )
                elif fmt == "xlsx":
                    total, hits = _inspect_text_runs(
                        text, re.compile(r"<t\b[^>]*>"), re.compile(r"</t>")
                    )
                elif fmt == "pptx":
                    total, hits = _inspect_text_runs(
                        text, re.compile(r"<a:t\b[^>]*>"), re.compile(r"</a:t>")
                    )
                else:  # odt
                    total, hits = _inspect_odt_text(text)
                if total:
                    out.append((info.filename, {"suspicious_total": total, "hits": hits}))
    except _ZIP_PARSE_ERRORS:
        pass
    return out


def _prune_dangling_relationships(
    rels_name: str, raw: bytes, kept_names: set[str]
) -> tuple[bytes, int]:
    """Drop <Relationship> entries whose internal target part no longer exists.

    Removing a part (e.g. a customXml tree) must also remove the relationships
    that point at it, or the package is malformed: python-docx refuses to open
    it and Word offers to repair it. External relationships (``TargetMode``)
    and the package root (``Target="/"``) are left alone. ``rels_name`` is the
    archive member like ``word/_rels/document.xml.rels``; ``kept_names`` is the
    set of archive members that survive cleaning.
    """
    base = posixpath.dirname(posixpath.dirname(rels_name))
    text = raw.decode("utf-8", errors="replace")
    dropped = [0]

    def _target_attr(tag: str) -> str:
        # XML allows single- or double-quoted attribute values; matching only
        # double quotes made every single-quoted Relationship look like an
        # empty target, which resolved to the package base and got pruned —
        # corrupting packages from tools that emit single quotes (#130).
        m = re.search(r"\bTarget\s*=\s*([\"'])(.*?)\1", tag, re.I)
        return m.group(2) if m else ""

    def _drop(m: re.Match[str]) -> str:
        tag = m.group(0)
        if re.search(r"\bTargetMode\s*=", tag, re.I):
            return tag  # external (http / mailto / ...) — never pruned
        target = _target_attr(tag)
        if target.startswith("/"):
            resolved = posixpath.normpath(target.lstrip("/"))
        else:
            resolved = posixpath.normpath(posixpath.join(base, target))
        if resolved in ("", "."):
            return tag  # points at the package root
        if resolved in kept_names:
            return tag
        dropped[0] += 1
        return ""

    new = re.sub(r"<Relationship\b[^>]*/>", _drop, text, flags=re.I)
    return new.encode("utf-8"), dropped[0]


def _prune_odt_manifest_entries(raw: bytes, dropped: set[str]) -> tuple[bytes, int]:
    """Remove manifest file-entry elements pointing at dropped parts.

    ODF packages list every member in META-INF/manifest.xml; dropping a part
    (e.g. one carrying AI/C2PA markers) without removing its entry makes
    readers flag the package as damaged. full-path is matched
    attribute-order-independently, mirroring _prune_dangling_relationships.
    The package-root entry (full-path="/") and entries for surviving parts
    are left untouched.
    """
    text = raw.decode("utf-8", errors="replace")
    removed = [0]

    def _full_path(tag: str) -> str:
        # Same single/double-quote tolerance as _target_attr (#130).
        m = re.search(r"\bfull-path\s*=\s*([\"'])(.*?)\1", tag, re.I)
        return m.group(2) if m else ""

    def _drop(m: re.Match[str]) -> str:
        tag = m.group(0)
        path = posixpath.normpath(_full_path(tag).lstrip("/"))
        if path in ("", "."):
            return tag  # package root — never pruned
        if path in dropped:
            removed[0] += 1
            return ""
        return tag

    new = re.sub(r"<manifest:file-entry\b[^>]*/>", _drop, text, flags=re.I)
    return new.encode("utf-8"), removed[0]


def _prune_opf_manifest(raw: bytes, opf_name: str, dropped: set[str]) -> tuple[bytes, int]:
    """Remove OPF <item> entries pointing at dropped parts (and spine refs).

    The package document lists every content part; a dropped part that is
    still referenced makes EPUB readers reject the book. href values are
    relative to the package document directory, and <itemref> spine entries
    for removed items are pruned too so no idref dangles.
    """
    base = posixpath.dirname(opf_name)
    text = raw.decode("utf-8", errors="replace")
    removed = [0]
    removed_ids: set[str] = set()

    def _attr(tag: str, name: str) -> str:
        m = re.search(rf'\b{name}\s*=\s*"([^"]*)"', tag, re.I)
        return m.group(1) if m else ""

    def _drop_item(m: re.Match[str]) -> str:
        tag = m.group(0)
        href = _attr(tag, "href")
        if not href:
            return tag
        resolved = posixpath.normpath(posixpath.join(base, href))
        if resolved in dropped:
            removed[0] += 1
            item_id = _attr(tag, "id")
            if item_id:
                removed_ids.add(item_id)
            return ""
        return tag

    new = re.sub(r"<item\b[^>]*/>", _drop_item, text, flags=re.I)

    if removed_ids:

        def _drop_itemref(m: re.Match[str]) -> str:
            tag = m.group(0)
            if _attr(tag, "idref") in removed_ids:
                removed[0] += 1
                return ""
            return tag

        new = re.sub(r"<itemref\b[^>]*/>", _drop_itemref, new, flags=re.I)
    return new.encode("utf-8"), removed[0]


def _scrub_ooxml_zip(
    data: bytes, fmt: str, *, also_layer_a_text: bool = True, normalize_spaces: bool = True
) -> tuple[bytes, list[str]]:
    """Scrub provenance and Layer-A carriers from an OOXML zip container.

    Rewrites the archive in place: metadata/provenance parts (docProps,
    customXml, Content_Types overrides) are scrubbed or dropped, embedded media
    under ``word``/``xl``/``ppt`` are cleaned, and text runs are Layer-A scrubbed
    when requested. Binary docProps members (e.g. the thumbnail) are kept
    byte-safe. Returns the rewritten bytes and a list of human-readable actions.
    """
    actions: list[str] = []
    budget = [0]
    layer_removed = 0
    layer_replaced = 0
    kept: list[tuple[zipfile.ZipInfo, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zin:
        for info in zin.infolist():
            _check_zip_budget(info, budget)
            name = info.filename
            raw = _read_zip_member(zin, info, budget)

            # 1. Clean embedded media (PNG, JPEG, WebP, AVIF, HEIC, GIF, BMP, TIFF, SVG)
            if re.search(
                r"^(?:word|xl|ppt)/media/.+\.(png|jpe?g|webp|avif|heic|gif|bmp|tiff?|svg)$",
                name,
                re.I,
            ):
                img_fmt = detect_image_format(raw)
                sub_actions: list[str] = []
                cleaned_bytes = raw
                try:
                    if img_fmt == "png":
                        cleaned_bytes, sub_actions = strip_png(raw, strip_all_text=True)
                    elif img_fmt == "jpeg":
                        cleaned_bytes, sub_actions = strip_jpeg(raw, strip_all_app=True)
                    elif img_fmt == "webp":
                        cleaned_bytes, sub_actions = strip_webp(raw, strip_all_metadata=True)
                    elif img_fmt in ("avif", "heic"):
                        cleaned_bytes, sub_actions = strip_isobmff(
                            raw, img_fmt, strip_all_metadata=True
                        )
                    elif img_fmt == "gif":
                        cleaned_bytes, sub_actions = strip_gif(raw, strip_all_metadata=True)
                    elif img_fmt == "bmp":
                        cleaned_bytes, sub_actions = strip_bmp(raw, strip_all_metadata=True)
                    elif img_fmt == "tiff":
                        cleaned_bytes, sub_actions = strip_tiff(raw, strip_all_metadata=True)
                    elif name.lower().endswith(".svg") or raw.lstrip().startswith(b"<"):
                        cleaned_bytes, sub_actions = clean_svg(raw)
                except Exception:  # noqa: S110 - malformed embedded media; keep original part
                    pass
                if any("drop" in a.lower() for a in sub_actions) and cleaned_bytes != raw:
                    actions.append(f"clean embedded media in {name} ({', '.join(sub_actions[:2])})")
                    raw = cleaned_bytes
                kept.append((info, raw))
                continue

            # 2. Drop customXml trees
            if name.startswith("customXml/"):
                actions.append(f"drop part {name}")
                continue

            # 3. docProps/ provenance
            if name in DOCX_META_PARTS or name.startswith("docProps/"):
                if name.endswith("custom.xml"):
                    actions.append(f"drop part {name}")
                    continue
                if not name.lower().endswith(".xml"):
                    # A binary docProps member (e.g. docProps/thumbnail.jpeg)
                    # must stay byte-safe: decoding it as UTF-8 and re-encoding
                    # corrupts it into U+FFFD replacement bytes (#312).
                    kept.append((info, raw))
                    continue
                text = raw.decode("utf-8", errors="replace")
                new = text
                for tag, label in DOCX_SCRUB_FIELDS:
                    open_re = re.compile(rf"<{tag}\b[^>]*>", re.I)
                    close_re = re.compile(rf"</{tag}>", re.I)
                    out = []
                    last = 0
                    n = 0
                    for os_, oe, cs_, ce in _iter_tag_blocks(new, open_re, close_re):
                        out.append(new[last:os_])
                        out.append(new[os_:oe] + new[cs_:ce])
                        last = ce
                        if new[oe:cs_]:
                            n += 1
                            actions.append(f"scrub {name} field {label}")
                    if n:
                        out.append(new[last:])
                        new = "".join(out)
                raw = new.encode("utf-8")

            # 4. [Content_Types].xml overrides
            if name == "[Content_Types].xml":
                text = raw.decode("utf-8", errors="replace")
                new, n = re.subn(
                    r'<Override\b[^>]*PartName="/customXml/[^"]*"[^>]*/>',
                    "",
                    text,
                )
                if n:
                    actions.append(f"drop Content_Types customXml overrides x{n}")
                    raw = new.encode("utf-8")
                new, n = re.subn(
                    r'<Override\b[^>]*PartName="/docProps/custom\.xml"[^>]*/>',
                    "",
                    raw.decode("utf-8", errors="replace"),
                )
                if n:
                    actions.append(f"drop Content_Types custom.xml override x{n}")
                    raw = new.encode("utf-8")

            # 5. Layer A text runs
            if also_layer_a_text and name.endswith(".xml"):
                if fmt == "docx" and name.startswith("word/"):
                    text = raw.decode("utf-8", errors="replace")
                    new, r, rp = _scrub_docx_text(text, normalize_spaces=normalize_spaces)
                    if r or rp:
                        layer_removed += r
                        layer_replaced += rp
                        raw = new.encode("utf-8")
                elif fmt == "xlsx" and name.startswith("xl/"):
                    text = raw.decode("utf-8", errors="replace")
                    new, r, rp = _scrub_xlsx_text(text, normalize_spaces=normalize_spaces)
                    if r or rp:
                        layer_removed += r
                        layer_replaced += rp
                        raw = new.encode("utf-8")
                elif fmt == "pptx" and name.startswith("ppt/"):
                    text = raw.decode("utf-8", errors="replace")
                    new, r, rp = _scrub_pptx_text(text, normalize_spaces=normalize_spaces)
                    if r or rp:
                        layer_removed += r
                        layer_replaced += rp
                        raw = new.encode("utf-8")

            kept.append((info, raw))

    kept_names = {info.filename for info, _ in kept}
    final: list[tuple[zipfile.ZipInfo, bytes]] = []
    for info, raw in kept:
        part_raw = raw
        if info.filename.endswith(".rels"):
            part_raw, n = _prune_dangling_relationships(info.filename, raw, kept_names)
            if n:
                actions.append(f"prune dangling relationships x{n} in {info.filename}")
        final.append((info, part_raw))

    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info, raw in final:
            zout.writestr(info, raw)
    if layer_removed or layer_replaced:
        actions.append(f"layer A text: removed={layer_removed} replaced={layer_replaced}")
    if not actions:
        actions.append(f"no {fmt.upper()} metadata parts removed")
    return out_buf.getvalue(), actions


def clean_docx(
    data: bytes, *, also_layer_a_text: bool = True, normalize_spaces: bool = True
) -> tuple[bytes, list[str]]:
    return _scrub_ooxml_zip(
        data,
        "docx",
        also_layer_a_text=also_layer_a_text,
        normalize_spaces=normalize_spaces,
    )


def clean_xlsx(
    data: bytes, *, also_layer_a_text: bool = True, normalize_spaces: bool = True
) -> tuple[bytes, list[str]]:
    return _scrub_ooxml_zip(
        data,
        "xlsx",
        also_layer_a_text=also_layer_a_text,
        normalize_spaces=normalize_spaces,
    )


def clean_pptx(
    data: bytes, *, also_layer_a_text: bool = True, normalize_spaces: bool = True
) -> tuple[bytes, list[str]]:
    return _scrub_ooxml_zip(
        data,
        "pptx",
        also_layer_a_text=also_layer_a_text,
        normalize_spaces=normalize_spaces,
    )


def inspect_odt(data: bytes) -> tuple[bool, bool, list[str], dict]:
    findings: list[str] = []
    has_c2pa = False
    has_ai = False
    budget = [0]
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                _check_zip_budget(info, budget)
                raw = _read_zip_member(zf, info, budget)
                c2, ai, hits = _blob_hits(raw)
                if c2 or ai:
                    if c2:
                        has_c2pa = True
                    if ai:
                        has_ai = True
                    findings.append(f"{info.filename}: {', '.join(hits[:6])}")
            if "meta.xml" in zf.namelist():
                meta = _read_zip_member(zf, zf.getinfo("meta.xml"), budget).decode(
                    "utf-8", errors="replace"
                )
                if re.search(r"generator|claude|openai|anthropic|gemini", meta, re.I):
                    has_ai = True
                    findings.append("meta.xml generator-like fields")
    except _ZIP_PARSE_ERRORS as exc:
        if findings:
            findings.append(
                f"partial read of ODT zip ({exc.__class__.__name__}); "
                "evidence above survives, later members were not scanned"
            )
            return has_c2pa, has_ai or has_c2pa, findings, {}
        return False, False, ["not a valid ODT zip"], {}
    return has_c2pa, has_ai or has_c2pa, findings, {}


def clean_odt(
    data: bytes, *, also_layer_a_text: bool = True, normalize_spaces: bool = True
) -> tuple[bytes, list[str]]:
    actions: list[str] = []
    budget = [0]
    layer_removed = 0
    layer_replaced = 0
    kept: list[tuple[zipfile.ZipInfo, bytes]] = []
    dropped: set[str] = set()
    with zipfile.ZipFile(io.BytesIO(data)) as zin:
        for info in zin.infolist():
            _check_zip_budget(info, budget)
            name = info.filename
            raw = _read_zip_member(zin, info, budget)
            if name == "meta.xml":
                text = raw.decode("utf-8", errors="replace")
                # Drop meta:generator blocks (linear scan - lazy .*? is
                # quadratic on unclosed tags, see _iter_tag_blocks)
                new, n = _drop_tag_blocks(text, _ODT_GENERATOR_OPEN_RE, _ODT_GENERATOR_CLOSE_RE)
                if n:
                    actions.append("drop meta:generator")
                    text = new

                # scrub creator-like if AI (linear scan)
                def _is_ai_creator(block: str) -> bool:
                    return bool(AI_META_NAME_RE.search(block))

                new, n = _drop_blocks_if(
                    text, _DC_CREATOR_OPEN_RE, _DC_CREATOR_CLOSE_RE, _is_ai_creator
                )
                if n:
                    actions.extend(["scrub creator-like meta"] * n)
                    text = new
                raw = text.encode("utf-8")
            else:
                c2, ai, _ = _blob_hits(raw)
                if (c2 or ai) and name not in (
                    "content.xml",
                    "styles.xml",
                    "mimetype",
                    "META-INF/manifest.xml",
                ):
                    actions.append(f"drop part {name} (AI/C2PA markers)")
                    dropped.add(name)
                    continue
            # Layer A over the visible paragraph text of the body part.
            if also_layer_a_text and name == "content.xml":
                text = raw.decode("utf-8", errors="replace")
                new, r, rp = _scrub_odt_text(text, normalize_spaces=normalize_spaces)
                if r or rp:
                    layer_removed += r
                    layer_replaced += rp
                    raw = new.encode("utf-8")
            kept.append((info, raw))

    # Two-pass: rewrite META-INF/manifest.xml now that the dropped set is
    # known. The manifest precedes the dropped parts in the archive, so a
    # single pass cannot remove entries for parts dropped later in the loop.
    if dropped:
        rewritten: list[tuple[zipfile.ZipInfo, bytes]] = []
        for info, part_raw in kept:
            out_raw = part_raw
            if info.filename == "META-INF/manifest.xml":
                pruned, n = _prune_odt_manifest_entries(part_raw, dropped)
                if n:
                    actions.append(f"drop manifest entries x{n}")
                    out_raw = pruned
            rewritten.append((info, out_raw))
        kept = rewritten

    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info, raw in kept:
            zout.writestr(info, raw)
    if layer_removed or layer_replaced:
        actions.append(f"layer A text: removed={layer_removed} replaced={layer_replaced}")
    if not actions:
        actions.append("no ODT metadata removed")
    return out_buf.getvalue(), actions


# ---------------------------------------------------------------------------
# EPUB
# ---------------------------------------------------------------------------

_EPUB_MEDIA_RE = re.compile(r"\.(png|jpe?g|webp|avif|heic|gif|bmp|tiff?|svg)$", re.I)


def _epub_content_part(name: str) -> bool:
    """True for parts that carry visible content or structure and must never be dropped.

    XHTML/HTML, CSS, JS, navigation (ncx), the package document (opf), raster
    and vector media, fonts, the 'mimetype' part, relationship files, and the
    OCF control files are content or structure. Everything else (META-INF
    custom parts, stray XML at the archive root) is a candidate for dropping
    when it carries AI/C2PA markers.
    """
    low = name.lower()
    return bool(
        low == "mimetype"
        or low.endswith(".rels")
        or low
        in (
            "meta-inf/container.xml",
            "meta-inf/encryption.xml",
            "meta-inf/signatures.xml",
            "meta-inf/rights.xml",
        )
        or re.search(
            r"\.(xhtml|html?|css|js|ncx|opf|svg|png|jpe?g|webp|avif|heic|gif|bmp|tiff?|ttf|otf|woff2?)$",
            low,
        )
    )


def _epub_encrypted_parts(data: bytes) -> set[str]:
    """Return part names listed as encrypted in META-INF/encryption.xml (OCF)."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            if "META-INF/encryption.xml" not in zf.namelist():
                return set()
            budget: list[int] = [0]
            xml = _read_zip_member(zf, zf.getinfo("META-INF/encryption.xml"), budget).decode(
                "utf-8", errors="replace"
            )
    except (zipfile.BadZipFile, KeyError):
        return set()
    names: set[str] = set()
    for uri in re.findall(r'CipherReference\s+URI="([^"]+)"', xml, re.I):
        # The URI is relative to META-INF/ per OCF, but producers are not
        # consistent, so accept every plausible normalization.
        names.add(posixpath.normpath(posixpath.join("META-INF", uri)))
        names.add(posixpath.normpath(uri))
        if uri.startswith("../"):
            names.add(posixpath.normpath(uri[3:]))
    return names


def inspect_epub(data: bytes) -> tuple[bool, bool, list[str], dict]:
    findings: list[str] = []
    has_c2pa = False
    has_ai = False
    budget = [0]
    encrypted = _epub_encrypted_parts(data)
    names: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            for info in zf.infolist():
                _check_zip_budget(info, budget)
                name = info.filename
                if name in encrypted:
                    findings.append(f"{name}: encrypted content (skipped)")
                    continue
                raw = _read_zip_member(zf, info, budget)
                if name.lower().endswith((".xhtml", ".html", ".htm")):
                    text = raw.decode("utf-8", errors="surrogateescape")
                    c2, ai, sub, _ = inspect_html(text)
                    if c2:
                        has_c2pa = True
                    if ai:
                        has_ai = True
                    for f in sub:
                        findings.append(f"{name}: {f}")
                    continue
                if name.lower().endswith(".opf"):
                    text = raw.decode("utf-8", errors="surrogateescape")
                    if AI_META_NAME_RE.search(text):
                        has_ai = True
                        findings.append(f"{name}: AI-ish metadata in package document")
                    c2, ai, hits = _blob_hits(raw)
                    if c2 or ai:
                        has_c2pa = has_c2pa or c2
                        has_ai = has_ai or ai
                        findings.append(f"{name}: {', '.join(hits[:6])}")
                    continue
                c2, ai, hits = _blob_hits(raw)
                if c2 or ai:
                    has_c2pa = has_c2pa or c2
                    has_ai = has_ai or ai
                    findings.append(f"{name}: {', '.join(hits[:6])}")
    except _ZIP_PARSE_ERRORS as exc:
        # EPUB previously caught only BadZipFile, so a zlib error propagated
        # (crashing the scan) while a CRC failure discarded everything —
        # neither is right. Same partial-read contract as the OOXML/ODT
        # inspectors (#164).
        if findings:
            findings.append(
                f"partial read of EPUB zip ({exc.__class__.__name__}); "
                "evidence above survives, later members were not scanned"
            )
            return has_c2pa, has_ai or has_c2pa, findings, {"parts": len(names)}
        return False, False, ["not a valid EPUB zip"], {}
    return has_c2pa, has_ai or has_c2pa, findings, {"parts": len(names)}


# Open/close tag halves for the linear OPF scans in _scrub_epub_opf.
_OPF_META_OPEN_RE = re.compile(r"<meta\b[^>]*>", re.I)
_OPF_META_CLOSE_RE = re.compile(r"</meta\s*>", re.I)
_DC_FIELDS_OPEN_RE = re.compile(
    r"<(dc:(?:creator|contributor|publisher|description|rights|source))\b[^>]*>",
    re.I,
)
_DC_FIELDS_CLOSE_RE = re.compile(
    r"</(dc:(?:creator|contributor|publisher|description|rights|source))\s*>",
    re.I,
)


def _scrub_epub_opf(text: str) -> tuple[str, list[str]]:
    """Scrub AI-ish metadata from the EPUB package document (OPF)."""
    actions: list[str] = []

    def _meta(m: re.Match[str]) -> str:
        tag = m.group(0)
        if AI_META_NAME_RE.search(tag):
            actions.append("drop OPF meta tag")
            return ""
        return tag

    new = re.sub(r"<meta\b[^>]*/>", _meta, text, flags=re.I)

    # Block-form <meta>...</meta> (linear scan; the lazy .*? form is
    # quadratic on unclosed opening tags, see _iter_tag_blocks)
    def _meta_block_is_ai(block: str) -> bool:
        if AI_META_NAME_RE.search(block):
            actions.append("drop OPF meta tag")
            return True
        return False

    # The predicate itself records the per-block action, so the count is unused.
    new, _n = _drop_blocks_if(new, _OPF_META_OPEN_RE, _OPF_META_CLOSE_RE, _meta_block_is_ai)

    # dc:... provenance fields. The original pattern paired each opening tag
    # with its backreferenced closing tag via a lazy .*? - quadratic on
    # unclosed openings. Pair opens with same-name closes directly: linear,
    # exact same match semantics (first same-name close at/after the open).
    out = []
    last = 0
    dc_closes: dict[str, list[re.Match[str]]] = {}
    for m in re.finditer(_DC_FIELDS_CLOSE_RE, new):
        dc_closes.setdefault(m.group(1), []).append(m)
    ptr: dict[str, int] = {name: 0 for name in dc_closes}
    for om in re.finditer(_DC_FIELDS_OPEN_RE, new):
        name = om.group(1)
        if om.start() < last:
            continue
        closes = dc_closes.get(name, ())
        i = ptr.get(name, 0)
        while i < len(closes) and closes[i].start() < om.end():
            i += 1
        ptr[name] = i
        if i >= len(closes):
            continue  # no same-name close - block never matches (kept)
        cm = closes[i]
        block = new[om.start() : cm.end()]
        if AI_META_NAME_RE.search(block):
            out.append(new[last : om.start()])
            out.append(f"<{name}/>")
            last = cm.end()
            actions.append(f"scrub {name} (AI vendor name)")
    if last:
        out.append(new[last:])
        new = "".join(out)

    if not actions:
        actions.append("no OPF metadata removed")
    return new, actions


def clean_epub(
    data: bytes, *, also_layer_a_text: bool = True, normalize_spaces: bool = True
) -> tuple[bytes, list[str]]:
    """Rewrite the EPUB: scrub OPF metadata, XHTML meta/JSON-LD, and Layer A.

    Embedded raster/SVG media get their own metadata stripped; structural and
    OCF control parts (mimetype, container.xml, encryption.xml, relationships,
    fonts) are kept so the book stays readable. Parts listed in
    META-INF/encryption.xml are copied verbatim — their bytes are ciphertext,
    so any rewrite would corrupt them. Non-content parts carrying AI/C2PA
    markers are dropped, and the OPF manifest is pruned afterwards so no
    dropped part stays referenced.
    """
    from text_unicode import clean_text  # local import to avoid cycles

    actions: list[str] = []
    budget = [0]
    layer_removed = 0
    layer_replaced = 0
    encrypted = _epub_encrypted_parts(data)
    kept: list[tuple[zipfile.ZipInfo, bytes]] = []
    dropped: set[str] = set()
    with zipfile.ZipFile(io.BytesIO(data)) as zin:
        for info in zin.infolist():
            _check_zip_budget(info, budget)
            name = info.filename
            raw = _read_zip_member(zin, info, budget)
            low = name.lower()

            # Encrypted parts are opaque ciphertext: pass through untouched.
            if name in encrypted:
                kept.append((info, raw))
                continue

            # 1. Embedded raster / vector media: strip metadata
            if _EPUB_MEDIA_RE.search(low):
                img_fmt = detect_image_format(raw)
                sub_actions: list[str] = []
                cleaned = raw
                try:
                    if img_fmt == "png":
                        cleaned, sub_actions = strip_png(raw, strip_all_text=True)
                    elif img_fmt == "jpeg":
                        cleaned, sub_actions = strip_jpeg(raw, strip_all_app=True)
                    elif img_fmt == "webp":
                        cleaned, sub_actions = strip_webp(raw, strip_all_metadata=True)
                    elif img_fmt in ("avif", "heic"):
                        cleaned, sub_actions = strip_isobmff(raw, img_fmt, strip_all_metadata=True)
                    elif img_fmt == "gif":
                        cleaned, sub_actions = strip_gif(raw, strip_all_metadata=True)
                    elif img_fmt == "bmp":
                        cleaned, sub_actions = strip_bmp(raw, strip_all_metadata=True)
                    elif img_fmt == "tiff":
                        cleaned, sub_actions = strip_tiff(raw, strip_all_metadata=True)
                    elif low.endswith(".svg") or raw.lstrip().startswith(b"<"):
                        cleaned, sub_actions = clean_svg(raw)
                except Exception:  # noqa: S110 - malformed embedded media; keep original
                    pass
                if any("drop" in a.lower() for a in sub_actions) and cleaned != raw:
                    actions.append(f"clean embedded media in {name} ({', '.join(sub_actions[:2])})")
                    raw = cleaned
                kept.append((info, raw))
                continue

            # 2. XHTML content: strip AI meta/JSON-LD, then Layer A
            if low.endswith((".xhtml", ".html", ".htm")):
                text = raw.decode("utf-8", errors="surrogateescape")
                text, sub_actions = clean_html(text)
                if sub_actions and sub_actions != ["no HTML AI meta removed"]:
                    actions.append(f"{name}: {', '.join(sub_actions[:2])}")
                if also_layer_a_text:
                    text2, stats = clean_text(text, normalize_spaces=normalize_spaces)
                    if stats["removed_count"] or stats["replaced_count"]:
                        layer_removed += stats["removed_count"]
                        layer_replaced += stats["replaced_count"]
                        text = text2
                raw = text.encode("utf-8", errors="surrogateescape")
                kept.append((info, raw))
                continue

            # 3. Package document (OPF): scrub AI-ish metadata
            if low.endswith(".opf"):
                text = raw.decode("utf-8", errors="surrogateescape")
                new_text, sub_actions = _scrub_epub_opf(text)
                if sub_actions and sub_actions != ["no OPF metadata removed"]:
                    actions.extend(f"{name}: {a}" for a in sub_actions)
                raw = new_text.encode("utf-8", errors="surrogateescape")
                kept.append((info, raw))
                continue

            # 4. Other parts: drop non-content parts carrying AI/C2PA markers
            c2, ai, _hits = _blob_hits(raw)
            if (c2 or ai) and not _epub_content_part(name):
                actions.append(f"drop part {name} (AI/C2PA markers)")
                dropped.add(name)
                continue

            # 5. mimetype must stay first and stored — it already is, since we
            #    write in the original entry order; force stored compression.
            if low == "mimetype":
                info.compress_type = zipfile.ZIP_STORED
            kept.append((info, raw))

    # Two-pass: prune the OPF manifest now that the dropped set is known.
    if dropped:
        rewritten: list[tuple[zipfile.ZipInfo, bytes]] = []
        for info, part_raw in kept:
            out_raw = part_raw
            if info.filename.lower().endswith(".opf"):
                pruned, n = _prune_opf_manifest(part_raw, info.filename, dropped)
                if n:
                    actions.append(f"prune OPF manifest entries x{n}")
                    out_raw = pruned
            rewritten.append((info, out_raw))
        kept = rewritten

    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info, raw in kept:
            zout.writestr(info, raw)

    if layer_removed or layer_replaced:
        actions.append(f"layer A text: removed={layer_removed} replaced={layer_replaced}")
    if not actions:
        actions.append("no EPUB metadata removed")
    return out_buf.getvalue(), actions


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

# Open/close halves for the linear XMP packet scan below. The lazy
# ".*?</\?xpacket end..." form was quadratic on a flood of "<?xpacket begin"
# markers with no end marker.
_XMP_PACKET_OPEN_RE = re.compile(rb"<\?xpacket begin", re.I)
_XMP_PACKET_CLOSE_RE = re.compile(rb"<\?xpacket end[^?]*\?>", re.I)
_PDF_STREAM_OPEN_RE = re.compile(rb"stream\r?\n")
_PDF_STREAM_CLOSE_RE = re.compile(rb"endstream")


def _pdf_structured_blob(data: bytes) -> bytes:
    """Return PDF bytes with stream payloads removed, plus XMP packets.

    Stream payloads are often compressed binary where an AI-marker byte
    sequence (e.g. "AIGC") can occur by chance. Scanning only dictionaries and
    XMP packets avoids treating those collisions as metadata findings.
    """
    parts = []
    last = 0
    for os_, _oe, _cs, ce in _iter_tag_blocks(data, _PDF_STREAM_OPEN_RE, _PDF_STREAM_CLOSE_RE):
        parts.append(data[last:os_])
        parts.append(b"stream endstream")
        last = ce
    parts.append(data[last:])
    no_streams = b"".join(parts)
    xmp = b"\n".join(
        data[os_:ce]
        for os_, _oe, _cs, ce in _iter_tag_blocks(data, _XMP_PACKET_OPEN_RE, _XMP_PACKET_CLOSE_RE)
    )
    return no_streams + b"\n" + xmp


def inspect_pdf(path: Path, data: bytes) -> tuple[bool, bool, list[str], dict]:
    findings: list[str] = []
    has_c2pa, has_ai, hits = _blob_hits(_pdf_structured_blob(data))
    findings.extend(f"pdf-structured:{h}" for h in hits)
    # XMP packet scan
    xmp_blob = b"\n".join(
        data[os_:ce]
        for os_, _oe, _cs, ce in _iter_tag_blocks(data, _XMP_PACKET_OPEN_RE, _XMP_PACKET_CLOSE_RE)
    )
    if xmp_blob:
        findings.append("XMP packet present")
        has_ai = has_ai or bool(
            re.search(
                rb"digitalSourceType|trainedAlgorithmicMedia|SoftwareAgent|c2pa",
                xmp_blob,
                re.I,
            )
        )
    tools = run_optional_tools(path)
    ct = tools.get("c2patool") or {}
    if ct.get("has_manifest"):
        has_c2pa = True
        findings.append("c2patool reports C2PA-related manifest")
    probe_note = c2patool_probe_note(tools)
    if probe_note:
        findings.append(probe_note)
    return has_c2pa, has_ai or has_c2pa, findings, {"tools": tools}


def _blank_xmp_packets(data: bytes) -> tuple[bytes, int]:
    """Overwrite XMP packets with spaces, leaving every byte offset intact.

    Deleting the bytes shifts every xref offset and stream /Length that follows,
    which is how the stdlib path could hand back a PDF that no longer parses --
    and with `deep_images="never"` nothing downstream rebuilds it. Blanking
    leaves the file structurally identical: XMP packets are padded with
    whitespace by design, so a whitespace-only one is unremarkable.
    """
    out = bytearray(data)
    count = 0
    for open_start, _oe, _cs, close_end in _iter_tag_blocks(
        data, _XMP_PACKET_OPEN_RE, _XMP_PACKET_CLOSE_RE
    ):
        out[open_start:close_end] = b" " * (close_end - open_start)
        count += 1
    return bytes(out), count


def _pdf_structural_rewrite(
    dest: Path, actions: list[str], deadline: "_Deadline | None" = None
) -> bool:
    """Rebuild a PDF so unreferenced objects are dropped.

    exiftool's PDF edits are incremental, so freed metadata objects survive in
    the byte stream. qpdf re-serializes from the object graph, which is what
    actually removes them. No-op (with a warning) when qpdf is absent.
    """
    qpdf = which("qpdf")
    if not qpdf:
        actions.append(
            "warning: exiftool PDF edits are incremental — the original metadata "
            "bytes remain recoverable; install qpdf for a structural rewrite"
        )
        return False

    if deadline is not None and deadline.spent():
        actions.append(
            "qpdf structural rewrite skipped: clean budget exhausted; "
            "metadata bytes may remain recoverable"
        )
        return False

    tmp = dest.with_name(dest.name + ".qpdf-tmp")
    try:
        r = subprocess.run(
            [qpdf, "--linearize", "--", safe_arg(str(dest)), safe_arg(str(tmp))],
            capture_output=True,
            text=True,
            timeout=(_QPDF_TIMEOUT if deadline is None else deadline.timeout(_QPDF_TIMEOUT)),
            check=False,
            preexec_fn=subprocess_preexec_fn,
            creationflags=subprocess_creationflags,
        )
    except Exception as e:
        tmp.unlink(missing_ok=True)
        actions.append(f"qpdf rewrite failed: {e}; metadata bytes may remain recoverable")
        return False

    # qpdf exit codes: 0 = clean, 3 = succeeded with warnings (output written).
    if r.returncode in (0, 3) and tmp.is_file() and tmp.stat().st_size > 0:
        tmp.replace(dest)
        actions.append(f"qpdf --linearize structural rewrite (rc={r.returncode})")
        return True

    tmp.unlink(missing_ok=True)
    actions.append(
        f"qpdf rewrite skipped (rc={r.returncode}); metadata bytes may remain recoverable"
    )
    return False


# APPn segments a JPEG can carry inside a PDF image XObject. APP0 (JFIF) is
# structural and APP2 (ICC) decides how the colours are interpreted, so neither
# counts as metadata to strip; APP1 (EXIF/XMP), APP11 (JUMBF/C2PA) and APP13
# (Photoshop resources) do.
_JPEG_METADATA_MARKERS = frozenset({0xE1, 0xEB, 0xED})
_JPEG_SCAN_END = frozenset({0xDA, 0xD9})


def _iter_jpeg_segments(blob: bytes, start: int):
    """Yield (marker, payload) for each header segment of one JPEG.

    A marker may be preceded by any number of 0xFF fill bytes, so the run has to
    be consumed before reading it. Taking the first 0xFF as the marker instead
    shifts the length read by one byte, and the walk then either stops early or
    wanders -- either way a provenance segment further along goes unseen.
    """
    i = start + 2
    limit = len(blob)
    while i + 3 < limit:
        if blob[i] != 0xFF:
            return
        while i + 1 < limit and blob[i + 1] == 0xFF:
            i += 1
        if i + 1 >= limit:
            return
        marker = blob[i + 1]
        if marker in _JPEG_SCAN_END:
            return
        if 0xD0 <= marker <= 0xD8:
            i += 2
            continue
        if i + 4 > limit:
            return
        length = int.from_bytes(blob[i + 2 : i + 4], "big")
        if length < 2:
            return
        yield marker, blob[i + 4 : i + 2 + length]
        i += 2 + length


def _jpeg_carries_metadata(blob: bytes, start: int) -> bool:
    """True when this JPEG holds a metadata APPn segment of any kind."""
    return any(marker in _JPEG_METADATA_MARKERS for marker, _ in _iter_jpeg_segments(blob, start))


def _jpeg_carries_provenance(blob: bytes, start: int) -> bool:
    """True when this JPEG holds JUMBF or a provenance XMP packet."""
    for marker, payload in _iter_jpeg_segments(blob, start):
        if marker in _JPEG_PROVENANCE_MARKERS:
            lowered = payload.lower()
            if any(sig in lowered for sig in _PROVENANCE_SIGNATURES):
                return True
    return False


def _any_jpeg(data: bytes, predicate) -> bool:
    i = 0
    while True:
        start = data.find(b"\xff\xd8\xff", i)
        if start < 0:
            return False
        if predicate(data, start):
            return True
        i = start + 2


# APP11 carries JUMBF, which is how C2PA travels; an APP1 XMP packet can name
# the provenance chain instead. Ordinary EXIF is deliberately not in this set.
_JPEG_PROVENANCE_MARKERS = frozenset({0xE1, 0xEB})
_PROVENANCE_SIGNATURES = (b"jumbf", b"c2pa", b"contentauth", b"dcterms:provenance")


def embedded_provenance_present(data: bytes) -> bool:
    """True when a JPEG inside *data* carries C2PA/JUMBF or provenance XMP.

    inspect_pdf's byte scan skips stream payloads, so a manifest that lives only
    inside an image XObject is invisible to it unless c2patool happens to be
    installed. `auto` decides whether to re-distill from exactly that scan, so
    without this a machine without c2patool would quietly leave embedded C2PA in
    place -- the case this whole pass exists for.

    Ordinary EXIF is not provenance and must not appear here: `auto` promises
    not to spend a re-distill on camera and editor traces.
    """
    return _any_jpeg(data, _jpeg_carries_provenance)


def embedded_image_metadata_present(data: bytes) -> bool:
    """True when a JPEG inside *data* still carries a metadata APPn segment.

    Ghostscript's JPEG pass-through copies the image bytes verbatim, so EXIF
    that lives inside the stream survives a lossless deep pass. Detecting it
    needs no external tool: the segment headers are readable straight from the
    PDF's bytes.
    """
    return _any_jpeg(data, _jpeg_carries_metadata)


DEEP_IMAGE_MODES = frozenset({"auto", "always", "lossless", "never"})

# One PDF can run two Ghostscript passes and three exiftool/qpdf pairs. Left to
# their own timeouts that is about nineteen minutes of wall clock, on a request
# thread, and /clean/batch works through its files one after another. Budget the
# whole clean instead and hand each child whatever is left of it.
PDF_CLEAN_BUDGET_SECONDS = float(os.environ.get("WATERMARKS_PDF_CLEAN_BUDGET", "420"))
_EXIFTOOL_TIMEOUT = 60.0
_QPDF_TIMEOUT = 120.0
_GHOSTSCRIPT_TIMEOUT = 300.0


class _Deadline:
    """What is left of a shared budget, clamped to each tool's own ceiling."""

    def __init__(self, budget: float) -> None:
        self._end = time.monotonic() + budget

    def remaining(self) -> float:
        return max(0.0, self._end - time.monotonic())

    def spent(self) -> bool:
        return self.remaining() <= 0.0

    def timeout(self, cap: float) -> float:
        # Never hand subprocess 0: it would raise instead of running.
        return max(1.0, min(cap, self.remaining()))


_GHOSTSCRIPT_NAMES = ("gs", "gswin64c", "gswin32c")


def which_ghostscript() -> str | None:
    """Locate a Ghostscript binary (POSIX ``gs``, Windows ``gswin64c``)."""
    for name in _GHOSTSCRIPT_NAMES:
        found = which(name)
        if found:
            return found
    return None


def _exiftool_strip(
    exiftool: str, dest: Path, actions: list[str], deadline: "_Deadline | None" = None
) -> bool:
    """Run ``exiftool -all=`` over *dest* in place, recording the outcome."""
    if deadline is not None and deadline.spent():
        actions.append("exiftool skipped: clean budget exhausted")
        return False
    try:
        r = subprocess.run(
            [exiftool, "-all=", "-overwrite_original", safe_arg(str(dest))],
            capture_output=True,
            text=True,
            timeout=(
                _EXIFTOOL_TIMEOUT if deadline is None else deadline.timeout(_EXIFTOOL_TIMEOUT)
            ),
            check=False,
            preexec_fn=subprocess_preexec_fn,
            creationflags=subprocess_creationflags,
        )
        actions.append(f"exiftool -all= (rc={r.returncode})")
        return r.returncode == 0
    except Exception as e:
        actions.append(f"exiftool failed: {e}")
        return False


def _pdf_deep_image_clean(
    dest: Path,
    actions: list[str],
    *,
    reencode: bool = False,
    deadline: "_Deadline | None" = None,
) -> bool:
    """Drop metadata carried *inside* embedded images by re-distilling.

    ``exiftool -all=`` and the qpdf rewrite both stop at document level. EXIF,
    Photoshop resource blocks and C2PA manifests that live in an image XObject
    survive them untouched. Ghostscript's pdfwrite rebuilds the page from the
    object graph, and with JPEG pass-through the compressed image data is
    copied byte-for-byte -- pixels are preserved while the metadata wrapped
    around them is dropped. No-op (with a warning) when Ghostscript is absent.

    Pass-through has one blind spot: a marker living *inside* the JPEG itself
    (EXIF in APP1, a C2PA/JUMBF manifest in APP11) travels with the bytes it is
    attached to. ``reencode=True`` turns pass-through off so the image is
    recompressed and those segments are dropped -- effective, but lossy, so
    callers escalate to it only when a marker provably survived the cheap pass.
    """
    if deadline is not None and deadline.spent():
        actions.append("deep image pass skipped: clean budget exhausted")
        return False

    gs = which_ghostscript()
    if not gs:
        actions.append(
            "warning: metadata inside embedded images left in place; "
            "install ghostscript for the deep image pass"
        )
        return False

    # A predictable path beside *dest* is a symlink-swap target: Ghostscript
    # opens it directly, which would sidestep the no-symlink contract that
    # safe_write_bytes upholds. Write into a private directory instead and
    # hand the validated bytes over through the safe writer.
    with tempfile.TemporaryDirectory(prefix="wr-gs-") as staging:
        tmp = Path(staging) / "deep-image.pdf"
        return _run_ghostscript(gs, dest, tmp, actions, reencode=reencode, deadline=deadline)


def _run_ghostscript(
    gs: str,
    dest: Path,
    tmp: Path,
    actions: list[str],
    *,
    reencode: bool,
    deadline: "_Deadline | None" = None,
) -> bool:
    try:
        r = subprocess.run(
            [
                gs,
                "-dBATCH",
                "-dNOPAUSE",
                "-dQUIET",
                "-dSAFER",
                "-sDEVICE=pdfwrite",
                "-dPDFSETTINGS=/prepress",
                "-dAutoRotatePages=/None",
                f"-dPassThroughJPEGImages={'false' if reencode else 'true'}",
                # JPEG2000 streams get the same treatment; other codecs (Flate,
                # CCITT) are re-encoded by pdfwrite and pass-through does not
                # apply to them, which the docs say plainly.
                f"-dPassThroughJPXImages={'false' if reencode else 'true'}",
                "-dDownsampleColorImages=false",
                "-dDownsampleGrayImages=false",
                "-dDownsampleMonoImages=false",
                f"-sOutputFile={tmp}",
                safe_arg(str(dest)),
            ],
            capture_output=True,
            text=True,
            timeout=(
                _GHOSTSCRIPT_TIMEOUT if deadline is None else deadline.timeout(_GHOSTSCRIPT_TIMEOUT)
            ),
            check=False,
            preexec_fn=subprocess_preexec_fn,
            creationflags=subprocess_creationflags,
        )
    except Exception as e:
        actions.append(f"ghostscript deep image pass failed: {e}")
        return False

    if r.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
        actions.append(
            f"ghostscript deep image pass skipped (rc={r.returncode}); "
            "metadata inside embedded images may remain"
        )
        return False

    safe_write_bytes(dest, tmp.read_bytes())
    actions.append(
        "ghostscript pdfwrite deep image pass, images re-encoded (rc=0)"
        if reencode
        else "ghostscript pdfwrite deep image pass, images passed through (rc=0)"
    )
    return True


def clean_pdf(path: Path, dest: Path, *, deep_images: str = "auto") -> tuple[list[str], dict]:
    """Best-effort PDF clean. Prefers exiftool; falls back to XMP strip warning.

    ``deep_images`` controls the Ghostscript re-distill that reaches metadata
    inside embedded images:

    * ``"auto"`` (default) -- run it only when the document-level strip left
      AI/C2PA markers behind, and recompress the images if a marker survives
      even that.
    * ``"always"`` -- run it for every PDF, which also clears non-AI traces
      such as camera and editor EXIF, with the same escalation.
    * ``"lossless"`` -- deep pass without the recompressing escalation, so
      image data is never touched.
    * ``"never"`` -- skip it.
    """
    actions: list[str] = []
    data = path.read_bytes()
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Silently treating a typo as "auto" would turn a request for lossless
    # cleaning into one that may recompress, so refuse it instead.
    if deep_images not in DEEP_IMAGE_MODES:
        raise ValueError(
            f"deep_images must be one of {sorted(DEEP_IMAGE_MODES)}, got {deep_images!r}"
        )

    deadline = _Deadline(PDF_CLEAN_BUDGET_SECONDS)

    exiftool = which("exiftool")
    rewritten = False

    def _stdlib_document_strip(source: bytes) -> str:
        """Apply the byte-preserving fallback and return its reported mode."""
        blanked, n = _blank_xmp_packets(source)
        safe_write_bytes(dest, blanked if n else source)
        if n:
            actions.append(f"blanked XMP xpacket x{n} (degraded; byte offsets preserved)")
            actions.append("warning: pure-stdlib PDF strip is best-effort; prefer exiftool")
            return "stdlib-xmp"
        actions.append(
            "no PDF cleaner available (install exiftool for reliable metadata "
            "strip); document-level metadata left as-is"
        )
        return "copy"

    if exiftool:
        safe_write_bytes(dest, data)
        if not _exiftool_strip(exiftool, dest, actions, deadline):
            # An installed exiftool can still reject malformed or unsupported
            # PDFs. Fall back to the same byte-preserving XMP scrub as the
            # no-exiftool path, and report the result as degraded instead of
            # claiming the optional cleaner completed.
            document_mode = _stdlib_document_strip(data)
            exiftool = None
        else:
            # exiftool writes PDFs *incrementally*: it appends a
            # %BeginExifToolUpdate block that frees the Info object and drops
            # /Info from the trailer, but the original metadata bytes stay in the
            # file verbatim and are trivially recoverable (exiftool itself can
            # revert them with -PDF-update:all=). A structural rewrite is what
            # actually drops the now-unreferenced objects.
            rewritten = _pdf_structural_rewrite(dest, actions, deadline)
            document_mode = "exiftool"
    else:
        # Degraded document-level strip: obvious XMP packets and nothing else.
        # The deep-image ladder below still runs -- Ghostscript reaches metadata
        # inside image XObjects on its own, and gating that on exiftool left the
        # only tool that can do that job unused.
        document_mode = _stdlib_document_strip(data)

    # Document-level strip is done. Metadata inside embedded images is out
    # of reach from here, so decide whether to re-distill.
    mode = deep_images

    def _markers_left() -> bool:
        current = dest.read_bytes()
        res_c2pa, res_ai, _f, _d = inspect_pdf(dest, current)
        # inspect_pdf excludes stream payloads, so ask the streams directly too:
        # without c2patool nothing else would notice a manifest that only exists
        # inside an image.
        return bool(res_c2pa or res_ai) or embedded_provenance_present(current)

    def _settle(ran: bool) -> None:
        # pdfwrite stamps its own /Producer, and exiftool edits PDFs
        # incrementally, so re-serialize once more after removing it.
        nonlocal document_mode, exiftool, rewritten
        if not ran:
            return
        if exiftool:
            current = dest.read_bytes()
            if _exiftool_strip(exiftool, dest, actions, deadline):
                rewritten = _pdf_structural_rewrite(dest, actions, deadline) or rewritten
            else:
                document_mode = _stdlib_document_strip(current)
                exiftool = None
                rewritten = False
        else:
            # Trading a vendor mark for a Ghostscript one is still worth it,
            # but the caller should not have to discover the swap.
            actions.append(
                "warning: the re-distill stamped its own /Producer and there "
                "is no exiftool to remove it"
            )

    deep_ran = False
    reencoded = False
    if mode == "never":
        pass
    elif mode in ("always", "lossless") or _markers_left():
        if mode == "auto":
            actions.append(
                "residual AI/C2PA markers after document-level strip "
                "(embedded image suspected); running deep image pass"
            )
        deep_ran = _pdf_deep_image_clean(dest, actions, deadline=deadline)
        _settle(deep_ran)
        # Anything inside the JPEG itself rides along with a passed-through
        # stream. Recompressing is the only way left to shift it, so spend
        # the quality only when something demonstrably survived: AI/C2PA
        # markers always count, and under `always` -- which promises to
        # clear camera and editor traces too -- so does ordinary EXIF.
        #
        # Two things make asking pointless. In `lossless` the answer is
        # discarded, and _markers_left() runs a full inspect_pdf that spawns
        # tools of its own. And if rung 1 never ran -- no Ghostscript, or the
        # budget is gone -- rung 2 cannot run either, and asking would only
        # append the same warning twice.
        survived = (
            mode != "lossless"
            and deep_ran
            and (
                _markers_left()
                or (mode == "always" and embedded_image_metadata_present(dest.read_bytes()))
            )
        )
        if survived:
            actions.append(
                "metadata survived the lossless pass (it is inside the image "
                "stream); escalating to a re-encoding pass"
            )
            reencoded = _pdf_deep_image_clean(dest, actions, reencode=True, deadline=deadline)
            deep_ran = deep_ran or reencoded
            _settle(reencoded)
    else:
        actions.append(
            "deep image pass not needed for AI/C2PA markers; pass "
            'deep_images="always" to also clear non-AI EXIF inside images'
        )

    c2patool = which("c2patool")
    # c2patool does not always strip; leave note
    if c2patool:
        actions.append("c2patool available for inspect; strip via exiftool/re-export")

    meta: dict[str, Any] = {
        "mode": document_mode,
        "structural_rewrite": rewritten,
        "deep_images": mode,
        "deep_image_pass": deep_ran,
        "images_reencoded": reencoded,
    }
    if not exiftool:
        # The deep pass may well have run, but the document-level strip was
        # the stdlib one, so the result is still best-effort.
        meta["degraded"] = True
    return actions, meta


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------


def inspect_container(path: Path, *, data: bytes | None = None) -> ContainerInspectReport:
    """Inspect a container for provenance, AI metadata, and Layer-A carriers.

    Delegates to the format-specific inspector and unions in a Layer-A scan of
    the visible text body for exactly the formats ``clean_container()`` scrubs,
    so /inspect predicts what /clean removes rather than contradicting it.
    """
    if data is None:
        data = path.read_bytes()
    fmt = detect_container_format(path, data)
    tools: dict[str, Any] = {}
    details: dict[str, Any] = {}
    # One shared ZIP decompression budget for the OOXML inspectors and the body
    # Layer-A scan, so they cumulatively enforce the cap instead of each
    # resetting it: an archive with ~128 MiB of metadata/media plus ~128 MiB of
    # body XML must not decompress both parts (#312 review). ODT/EPUB use their
    # own budgets because their inspectors already read every member, so sharing
    # them with the body re-scan would charge content.xml twice and falsely
    # reject a large but legitimate file.
    zip_budget: list[int] = [0]

    if fmt == "svg":
        has_c2pa, has_ai, findings, details = inspect_svg(data)
    elif fmt == "pdf":
        has_c2pa, has_ai, findings, details = inspect_pdf(path, data)
        tools = details.pop("tools", {})
    elif fmt == "docx":
        has_c2pa, has_ai, findings, details = inspect_docx(data, zip_budget)
    elif fmt == "xlsx":
        has_c2pa, has_ai, findings, details = inspect_xlsx(data, zip_budget)
    elif fmt == "pptx":
        has_c2pa, has_ai, findings, details = inspect_pptx(data, zip_budget)
    elif fmt == "odt":
        # inspect_odt reads the whole archive (including content.xml), so it
        # keeps its own budget: sharing the body-scan budget with it would
        # charge content.xml twice (byte for byte) and falsely reject a large
        # but legitimate ODT (#312 review).
        has_c2pa, has_ai, findings, details = inspect_odt(data)
    elif fmt == "epub":
        has_c2pa, has_ai, findings, details = inspect_epub(data)
    elif fmt == "html":
        # surrogateescape, not replace: clean_container() decodes the same way,
        # and U+FFFD substitutions would make the two disagree on the counts.
        body = data.decode("utf-8", errors="surrogateescape")
        has_c2pa, has_ai, findings, details = inspect_html(body)
    elif fmt == "markdown":
        body = data.decode("utf-8", errors="surrogateescape")
        has_c2pa, has_ai, findings, details = inspect_markdown(body)
    else:
        has_c2pa, has_ai, findings = False, False, [f"unsupported container: {fmt}"]
        details = {"unsupported": True}

    # Layer A body scan for exactly the formats clean_container() scrubs, so
    # inspect predicts clean rather than contradicting it.
    layer_a_total = 0
    layer_a_hits: list[dict] = []
    if fmt in ("markdown", "html"):
        from text_unicode import inspect_text  # local import to avoid cycles

        ta = inspect_text(body).to_dict()
        layer_a_total = ta["suspicious_total"]
        layer_a_hits = ta["hits"]
        for h in layer_a_hits:
            findings.append(f"layer-a: {h['codepoint']} {h['label']} x{h['count']} ({h['kind']})")
    elif fmt == "epub":
        from text_unicode import inspect_text  # local import to avoid cycles

        encrypted = _epub_encrypted_parts(data)
        budget: list[int] = [0]
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    if info.filename in encrypted:
                        continue
                    if info.filename.lower().endswith((".xhtml", ".html", ".htm")):
                        ta = inspect_text(
                            _read_zip_member(zf, info, budget).decode(
                                "utf-8", errors="surrogateescape"
                            )
                        ).to_dict()
                        layer_a_total += ta["suspicious_total"]
                        for h in ta["hits"]:
                            layer_a_hits.append(h)
                            findings.append(
                                f"layer-a ({info.filename}): {h['codepoint']} {h['label']} "
                                f"x{h['count']} ({h['kind']})"
                            )
        except zipfile.BadZipFile:
            pass
    elif fmt in ("docx", "xlsx", "pptx", "odt"):
        for part_name, ta in _inspect_container_body_layer_a(data, fmt, zip_budget):
            layer_a_total += ta["suspicious_total"]
            for h in ta["hits"]:
                layer_a_hits.append(h)
                findings.append(
                    f"layer-a ({part_name}): {h['codepoint']} {h['label']} "
                    f"x{h['count']} ({h['kind']})"
                )

    notes: list[str] = []
    if fmt == "pdf":
        notes.append(
            "PDF inspection is best-effort; exiftool/c2patool give more reliable metadata detection"
        )
    elif fmt in ("docx", "xlsx", "pptx"):
        notes.append(f"{fmt.upper()}: metadata/provenance and embedded media are scanned")
    elif fmt == "epub":
        notes.append(
            "EPUB: package-document metadata, XHTML meta/JSON-LD, and embedded media are scanned"
        )
    if "unsupported" in details:
        notes.append(f"format not fully inspected: {fmt}")
    if layer_a_total:
        notes.append(
            f"layer A: {layer_a_total} invisible/format codepoint(s) in body text; "
            "clean removes these"
        )

    if fmt in ("svg", "pdf", "docx", "xlsx", "pptx") and not tools:
        tools = run_optional_tools(path)

    return ContainerInspectReport(
        path=str(path),
        format=fmt,
        has_c2pa=has_c2pa,
        has_ai_metadata=has_ai,
        findings=findings,
        tools=tools,
        details=details,
        notes=notes,
        layer_a_total=layer_a_total,
        layer_a_hits=layer_a_hits,
    )


def clean_container(
    path: Path,
    dest: Path,
    fmt: str | None = None,
    *,
    also_layer_a_text: bool = True,
    deep_images: str = "auto",
    normalize_spaces: bool = True,
) -> dict[str, Any]:
    """Clean container metadata; optionally Layer-A scrub text bodies for md/html.

    ``fmt`` pins the container format when the caller already knows it. This
    matters for ``--in-place`` flows where *path* is a ``.bak`` copy whose
    suffix would otherwise make markdown/HTML (which have no magic bytes)
    classify as ``unknown``.
    """
    from text_unicode import clean_text  # local import to avoid cycles

    data = path.read_bytes()
    fmt = fmt or detect_container_format(path, data)
    actions: list[str] = []
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"format": fmt}

    if fmt == "svg":
        cleaned, actions = clean_svg(data)
        safe_write_bytes(dest, cleaned)
    elif fmt == "pdf":
        actions, meta_extra = clean_pdf(path, dest, deep_images=deep_images)
        meta.update(meta_extra)
    elif fmt == "docx":
        cleaned, actions = clean_docx(
            data, also_layer_a_text=also_layer_a_text, normalize_spaces=normalize_spaces
        )
        safe_write_bytes(dest, cleaned)
    elif fmt == "xlsx":
        cleaned, actions = clean_xlsx(
            data, also_layer_a_text=also_layer_a_text, normalize_spaces=normalize_spaces
        )
        safe_write_bytes(dest, cleaned)
    elif fmt == "pptx":
        cleaned, actions = clean_pptx(
            data, also_layer_a_text=also_layer_a_text, normalize_spaces=normalize_spaces
        )
        safe_write_bytes(dest, cleaned)
    elif fmt == "odt":
        cleaned, actions = clean_odt(
            data, also_layer_a_text=also_layer_a_text, normalize_spaces=normalize_spaces
        )
        safe_write_bytes(dest, cleaned)
    elif fmt == "epub":
        cleaned, actions = clean_epub(
            data, also_layer_a_text=also_layer_a_text, normalize_spaces=normalize_spaces
        )
        safe_write_bytes(dest, cleaned)
    elif fmt == "html":
        text = data.decode("utf-8", errors="surrogateescape")
        text, actions = clean_html(text)
        if also_layer_a_text:
            text2, stats = clean_text(text, normalize_spaces=normalize_spaces)
            if stats["removed_count"] or stats["replaced_count"]:
                actions.append(
                    f"layer A text: removed={stats['removed_count']} replaced={stats['replaced_count']}"
                )
                text = text2
        safe_write_text(dest, text)
    elif fmt == "markdown":
        text = data.decode("utf-8", errors="surrogateescape")
        text, actions = clean_markdown(text)
        if also_layer_a_text:
            text2, stats = clean_text(text, normalize_spaces=normalize_spaces)
            if stats["removed_count"] or stats["replaced_count"]:
                actions.append(
                    f"layer A text: removed={stats['removed_count']} replaced={stats['replaced_count']}"
                )
                text = text2
        safe_write_text(dest, text)
    else:
        raise ValueError(f"unsupported container format: {fmt}")

    after = inspect_container(dest)
    changed = dest.read_bytes() != data
    return {
        "input": str(path),
        "output": str(dest),
        "format": fmt,
        "actions": actions,
        "bytes_in": len(data),
        "bytes_out": dest.stat().st_size,
        "changed": changed,
        "still_has_c2pa": after.has_c2pa,
        "still_has_ai_metadata": after.has_ai_metadata,
        "post_findings": after.findings,
        "meta": meta,
    }
