#!/usr/bin/env python3
"""AI/C2PA provenance metadata for audio and video containers.

Extends the file-cleaners layer (image_meta.py for PNG/JPEG/..., container_meta.py
for SVG/PDF/DOCX/...) to MP4/MOV/M4A/M4V (ISOBMFF), WAV, and MP3. Generative
audio/video tools embed provenance the same way image generators do -- C2PA
manifests and XMP in ISOBMFF boxes, generator tags in RIFF chunks and ID3v2
frames -- so this reuses the existing ISOBMFF box walker from image_meta.py
(the same mechanism already proven for AVIF/HEIC) rather than duplicating it.

Metadata only: waveform/pixel data is never touched, matching every other
cleaner in this project. A box/chunk/frame is either kept byte-identical or
dropped whole -- nothing here does a partial in-place rewrite of a box's
payload, so a container can never come out semantically mangled.

Known scope limits (documented, not silently mishandled):
- MP4/MOV: legacy QuickTime files with no top-level `ftyp` box are not
  detected by signature (rare in practice; modern encoders always write one).
- MP3: ID3v2.2 (3-byte frame IDs, pre-iTunes era) tags are detected but not
  decomposed into frames -- stripping falls back to a whole-tag drop, which
  is always safe. ID3v1 (fixed 128-byte trailer at EOF) is not handled.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common import classify_finding_confidence, safe_write_bytes
from image_meta import (
    AI_META_HINTS,
    XMP_UUID,  # noqa: F401 -- re-exported for callers that want the raw constant
    _build_isobmff_box,
    _contains_any,
    _isobmff_free_box,
    _parse_isobmff_boxes,
    inspect_isobmff,
    strip_isobmff,
)

AV_EXTS = {".mp4", ".mov", ".m4a", ".m4v", ".wav", ".mp3"}


@dataclass
class AVInspectReport:
    path: str
    format: str  # mp4 | wav | mp3 | unknown
    has_c2pa: bool
    has_ai_metadata: bool
    findings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "format": self.format,
            "has_c2pa": self.has_c2pa,
            "has_ai_metadata": self.has_ai_metadata,
            "findings": self.findings,
            "findings_confidence": [classify_finding_confidence(f) for f in self.findings],
            "notes": self.notes,
        }


def detect_av_format(data: bytes) -> str:
    """Sniff MP4/MOV/M4A/M4V (ISOBMFF), WAV, or MP3 from magic bytes."""
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "mp4"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if len(data) >= 3 and data[:3] == b"ID3":
        return "mp3"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "mp3"  # MPEG frame sync with no ID3v2 header (rare but valid)
    return "unknown"


def _classify_c2pa(hits: list[str]) -> bool:
    return any(h.lower() in ("c2pa", "contentcredentials", "jumb", "contentauth") for h in hits)


# ---------------------------------------------------------------------------
# MP4 / MOV / M4A / M4V (ISOBMFF)
# ---------------------------------------------------------------------------
#
# Top-level C2PA (jumb/c2pa box) and XMP (uuid box) detection/stripping reuse
# inspect_isobmff() / strip_isobmff() from image_meta.py unchanged -- that is
# exactly the mechanism the C2PA spec defines for ISOBMFF-family containers,
# already proven correct for AVIF/HEIC. moov/udta (QuickTime "user data",
# where generator/tool tags commonly live) is MP4-specific and handled here.


def _inspect_moov_udta(data: bytes) -> tuple[bool, bool, list[str]]:
    has_c2pa = False
    has_ai = False
    findings: list[str] = []
    for fourcc, payload, _size, _hdr in _parse_isobmff_boxes(data)[0]:
        if fourcc != b"moov":
            continue
        for s_fourcc, s_payload, _s_size, _s_hdr in _parse_isobmff_boxes(payload)[0]:
            if s_fourcc != b"udta":
                continue
            hits = _contains_any(s_payload, AI_META_HINTS)
            if hits:
                has_ai = True
                if _classify_c2pa(hits):
                    has_c2pa = True
                findings.append(f"MP4 moov/udta box: {', '.join(hits[:8])}")
    return has_c2pa, has_ai, findings


def _strip_moov_udta(data: bytes, *, strip_all_metadata: bool) -> tuple[bytes, list[str]]:
    actions: list[str] = []
    out = bytearray()
    for fourcc, payload, _size, hdr in _parse_isobmff_boxes(data)[0]:
        if fourcc != b"moov":
            out.extend(_build_isobmff_box(fourcc, payload, hdr))
            continue
        new_moov = bytearray()
        for s_fourcc, s_payload, s_size, s_hdr in _parse_isobmff_boxes(payload)[0]:
            if s_fourcc == b"udta" and (
                strip_all_metadata or _contains_any(s_payload, AI_META_HINTS)
            ):
                actions.append("drop moov/udta box (generator/user-data tags)")
                new_moov.extend(_isobmff_free_box(s_size, s_hdr))
                continue
            new_moov.extend(_build_isobmff_box(s_fourcc, s_payload, s_hdr))
        out.extend(_build_isobmff_box(b"moov", bytes(new_moov), hdr))
    return bytes(out), actions


def _inspect_mp4(data: bytes) -> tuple[bool, bool, list[str]]:
    has_c2pa, has_ai, findings = inspect_isobmff(data, fmt="mp4")
    udta_c2pa, udta_ai, udta_findings = _inspect_moov_udta(data)
    return has_c2pa or udta_c2pa, has_ai or udta_ai, findings + udta_findings


def _strip_mp4(data: bytes, *, strip_all_metadata: bool) -> tuple[bytes, list[str]]:
    cleaned, actions = strip_isobmff(data, fmt="mp4", strip_all_metadata=strip_all_metadata)
    cleaned, udta_actions = _strip_moov_udta(cleaned, strip_all_metadata=strip_all_metadata)
    actions = [a for a in actions if not a.startswith("no MP4 metadata")] + udta_actions
    if not actions:
        actions = ["no MP4 metadata boxes removed (already clean or none matched)"]
    return cleaned, actions


# ---------------------------------------------------------------------------
# ID3v2 (shared by MP3 files and WAV's optional `id3 ` chunk)
# ---------------------------------------------------------------------------


def _id3v2_size(data: bytes, offset: int) -> int:
    b0, b1, b2, b3 = data[offset], data[offset + 1], data[offset + 2], data[offset + 3]
    return ((b0 & 0x7F) << 21) | ((b1 & 0x7F) << 14) | ((b2 & 0x7F) << 7) | (b3 & 0x7F)


def _id3v2_size_bytes(n: int) -> bytes:
    return bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])


def _parse_id3v2_frames(data: bytes) -> tuple[int, int, list[tuple[bytes, bytes]]] | None:
    """Parse an ID3v2 tag at the start of *data*.

    Returns (tag_total_size, major_version, frames); frames is a list of
    (frame_id, frame_payload) for v2.3/v2.4 tags (4-byte frame IDs). v2.2
    tags (3-byte frame IDs) are detected but returned with an empty frame
    list -- callers fall back to whole-tag byte-scanning and whole-tag drop.
    """
    if len(data) < 10 or data[:3] != b"ID3":
        return None
    major = data[3]
    tag_size = _id3v2_size(data, 6)
    total = 10 + tag_size
    if total > len(data):
        return None
    if major < 3:
        return total, major, []

    frames: list[tuple[bytes, bytes]] = []
    pos = 10
    while pos + 10 <= total:
        frame_id = data[pos : pos + 4]
        if frame_id == b"\x00\x00\x00\x00":
            break  # padding
        frame_size = (
            _id3v2_size(data, pos + 4)
            if major == 4
            else struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        )
        frame_start = pos + 10
        frame_end = frame_start + frame_size
        if frame_size < 0 or frame_end > total:
            break
        frames.append((frame_id, data[frame_start:frame_end]))
        pos = frame_end
    return total, major, frames


def _inspect_id3v2(data: bytes) -> tuple[bool, bool, list[str]]:
    parsed = _parse_id3v2_frames(data)
    if parsed is None:
        return False, False, []
    total, major, frames = parsed
    findings: list[str] = []
    has_ai = False
    has_c2pa = False

    if not frames:
        hits = _contains_any(data[:total], AI_META_HINTS)
        if hits:
            has_ai = True
            has_c2pa = _classify_c2pa(hits)
            findings.append(f"ID3v2.{major} tag: {', '.join(hits[:8])}")
        return has_c2pa, has_ai, findings

    for frame_id, payload in frames:
        hits = _contains_any(payload, AI_META_HINTS)
        if hits:
            has_ai = True
            if _classify_c2pa(hits):
                has_c2pa = True
            label = frame_id.decode("latin-1", errors="replace")
            findings.append(f"ID3v2 frame {label}: {', '.join(hits[:8])}")
    return has_c2pa, has_ai, findings


def _strip_id3v2(data: bytes, *, strip_all_metadata: bool) -> tuple[bytes, list[str]]:
    parsed = _parse_id3v2_frames(data)
    if parsed is None:
        return data, []
    total, major, frames = parsed
    rest = data[total:]

    if not frames:
        # v2.2 (undecomposed) or an empty v2.3/2.4 tag: only a whole-tag drop
        # is safe here, since frame boundaries were never decoded.
        if not strip_all_metadata and not _contains_any(data[:total], AI_META_HINTS):
            return data, ["no ID3v2 tag removed (no AI/C2PA markers found)"]
        return rest, [f"drop ID3v2.{major} tag ({total} bytes)"]

    if strip_all_metadata:
        return rest, [f"drop ID3v2.{major} tag ({total} bytes)"]

    kept = bytearray()
    actions: list[str] = []
    for frame_id, payload in frames:
        hits = _contains_any(payload, AI_META_HINTS)
        if hits:
            label = frame_id.decode("latin-1", errors="replace")
            actions.append(f"drop ID3v2 frame {label}: {', '.join(hits[:8])}")
            continue
        size_bytes = (
            _id3v2_size_bytes(len(payload)) if major == 4 else struct.pack(">I", len(payload))
        )
        kept.extend(frame_id + size_bytes + b"\x00\x00" + payload)

    if not actions:
        return data, ["no ID3v2 frames removed (already clean or none matched)"]

    header = bytes([ord("I"), ord("D"), ord("3"), major, 0, 0]) + _id3v2_size_bytes(len(kept))
    return header + bytes(kept) + rest, actions


# ---------------------------------------------------------------------------
# WAV (RIFF)
# ---------------------------------------------------------------------------


def _inspect_wav(data: bytes) -> tuple[bool, bool, list[str]]:
    findings: list[str] = []
    has_ai = False
    has_c2pa = False
    pos = 12  # past "RIFF" + size(4) + "WAVE"
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        csize = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        cstart = pos + 8
        cend = cstart + csize
        if cend > len(data):
            break
        payload = data[cstart:cend]
        if cid == b"C2PA":
            has_c2pa = True
            findings.append("WAV C2PA-related manifest chunk")
        elif cid == b"LIST" and payload[:4] == b"INFO":
            hits = _contains_any(payload, AI_META_HINTS)
            if hits:
                has_ai = True
                if _classify_c2pa(hits):
                    has_c2pa = True
                findings.append(f"WAV LIST INFO chunk: {', '.join(hits[:8])}")
        elif cid in (b"id3 ", b"ID3 "):
            c2pa, ai, sub_findings = _inspect_id3v2(payload)
            if ai:
                has_ai = True
                has_c2pa = has_c2pa or c2pa
                findings.extend(f"WAV id3 chunk / {f}" for f in sub_findings)
        pos = cend + (csize & 1)  # chunks are word-aligned
    return has_c2pa, has_ai, findings


def _strip_wav(data: bytes, *, strip_all_metadata: bool) -> tuple[bytes, list[str]]:
    actions: list[str] = []
    out = bytearray(data[:12])
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        csize = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        cstart = pos + 8
        cend = cstart + csize
        if cend > len(data):
            out.extend(data[pos:])
            pos = len(data)
            break
        payload = data[cstart:cend]
        pad = csize & 1
        chunk_total = data[pos : cend + pad]

        drop = False
        is_c2pa = cid == b"C2PA"
        is_info = cid == b"LIST" and payload[:4] == b"INFO"
        is_id3 = cid in (b"id3 ", b"ID3 ")
        if is_c2pa or (
            (is_info or is_id3) and (strip_all_metadata or _contains_any(payload, AI_META_HINTS))
        ):
            label = "C2PA" if is_c2pa else "LIST INFO" if is_info else "id3"
            actions.append(f"drop WAV {label} chunk")
            drop = True

        if not drop:
            out.extend(chunk_total)
        pos = cend + pad

    struct.pack_into("<I", out, 4, len(out) - 8)
    if not actions:
        actions.append("no WAV metadata chunks removed (already clean or none matched)")
    return bytes(out), actions


# ---------------------------------------------------------------------------
# Unified inspect / clean
# ---------------------------------------------------------------------------


def inspect_av(path: Path) -> AVInspectReport:
    data = path.read_bytes()
    fmt = detect_av_format(data)
    if fmt == "mp4":
        has_c2pa, has_ai, findings = _inspect_mp4(data)
    elif fmt == "wav":
        has_c2pa, has_ai, findings = _inspect_wav(data)
    elif fmt == "mp3":
        has_c2pa, has_ai, findings = _inspect_id3v2(data)
    else:
        has_c2pa, has_ai, findings = False, False, ["unsupported format (MP4/MOV/M4A/WAV/MP3)"]

    notes: list[str] = []
    if fmt == "unknown":
        notes.append("format not fully inspected; only MP4/MOV/M4A/WAV/MP3 are supported")

    return AVInspectReport(
        path=str(path),
        format=fmt,
        has_c2pa=has_c2pa,
        has_ai_metadata=has_ai,
        findings=findings,
        notes=notes,
    )


def clean_av(path: Path, dest: Path, *, strip_all_metadata: bool = True) -> dict[str, Any]:
    data = path.read_bytes()
    fmt = detect_av_format(data)
    if fmt == "mp4":
        cleaned, actions = _strip_mp4(data, strip_all_metadata=strip_all_metadata)
    elif fmt == "wav":
        cleaned, actions = _strip_wav(data, strip_all_metadata=strip_all_metadata)
    elif fmt == "mp3":
        cleaned, actions = _strip_id3v2(data, strip_all_metadata=strip_all_metadata)
    else:
        raise ValueError(f"unsupported audio/video format for cleaning: {fmt}")

    safe_write_bytes(dest, cleaned)

    after = inspect_av(dest)
    return {
        "input": str(path),
        "output": str(dest),
        "format": fmt,
        "actions": actions,
        "bytes_in": len(data),
        "bytes_out": len(cleaned),
        "still_has_c2pa": after.has_c2pa,
        "still_has_ai_metadata": after.has_ai_metadata,
        "post_findings": after.findings,
    }
