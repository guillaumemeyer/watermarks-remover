#!/usr/bin/env python3
"""AI/C2PA provenance metadata for audio and video containers.

Extends the file-cleaners layer (image_meta.py for PNG/JPEG/..., container_meta.py
for SVG/PDF/DOCX/...) to MP4/MOV/M4A/M4V (ISOBMFF), WAV, MP3, and FLAC. Generative
audio/video tools embed provenance the same way image generators do -- C2PA
manifests and XMP in ISOBMFF boxes, generator tags in RIFF chunks and ID3v2
frames (including FLAC's standard C2PA carrier) -- so this reuses the existing
ISOBMFF box walker from image_meta.py
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

AV_EXTS = {".mp4", ".mov", ".m4a", ".m4v", ".wav", ".mp3", ".flac"}


@dataclass
class AVInspectReport:
    path: str
    format: str  # mp4 | wav | mp3 | flac | unknown
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
    """Sniff MP4/MOV/M4A/M4V (ISOBMFF), WAV, MP3, or FLAC from magic bytes."""
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "mp4"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"fLaC":
        return "flac"
    if len(data) >= 10 and data[:3] == b"ID3":
        parsed = _parse_id3v2_frames(data)
        if parsed is not None and data[parsed[0] : parsed[0] + 4] == b"fLaC":
            return "flac"
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


def _strip_moov_udta(data: bytes, *, strip_all_metadata: bool) -> tuple[bytes, list[str], bool]:
    actions: list[str] = []
    out = bytearray()
    boxes, scanned_end = _parse_isobmff_boxes(data)
    for fourcc, payload, _size, hdr in boxes:
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
    out.extend(data[scanned_end:])
    return bytes(out), actions, len(data) - scanned_end >= 8


def _inspect_mp4(data: bytes) -> tuple[bool, bool, list[str]]:
    has_c2pa, has_ai, findings = inspect_isobmff(data, fmt="mp4")
    udta_c2pa, udta_ai, udta_findings = _inspect_moov_udta(data)
    return has_c2pa or udta_c2pa, has_ai or udta_ai, findings + udta_findings


def _strip_mp4(data: bytes, *, strip_all_metadata: bool) -> tuple[bytes, list[str], bool]:
    cleaned, actions = strip_isobmff(data, fmt="mp4", strip_all_metadata=strip_all_metadata)
    cleaned, udta_actions, inspection_incomplete = _strip_moov_udta(
        cleaned, strip_all_metadata=strip_all_metadata
    )
    actions = [a for a in actions if not a.startswith("no MP4 metadata")] + udta_actions
    if not actions:
        actions = ["no MP4 metadata boxes removed (already clean or none matched)"]
    return cleaned, actions, inspection_incomplete


# ---------------------------------------------------------------------------
# ID3v2 (shared by MP3 files and WAV's optional `id3 ` chunk)
# ---------------------------------------------------------------------------


def _id3v2_size(data: bytes, offset: int) -> int:
    b0, b1, b2, b3 = data[offset], data[offset + 1], data[offset + 2], data[offset + 3]
    return ((b0 & 0x7F) << 21) | ((b1 & 0x7F) << 14) | ((b2 & 0x7F) << 7) | (b3 & 0x7F)


def _id3v2_size_bytes(n: int) -> bytes:
    return bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])


def _id3v2_frames_start(data: bytes, total: int, major: int) -> int | None:
    pos = 10
    if not (data[5] & 0x40):
        return pos
    if pos + 4 > total:
        return None
    ext_size = (
        _id3v2_size(data, pos) if major == 4 else struct.unpack(">I", data[pos : pos + 4])[0] + 4
    )
    pos += ext_size
    return pos if pos <= total else None


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
    frames_end = 10 + tag_size
    footer_size = 10 if major == 4 and data[5] & 0x10 else 0
    total = frames_end + footer_size
    if total > len(data):
        return None
    if footer_size and data[frames_end:total] != b"3DI" + data[3:10]:
        return None
    if major < 3:
        return total, major, []

    frames: list[tuple[bytes, bytes]] = []
    pos = _id3v2_frames_start(data, frames_end, major)
    if pos is None:
        return None
    while pos + 10 <= frames_end:
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
        if frame_size < 0 or frame_end > frames_end:
            return None
        frames.append((frame_id, data[frame_start:frame_end]))
        pos = frame_end
    if any(data[pos:frames_end]):
        return None
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
# FLAC (C2PA's standardized ID3v2 GEOB carrier only)
# ---------------------------------------------------------------------------


def _geob_text_end(payload: bytes, start: int, encoding: int) -> int | None:
    terminator = b"\x00" if encoding in (0, 3) else b"\x00\x00"
    step = len(terminator)
    for pos in range(start, len(payload) - step + 1, step):
        if payload[pos : pos + step] == terminator:
            return pos + step
    return None


def _is_c2pa_geob(frame_id: bytes, payload: bytes) -> bool:
    if frame_id != b"GEOB" or not payload or payload[0] not in range(4):
        return False
    mime_end = payload.find(b"\x00", 1)
    if mime_end < 0 or payload[1:mime_end].lower() != b"application/c2pa":
        return False
    filename_end = _geob_text_end(payload, mime_end + 1, payload[0])
    if filename_end is None:
        return False
    description_end = _geob_text_end(payload, filename_end, payload[0])
    return description_end is not None and description_end < len(payload)


def _inspect_flac(data: bytes) -> tuple[bool, bool, list[str]]:
    parsed = _parse_id3v2_frames(data)
    if parsed is None:
        return False, False, []
    _total, _major, frames = parsed
    if any(_is_c2pa_geob(frame_id, payload) for frame_id, payload in frames):
        return True, True, ["C2PA-related manifest in ID3v2 frame GEOB: application/c2pa"]
    return False, False, []


def _strip_flac(data: bytes, *, strip_all_metadata: bool) -> tuple[bytes, list[str]]:
    parsed = _parse_id3v2_frames(data)
    if parsed is None:
        return data, ["no FLAC ID3v2 metadata removed (already clean or none matched)"]
    total, major, frames = parsed
    rest = data[total:]
    if strip_all_metadata:
        return rest, [f"drop FLAC ID3v2.{major} tag ({total} bytes)"]

    kept = bytearray()
    actions: list[str] = []
    pos = _id3v2_frames_start(data, total, major)
    if pos is None:
        return data, ["no FLAC C2PA metadata removed (invalid ID3v2 extended header)"]
    for frame_id, payload in frames:
        frame_end = pos + 10 + len(payload)
        if _is_c2pa_geob(frame_id, payload):
            actions.append("drop FLAC ID3v2 frame GEOB: application/c2pa")
        else:
            kept.extend(data[pos:frame_end])
        pos = frame_end

    if not actions:
        return data, ["no FLAC C2PA metadata removed (already clean or none matched)"]

    if not kept:
        return rest, actions

    header = data[:5] + bytes([data[5] & ~0x50]) + _id3v2_size_bytes(len(kept))
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
    elif fmt == "flac":
        has_c2pa, has_ai, findings = _inspect_flac(data)
    else:
        has_c2pa, has_ai, findings = False, False, ["unsupported format (MP4/MOV/M4A/WAV/MP3/FLAC)"]

    notes: list[str] = []
    if fmt == "unknown":
        notes.append("format not fully inspected; only MP4/MOV/M4A/WAV/MP3/FLAC are supported")

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
    inspection_incomplete = False
    if fmt == "mp4":
        cleaned, actions, inspection_incomplete = _strip_mp4(
            data, strip_all_metadata=strip_all_metadata
        )
    elif fmt == "wav":
        cleaned, actions = _strip_wav(data, strip_all_metadata=strip_all_metadata)
    elif fmt == "mp3":
        cleaned, actions = _strip_id3v2(data, strip_all_metadata=strip_all_metadata)
    elif fmt == "flac":
        cleaned, actions = _strip_flac(data, strip_all_metadata=strip_all_metadata)
    else:
        raise ValueError(f"unsupported audio/video format for cleaning: {fmt}")

    safe_write_bytes(dest, cleaned)

    after = inspect_av(dest)
    post_findings = list(after.findings)
    if inspection_incomplete:
        post_findings.append("MP4 not fully inspected: preserved a truncated top-level box tail")
    return {
        "input": str(path),
        "output": str(dest),
        "format": fmt,
        "actions": actions,
        "bytes_in": len(data),
        "bytes_out": len(cleaned),
        "still_has_c2pa": after.has_c2pa,
        "still_has_ai_metadata": after.has_ai_metadata or inspection_incomplete,
        "post_findings": post_findings,
    }
