"""Tests for av_meta.py (MP4/MOV, WAV, MP3 AI/C2PA provenance metadata)."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from av_meta import (
    clean_av,
    detect_av_format,
    inspect_av,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _isobmff_box(fourcc: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + fourcc + payload


def _extended_isobmff_box(fourcc: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4sQ", 1, fourcc, len(payload) + 16) + payload


def _mp4(*top_level_boxes: bytes) -> bytes:
    ftyp = _isobmff_box(b"ftyp", b"isom" + struct.pack(">I", 0) + b"isomiso2mp41")
    return ftyp + b"".join(top_level_boxes)


def _moov_with_udta(udta_payload: bytes) -> bytes:
    mvhd = _isobmff_box(b"mvhd", b"\x00" * 20)
    udta = _isobmff_box(b"udta", udta_payload)
    return _isobmff_box(b"moov", mvhd + udta)


XMP_UUID_HEX = bytes.fromhex("be7acfcb97a942e89c71999491e3afac")


def _mp4_with_xmp(xmp_text: bytes) -> bytes:
    uuid_box = _isobmff_box(b"uuid", XMP_UUID_HEX + xmp_text)
    mdat = _isobmff_box(b"mdat", b"\x00" * 16)
    return _mp4(uuid_box, mdat)


def _mp4_with_udta_tag(tag_text: bytes) -> bytes:
    moov = _moov_with_udta(
        b"\xa9too" + struct.pack(">I", len(tag_text) + 4) + b"\x00\x00\x00\x00" + tag_text
    )
    mdat = _isobmff_box(b"mdat", b"\x00" * 16)
    return _mp4(moov, mdat)


def _riff_chunk(cid: bytes, payload: bytes) -> bytes:
    pad = b"\x00" if len(payload) & 1 else b""
    return cid + struct.pack("<I", len(payload)) + payload + pad


def _wav(*chunks: bytes) -> bytes:
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _wav_fmt_chunk() -> bytes:
    return _riff_chunk(b"fmt ", struct.pack("<HHIIHH", 1, 1, 44100, 88200, 2, 16))


def _wav_data_chunk(n: int = 8) -> bytes:
    return _riff_chunk(b"data", b"\x01" * n)


def _wav_list_info(text: bytes) -> bytes:
    isft = _riff_chunk(b"ISFT", text + (b"\x00" if len(text) % 2 == 0 else b""))
    return _riff_chunk(b"LIST", b"INFO" + isft)


def _id3v2_size_bytes(n: int) -> bytes:
    return bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])


def _id3v2_frame(
    frame_id: bytes, payload: bytes, *, major: int = 3, flags: bytes = b"\x00\x00"
) -> bytes:
    size = _id3v2_size_bytes(len(payload)) if major == 4 else struct.pack(">I", len(payload))
    return frame_id + size + flags + payload


def _mp3(*frames: bytes, major: int = 3) -> bytes:
    body = b"".join(frames)
    header = b"ID3" + bytes([major, 0, 0]) + _id3v2_size_bytes(len(body))
    audio = bytes([0xFF, 0xFB, 0x90, 0x00]) * 4  # placeholder MPEG frame-sync bytes
    return header + body + audio


def _flac(
    *frames: bytes, major: int = 3, extended_header: bytes = b"", footer: bool = False
) -> bytes:
    body = extended_header + b"".join(frames)
    flags = (0x40 if extended_header else 0) | (0x10 if footer else 0)
    size = _id3v2_size_bytes(len(body))
    id3_footer = b"3DI" + bytes([major, 0, flags]) + size if footer else b""
    id3 = b"ID3" + bytes([major, 0, flags]) + size + body + id3_footer if body else b""
    streaminfo = b"\x80\x00\x00\x22" + b"\x00" * 34
    return id3 + b"fLaC" + streaminfo


def _c2pa_geob() -> bytes:
    return b"\x00application/c2pa\x00manifest.c2pa\x00Content Credentials\x00jumbf-data"


# ---------------------------------------------------------------------------
# detect_av_format
# ---------------------------------------------------------------------------


def test_detect_mp4():
    assert detect_av_format(_mp4()) == "mp4"


def test_detect_wav():
    assert detect_av_format(_wav(_wav_fmt_chunk())) == "wav"


def test_detect_mp3_with_id3():
    assert detect_av_format(_mp3(_id3v2_frame(b"TIT2", b"\x00Song"))) == "mp3"


def test_detect_mp3_frame_sync_only():
    data = bytes([0xFF, 0xFB, 0x90, 0x00]) * 10
    assert detect_av_format(data) == "mp3"


def test_detect_flac_with_or_without_id3():
    assert detect_av_format(_flac()) == "flac"
    assert detect_av_format(_flac(_id3v2_frame(b"TIT2", b"\x00My Track"))) == "flac"


def test_detect_unknown():
    assert detect_av_format(b"not a known av container") == "unknown"


def test_flac_c2pa_geob_detected(tmp_path):
    data = _flac(_id3v2_frame(b"GEOB", _c2pa_geob()))
    src = tmp_path / "voice.flac"
    src.write_bytes(data)

    report = inspect_av(src)
    assert report.format == "flac"
    assert report.has_c2pa is True
    assert report.has_ai_metadata is True
    assert any("GEOB" in finding for finding in report.findings)
    assert report.to_dict()["findings_confidence"] == ["confirmed"]


def test_flac_keep_mode_drops_c2pa_geob_and_preserves_audio_and_title(tmp_path):
    flac_payload = _flac()
    data = _flac(
        _id3v2_frame(b"TIT2", b"\x00My Track"),
        _id3v2_frame(b"GEOB", _c2pa_geob()),
    )
    src = tmp_path / "voice.flac"
    src.write_bytes(data)
    dest = tmp_path / "voice.cleaned.flac"

    result = clean_av(src, dest, strip_all_metadata=False)

    cleaned = dest.read_bytes()
    assert result["format"] == "flac"
    assert result["still_has_c2pa"] is False
    assert b"My Track" in cleaned
    assert b"application/c2pa" not in cleaned
    assert cleaned.endswith(flac_payload)
    assert any("GEOB" in action for action in result["actions"])


def test_flac_keep_mode_drops_empty_id3_tag(tmp_path):
    flac_payload = _flac()
    src = tmp_path / "voice.flac"
    src.write_bytes(_flac(_id3v2_frame(b"GEOB", _c2pa_geob())))
    dest = tmp_path / "voice.cleaned.flac"

    clean_av(src, dest, strip_all_metadata=False)

    assert dest.read_bytes() == flac_payload


def test_flac_keep_mode_preserves_retained_frame_bytes(tmp_path):
    title_frame = _id3v2_frame(b"TIT2", b"\x00My Track", major=4, flags=b"\x20\x00")
    src = tmp_path / "voice.flac"
    src.write_bytes(_flac(title_frame, _id3v2_frame(b"GEOB", _c2pa_geob(), major=4), major=4))
    dest = tmp_path / "voice.cleaned.flac"

    clean_av(src, dest, strip_all_metadata=False)

    assert title_frame in dest.read_bytes()


def test_flac_c2pa_geob_after_id3_extended_header(tmp_path):
    v23_extended = struct.pack(">I", 10) + b"\x80\x00" + struct.pack(">I", 0) + b"\x00" * 4
    v24_extended = _id3v2_size_bytes(6) + b"\x01\x00"
    flac_payload = _flac()

    for major, extended_header in ((3, v23_extended), (4, v24_extended)):
        title_frame = _id3v2_frame(b"TIT2", b"\x00My Track", major=major)
        src = tmp_path / f"voice-v24-{major}.flac"
        src.write_bytes(
            _flac(
                title_frame,
                _id3v2_frame(b"GEOB", _c2pa_geob(), major=major),
                major=major,
                extended_header=extended_header,
            )
        )

        report = inspect_av(src)
        assert report.has_c2pa is True

        dest = tmp_path / f"voice-v24-{major}.cleaned.flac"
        clean_av(src, dest, strip_all_metadata=False)
        cleaned = dest.read_bytes()
        assert b"application/c2pa" not in cleaned
        assert title_frame in cleaned
        assert cleaned.endswith(flac_payload)
        assert cleaned[5] & 0x40 == 0


def test_flac_c2pa_geob_before_id3v24_footer(tmp_path):
    title_frame = _id3v2_frame(b"TIT2", b"\x00My Track", major=4)
    src = tmp_path / "voice.flac"
    src.write_bytes(
        _flac(
            title_frame,
            _id3v2_frame(b"GEOB", _c2pa_geob(), major=4),
            major=4,
            footer=True,
        )
    )

    report = inspect_av(src)
    assert report.format == "flac"
    assert report.has_c2pa is True

    dest = tmp_path / "voice.cleaned.flac"
    clean_av(src, dest, strip_all_metadata=False)
    cleaned = dest.read_bytes()
    assert detect_av_format(cleaned) == "flac"
    assert title_frame in cleaned
    assert b"application/c2pa" not in cleaned
    assert b"3DI" not in cleaned
    assert cleaned[5] & 0x10 == 0
    assert cleaned.endswith(_flac())


def test_flac_ignores_non_geob_ai_text(tmp_path):
    data = _flac(_id3v2_frame(b"COMM", b"\x00Generated by AI"))
    src = tmp_path / "voice.flac"
    src.write_bytes(data)

    report = inspect_av(src)
    assert report.has_c2pa is False
    assert report.has_ai_metadata is False

    dest = tmp_path / "voice.cleaned.flac"
    clean_av(src, dest, strip_all_metadata=False)
    assert dest.read_bytes() == data


def test_flac_ignores_truncated_c2pa_geob(tmp_path):
    data = _flac(_id3v2_frame(b"GEOB", b"\x00application/c2pa\x00"))
    src = tmp_path / "voice.flac"
    src.write_bytes(data)

    report = inspect_av(src)
    assert report.has_c2pa is False
    assert report.has_ai_metadata is False

    dest = tmp_path / "voice.cleaned.flac"
    clean_av(src, dest, strip_all_metadata=False)
    assert dest.read_bytes() == data


def test_flac_keep_mode_preserves_tag_with_truncated_frame(tmp_path):
    truncated_title = b"TIT2" + struct.pack(">I", 20) + b"\x00\x00partial"
    data = _flac(
        _id3v2_frame(b"GEOB", _c2pa_geob()),
        truncated_title,
    )
    src = tmp_path / "voice.flac"
    src.write_bytes(data)
    dest = tmp_path / "voice.cleaned.flac"

    clean_av(src, dest, strip_all_metadata=False)

    assert dest.read_bytes() == data


def test_flac_keep_mode_preserves_tag_with_nonzero_short_tail(tmp_path):
    data = _flac(
        _id3v2_frame(b"GEOB", _c2pa_geob()),
        b"TIT2bad",
    )
    src = tmp_path / "voice.flac"
    src.write_bytes(data)
    dest = tmp_path / "voice.cleaned.flac"

    clean_av(src, dest, strip_all_metadata=False)

    assert dest.read_bytes() == data


def test_flac_keep_mode_accepts_zero_padding(tmp_path):
    src = tmp_path / "voice.flac"
    src.write_bytes(
        _flac(
            _id3v2_frame(b"GEOB", _c2pa_geob()),
            b"\x00" * 7,
        )
    )
    dest = tmp_path / "voice.cleaned.flac"

    clean_av(src, dest, strip_all_metadata=False)

    assert dest.read_bytes() == _flac()


# ---------------------------------------------------------------------------
# MP4 / MOV
# ---------------------------------------------------------------------------


def test_mp4_xmp_uuid_detected_and_stripped(tmp_path):
    data = _mp4_with_xmp(b"Generated by AI toolchain")
    src = tmp_path / "clip.mp4"
    src.write_bytes(data)

    report = inspect_av(src)
    assert report.format == "mp4"
    assert report.has_ai_metadata is True
    assert any("uuid" in f.lower() for f in report.findings)

    dest = tmp_path / "clip.cleaned.mp4"
    result = clean_av(src, dest, strip_all_metadata=True)
    assert result["still_has_ai_metadata"] is False
    assert b"Generated by AI" not in dest.read_bytes()
    assert any("uuid" in a.lower() for a in result["actions"])


def test_mp4_xmp_stripping_preserves_mdat_offset(tmp_path):
    data = _mp4_with_xmp(b"Generated by AI toolchain")
    src = tmp_path / "clip.mp4"
    src.write_bytes(data)
    dest = tmp_path / "clip.cleaned.mp4"

    clean_av(src, dest, strip_all_metadata=True)

    cleaned = dest.read_bytes()
    assert cleaned.index(b"mdat") == data.index(b"mdat")
    assert b"Generated by AI" not in cleaned


def test_mp4_extended_xmp_stripping_preserves_mdat_offset(tmp_path):
    xmp = _extended_isobmff_box(b"uuid", XMP_UUID_HEX + b"Generated by AI")
    data = _mp4(xmp, _isobmff_box(b"mdat", b"\x00" * 16))
    src = tmp_path / "clip.mp4"
    src.write_bytes(data)
    dest = tmp_path / "clip.cleaned.mp4"

    clean_av(src, dest, strip_all_metadata=True)

    cleaned = dest.read_bytes()
    assert cleaned.index(b"mdat") == data.index(b"mdat")
    assert cleaned[28:36] == b"\x00\x00\x00\x01free"
    assert b"Generated by AI" not in cleaned


def test_mp4_moov_udta_generator_tag_detected_and_stripped(tmp_path):
    data = _mp4_with_udta_tag(b"ElevenLabs AI Voice Generator")
    src = tmp_path / "clip.mp4"
    src.write_bytes(data)

    report = inspect_av(src)
    assert report.has_ai_metadata is False  # ElevenLabs isn't in the flat hint list

    # Use an explicit AI hint that IS in the flat list to prove detection works.
    data2 = _mp4_with_udta_tag(b"Generated by AI")
    src2 = tmp_path / "clip2.mp4"
    src2.write_bytes(data2)
    report2 = inspect_av(src2)
    assert report2.has_ai_metadata is True
    assert any("udta" in f for f in report2.findings)

    dest = tmp_path / "clip2.cleaned.mp4"
    result = clean_av(src2, dest, strip_all_metadata=True)
    assert b"Generated by AI" not in dest.read_bytes()
    assert result["still_has_ai_metadata"] is False


def test_mp4_udta_stripping_preserves_mdat_offset(tmp_path):
    data = _mp4_with_udta_tag(b"Generated by AI")
    src = tmp_path / "clip.mp4"
    src.write_bytes(data)
    dest = tmp_path / "clip.cleaned.mp4"

    clean_av(src, dest, strip_all_metadata=True)

    cleaned = dest.read_bytes()
    assert cleaned.index(b"mdat") == data.index(b"mdat")
    assert b"Generated by AI" not in cleaned


def test_mp4_udta_stripped_by_default_even_without_ai_hint(tmp_path):
    """Default strip_all_metadata=True strips udta regardless of hint match,
    matching this project's existing default behaviour for image metadata."""
    data = _mp4_with_udta_tag(b"Adobe Premiere Pro 2026")
    src = tmp_path / "clip.mp4"
    src.write_bytes(data)
    dest = tmp_path / "clip.cleaned.mp4"
    result = clean_av(src, dest, strip_all_metadata=True)
    assert b"Adobe Premiere Pro 2026" not in dest.read_bytes()
    assert any("udta" in a for a in result["actions"])


def test_mp4_truncated_tail_survives_udta_stripping(tmp_path):
    moov = _moov_with_udta(b"    toolGenerated by OpenAI Sora")
    mdat = _isobmff_box(b"mdat", b"\x00" * 4096)
    whole = _mp4(moov, mdat)
    mdat_start = whole.index(mdat)
    data = whole[:-1024]
    src = tmp_path / "truncated.mp4"
    src.write_bytes(data)
    dest = tmp_path / "truncated.cleaned.mp4"

    result = clean_av(src, dest, strip_all_metadata=True)

    cleaned = dest.read_bytes()
    assert len(cleaned) == len(data)
    assert cleaned[mdat_start:] == data[mdat_start:]
    assert b"Generated by OpenAI Sora" not in cleaned
    assert any("truncated tail" in action for action in result["actions"])
    assert result["still_has_ai_metadata"] is True
    assert any("not fully inspected" in finding for finding in result["post_findings"])


def test_mp4_truncated_metadata_box_is_reported_as_inconclusive(tmp_path):
    moov = _moov_with_udta(b"    toolGenerated by OpenAI Sora" + b"\x00" * 32)
    data = _mp4(moov)[:-16]
    src = tmp_path / "truncated-metadata.mp4"
    src.write_bytes(data)
    dest = tmp_path / "truncated-metadata.cleaned.mp4"

    result = clean_av(src, dest, strip_all_metadata=True)

    assert dest.read_bytes() == data
    assert result["still_has_ai_metadata"] is True
    assert any("not fully inspected" in finding for finding in result["post_findings"])


def test_mp4_keep_non_ai_metadata_preserves_unflagged_udta(tmp_path):
    data = _mp4_with_udta_tag(b"Adobe Premiere Pro 2026")
    src = tmp_path / "clip.mp4"
    src.write_bytes(data)
    dest = tmp_path / "clip.cleaned.mp4"
    result = clean_av(src, dest, strip_all_metadata=False)
    assert b"Adobe Premiere Pro 2026" in dest.read_bytes()
    assert not any("udta" in a for a in result["actions"])


def test_mp4_clean_file_is_idempotent_when_already_clean(tmp_path):
    data = _mp4()
    src = tmp_path / "clean.mp4"
    src.write_bytes(data)
    dest = tmp_path / "clean.cleaned.mp4"
    result = clean_av(src, dest, strip_all_metadata=True)
    assert result["still_has_ai_metadata"] is False
    assert result["still_has_c2pa"] is False


# ---------------------------------------------------------------------------
# WAV
# ---------------------------------------------------------------------------


def test_wav_list_info_ai_hint_detected_and_stripped(tmp_path):
    data = _wav(_wav_fmt_chunk(), _wav_list_info(b"Generated by AI"), _wav_data_chunk(8))
    src = tmp_path / "voice.wav"
    src.write_bytes(data)

    report = inspect_av(src)
    assert report.format == "wav"
    assert report.has_ai_metadata is True
    assert any("LIST INFO" in f for f in report.findings)

    dest = tmp_path / "voice.cleaned.wav"
    result = clean_av(src, dest, strip_all_metadata=True)
    cleaned = dest.read_bytes()
    assert b"Generated by AI" not in cleaned
    assert result["still_has_ai_metadata"] is False


def test_wav_c2pa_chunk_detected_and_stripped(tmp_path):
    manifest = b"C2PA manifest store"
    fmt_chunk = _wav_fmt_chunk()
    data_chunk = _wav_data_chunk(8)
    data = _wav(fmt_chunk, _riff_chunk(b"C2PA", manifest), data_chunk)
    src = tmp_path / "voice.wav"
    src.write_bytes(data)

    report = inspect_av(src)
    assert report.has_c2pa is True
    assert any("C2PA" in f for f in report.findings)

    dest = tmp_path / "voice.cleaned.wav"
    result = clean_av(src, dest, strip_all_metadata=False)
    cleaned = dest.read_bytes()
    assert result["actions"] == ["drop WAV C2PA chunk"]
    assert cleaned == _wav(fmt_chunk, data_chunk)
    assert b"C2PA" not in cleaned
    assert manifest not in cleaned
    assert result["still_has_c2pa"] is False


def test_wav_audio_data_untouched(tmp_path):
    audio = bytes(range(256)) * 4
    data = _wav(_wav_fmt_chunk(), _wav_list_info(b"Generated by AI"), _riff_chunk(b"data", audio))
    src = tmp_path / "voice.wav"
    src.write_bytes(data)
    dest = tmp_path / "voice.cleaned.wav"
    clean_av(src, dest, strip_all_metadata=True)
    cleaned = dest.read_bytes()
    assert audio in cleaned
    # RIFF size field must match the new (shorter) total.
    riff_size = struct.unpack("<I", cleaned[4:8])[0]
    assert riff_size == len(cleaned) - 8


def test_wav_clean_file_already_clean_no_changes(tmp_path):
    audio = b"\x01\x02\x03\x04" * 4
    data = _wav(_wav_fmt_chunk(), _riff_chunk(b"data", audio))
    src = tmp_path / "clean.wav"
    src.write_bytes(data)
    dest = tmp_path / "clean.cleaned.wav"
    result = clean_av(src, dest, strip_all_metadata=True)
    assert "no WAV metadata chunks removed" in result["actions"][0]
    assert dest.read_bytes() == data


def test_wav_keep_non_ai_metadata_preserves_unflagged_info(tmp_path):
    data = _wav(_wav_fmt_chunk(), _wav_list_info(b"Adobe Audition"), _wav_data_chunk())
    src = tmp_path / "voice.wav"
    src.write_bytes(data)
    dest = tmp_path / "voice.cleaned.wav"
    clean_av(src, dest, strip_all_metadata=False)
    assert b"Adobe Audition" in dest.read_bytes()


# ---------------------------------------------------------------------------
# MP3 (ID3v2)
# ---------------------------------------------------------------------------


def test_mp3_id3v23_ai_hint_detected(tmp_path):
    data = _mp3(
        _id3v2_frame(b"TIT2", b"\x00My Track"),
        _id3v2_frame(b"TSSE", b"\x00Generated by AI"),
    )
    src = tmp_path / "song.mp3"
    src.write_bytes(data)

    report = inspect_av(src)
    assert report.format == "mp3"
    assert report.has_ai_metadata is True
    assert any("TSSE" in f for f in report.findings)


def test_mp3_strip_all_drops_whole_tag(tmp_path):
    data = _mp3(
        _id3v2_frame(b"TIT2", b"\x00My Track"),
        _id3v2_frame(b"TSSE", b"\x00Generated by AI"),
    )
    src = tmp_path / "song.mp3"
    src.write_bytes(data)
    dest = tmp_path / "song.cleaned.mp3"
    result = clean_av(src, dest, strip_all_metadata=True)
    cleaned = dest.read_bytes()
    assert b"My Track" not in cleaned
    assert b"Generated by AI" not in cleaned
    assert result["still_has_ai_metadata"] is False
    # The MPEG frame-sync audio placeholder bytes survive untouched.
    assert cleaned.endswith(bytes([0xFF, 0xFB, 0x90, 0x00]) * 4)


def test_mp3_keep_mode_drops_only_flagged_frame(tmp_path):
    data = _mp3(
        _id3v2_frame(b"TIT2", b"\x00My Track"),
        _id3v2_frame(b"TSSE", b"\x00Generated by AI"),
    )
    src = tmp_path / "song.mp3"
    src.write_bytes(data)
    dest = tmp_path / "song.cleaned.mp3"
    result = clean_av(src, dest, strip_all_metadata=False)
    cleaned = dest.read_bytes()
    assert b"My Track" in cleaned  # legitimate tag survives
    assert b"Generated by AI" not in cleaned  # flagged frame is gone
    assert any("TSSE" in a for a in result["actions"])


def test_mp3_id3v24_syncsafe_frame_size_round_trip(tmp_path):
    data = _mp3(
        _id3v2_frame(b"TIT2", b"\x00My Track", major=4),
        _id3v2_frame(b"TSSE", b"\x00Generated by AI", major=4),
        major=4,
    )
    src = tmp_path / "song.mp3"
    src.write_bytes(data)
    report = inspect_av(src)
    assert report.has_ai_metadata is True

    dest = tmp_path / "song.cleaned.mp3"
    result = clean_av(src, dest, strip_all_metadata=False)
    cleaned = dest.read_bytes()
    assert b"My Track" in cleaned
    assert b"Generated by AI" not in cleaned
    # Re-inspecting the cleaned, rewritten v2.4 tag must still parse correctly.
    after = inspect_av(dest)
    assert after.has_ai_metadata is False
    assert result["still_has_ai_metadata"] is False


def test_mp3_no_id3_tag_clean_noop(tmp_path):
    data = bytes([0xFF, 0xFB, 0x90, 0x00]) * 20
    src = tmp_path / "raw.mp3"
    src.write_bytes(data)
    dest = tmp_path / "raw.cleaned.mp3"
    result = clean_av(src, dest, strip_all_metadata=True)
    assert dest.read_bytes() == data
    assert result["still_has_ai_metadata"] is False


def test_mp3_id3v22_falls_back_to_whole_tag_scan_and_drop(tmp_path):
    # v2.2: 3-byte frame IDs, no per-frame decomposition -- whole-tag handling.
    body = b"TT2\x00\x00\x10\x00Generated by AI"
    header = b"ID3" + bytes([2, 0, 0]) + _id3v2_size_bytes(len(body))
    data = header + body + bytes([0xFF, 0xFB, 0x90, 0x00]) * 4
    src = tmp_path / "old.mp3"
    src.write_bytes(data)

    report = inspect_av(src)
    assert report.has_ai_metadata is True

    dest = tmp_path / "old.cleaned.mp3"
    result = clean_av(src, dest, strip_all_metadata=False)
    cleaned = dest.read_bytes()
    assert b"Generated by AI" not in cleaned
    assert any("ID3v2.2" in a for a in result["actions"])
