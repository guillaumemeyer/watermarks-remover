"""Tests for audio routing (.ogg, .opus, .aac) in av_meta and format_dispatch."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from av_meta import AV_EXTS, clean_av, detect_av_format
from clean_audio import AUDIO_EXTS
from format_dispatch import classify_bytes


def test_classify_bytes_by_extension():
    ogg_bytes = b"OggS" + b"\x00" * 60
    assert classify_bytes(ogg_bytes, ".ogg") == "av"
    assert classify_bytes(ogg_bytes, ".opus") == "av"
    assert classify_bytes(b"\xff\xf1" + b"\x00" * 60, ".aac") == "av"


def test_classify_bytes_by_magic_bytes_alone():
    ogg_bytes = b"OggS" + b"\x00" * 60
    assert classify_bytes(ogg_bytes, None) == "av"


def test_detect_av_format_aac_vs_mp3_ordering():
    # ADTS sync words (0xFFF...) must be detected as aac rather than falling
    # through to the MPEG audio frame-sync mask, including CRC-protected variants.
    for sync in (b"\xff\xf0", b"\xff\xf1", b"\xff\xf8", b"\xff\xf9"):
        assert detect_av_format(sync + b"\x00" * 60) == "aac"
    mp3_bytes = b"\xff\xfb" + b"\x00" * 60
    assert detect_av_format(mp3_bytes) == "mp3"


def test_clean_av_passthrough_for_ogg_and_aac(tmp_path):
    ogg_file = tmp_path / "sample.ogg"
    ogg_data = b"OggS" + b"\x00" * 60
    ogg_file.write_bytes(ogg_data)
    dest_ogg = tmp_path / "out.ogg"

    res_ogg = clean_av(ogg_file, dest_ogg)
    assert dest_ogg.read_bytes() == ogg_data
    assert res_ogg["format"] == "ogg"
    assert res_ogg["changed"] is False

    aac_file = tmp_path / "sample.aac"
    aac_data = b"\xff\xf1" + b"\x00" * 60
    aac_file.write_bytes(aac_data)
    dest_aac = tmp_path / "out.aac"

    res_aac = clean_av(aac_file, dest_aac)
    assert dest_aac.read_bytes() == aac_data
    assert res_aac["format"] == "aac"
    assert res_aac["changed"] is False


def test_detect_av_format_id3_and_flac_after_id3():
    id3_mp3 = b"ID3\x03\x00\x00\x00\x00\x00\x00" + (b"\xff\xfb\x90\x00" * 4)
    id3_flac = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"fLaC" + (b"\x00" * 34)
    assert detect_av_format(id3_mp3) == "mp3"
    assert detect_av_format(id3_flac) == "flac"


def test_audio_exts_subset_of_av_exts():
    # Every audio format known to clean_audio must be routable by av_meta/format_dispatch.
    diff = AUDIO_EXTS - AV_EXTS
    assert diff == set()
