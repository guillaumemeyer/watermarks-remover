"""Tests for clean_audio.py (destructive transform chain for audio watermarks)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from clean_audio import (
    _ffmpeg_available,
    audio_purify,
    is_audio_format,
    is_audio_name,
    plan_audio_degrade,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg/ffprobe not installed")


def _probe(path: Path) -> dict:
    r = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return json.loads(r.stdout)


# ---------------------------------------------------------------------------
# is_audio_format
# ---------------------------------------------------------------------------


def test_is_audio_format():
    assert is_audio_format("wav") is True
    assert is_audio_format("mp3") is True
    assert is_audio_format("flac") is True
    assert is_audio_format("mp4") is False


def test_is_audio_name_by_extension():
    assert is_audio_name("song.m4a") is True
    assert is_audio_name("voice.opus") is True
    assert is_audio_name("clip.mp3") is True
    assert is_audio_name("clip.mp4") is False
    assert is_audio_name(None) is False


# ---------------------------------------------------------------------------
# plan_audio_degrade -- validates the destructive chain
# ---------------------------------------------------------------------------


def test_plan_defaults_exit_survival_envelope():
    plan = plan_audio_degrade()
    assert plan["tempo"] == 1.08
    assert plan["pitch_semitones"] == 2.0
    assert any("tempo change >5%" in e for e in plan["survival_limits_exceeded"])
    assert any("pitch shift >1 semitone" in e for e in plan["survival_limits_exceeded"])
    assert any("lossy re-encode below 128 kbps" in e for e in plan["survival_limits_exceeded"])


def test_plan_rejects_small_tempo():
    with pytest.raises(ValueError):
        plan_audio_degrade(tempo=1.02)


def test_plan_rejects_small_pitch():
    with pytest.raises(ValueError):
        plan_audio_degrade(pitch_semitones=0.5)


def test_plan_rejects_high_bitrate():
    with pytest.raises(ValueError):
        plan_audio_degrade(reencode_bitrate="192k")


def test_plan_rejects_out_of_range_tempo():
    with pytest.raises(ValueError):
        plan_audio_degrade(tempo=3.0)


def test_plan_rejects_malformed_bitrate():
    with pytest.raises(ValueError):
        plan_audio_degrade(reencode_bitrate="invalid")


def test_bitrate_kbps_parses_canonical_value():
    from clean_audio import _bitrate_kbps

    assert _bitrate_kbps("96k") == 96
    assert _bitrate_kbps("invalid") == 0
    assert _bitrate_kbps("0k") == 0


# ---------------------------------------------------------------------------
# audio_purify -- availability gating and the destructive chain
# ---------------------------------------------------------------------------


def test_audio_purify_unavailable_without_ffmpeg(tmp_path, monkeypatch):
    import clean_audio

    monkeypatch.setattr(clean_audio, "_ffmpeg_available", lambda: False)
    src = tmp_path / "in.wav"
    src.write_bytes(b"\x00" * 16)
    dest = tmp_path / "out.m4a"
    result = audio_purify(src, dest)
    assert result["available"] is False
    assert "ffmpeg" in result["error"]
    assert not dest.exists()


def test_audio_purify_rejects_missing_file(tmp_path, monkeypatch):
    # Mock ffmpeg present so the test reaches the missing-file branch regardless
    # of whether the runner has ffmpeg installed.
    import clean_audio

    monkeypatch.setattr(clean_audio, "_ffmpeg_available", lambda: True)
    result = audio_purify(tmp_path / "missing.wav", tmp_path / "out.m4a")
    assert result["available"] is False
    assert "not a file" in result["error"]


@needs_ffmpeg
def test_media_has_video_distinguishes_audio_from_video(tmp_path):
    import clean_audio

    audio = tmp_path / "tone.wav"
    subprocess.run(
        [FFMPEG, "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(audio)],
        check=True,
        capture_output=True,
    )
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=64x64:rate=10",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    assert clean_audio.media_has_video(audio) is False
    assert clean_audio.media_has_video(video) is True


def test_media_has_video_indeterminate_when_probe_unavailable(tmp_path, monkeypatch):
    # A failed/inconclusive probe must not be conflated with "no video", so the
    # server never drops a video track via the -vn audio chain.
    import clean_audio

    monkeypatch.setattr(clean_audio, "which", lambda cmd: None)
    assert clean_audio.media_has_video(tmp_path / "clip.mp4") is None


@needs_ffmpeg
def test_audio_purify_applies_destructive_chain(tmp_path):
    src = tmp_path / "tone.wav"
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-ar",
            "44100",
            str(src),
        ],
        check=True,
        capture_output=True,
    )
    in_duration = float(_probe(src)["format"]["duration"])

    dest = tmp_path / "tone.m4a"
    result = audio_purify(src, dest)

    assert result["available"] is True
    assert result["codec"] == "aac"
    assert result["bitrate"] == "96k"
    assert dest.is_file() and dest.stat().st_size > 0

    meta = _probe(dest)
    assert meta["streams"][0]["codec_type"] == "audio"
    assert meta["streams"][0]["codec_name"] == "aac"
    out_duration = float(meta["format"]["duration"])
    assert abs(out_duration - in_duration) > 0.05  # tempo/pitch changed the timing
    assert "atempo=" in result["filter"]
