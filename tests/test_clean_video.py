"""Tests for clean_video.py (per-frame TrustMark video purification)."""

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

from clean_video import (
    _ffmpeg_available,
    plan_frame_purge,
    video_purify,
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


def _probe_duration(path: Path) -> float:
    return float(_probe(path)["format"]["duration"])


def _audio_codecs(path: Path) -> list[str]:
    return [s["codec_name"] for s in _probe(path)["streams"] if s["codec_type"] == "audio"]


# ---------------------------------------------------------------------------
# plan_frame_purge -- the deterministic vote-collapse core
# ---------------------------------------------------------------------------


def test_plan_default_purifies_all_frames():
    plan = plan_frame_purge(10)
    assert plan["frames_total"] == 10
    assert plan["frames_to_purge"] == 10
    assert plan["fraction"] == 1.0
    assert plan["indices"] == list(range(10))


def test_plan_raises_insufficient_fraction_to_cross_threshold():
    # frame_fraction=0.5 with the default vote_threshold=0.5 leaves 50% marked,
    # which is not strictly below 0.5 -- the plan must raise the purge count.
    plan = plan_frame_purge(10, frame_fraction=0.5)
    assert plan["frames_to_purge"] == 6
    assert plan["fraction"] == 0.6
    assert len(plan["indices"]) == 6
    assert len(set(plan["indices"])) == 6  # no duplicates
    assert "raised" in plan["note"]


def test_plan_min_fraction_crosses_vote_threshold():
    plan = plan_frame_purge(8, vote_threshold=0.25)
    assert plan["frames_to_purge"] == 8
    assert plan["fraction"] == 1.0


def test_plan_spreads_indices_uniformly():
    plan = plan_frame_purge(10, frame_fraction=0.6)
    indices = plan["indices"]
    assert indices[0] == 0
    assert indices[-1] == 9
    assert all(0 <= i < 10 for i in indices)


def test_plan_empty_video():
    plan = plan_frame_purge(0)
    assert plan["frames_total"] == 0
    assert plan["indices"] == []
    assert "empty" in plan["note"]


def test_plan_single_frame():
    plan = plan_frame_purge(1)
    assert plan["frames_to_purge"] == 1
    assert plan["indices"] == [0]


def test_plan_clamps_fraction_over_one():
    plan = plan_frame_purge(5, frame_fraction=3.0)
    assert plan["fraction"] == 1.0
    assert plan["frames_to_purge"] == 5


def test_plan_rejects_negative_inputs():
    with pytest.raises(ValueError):
        plan_frame_purge(-1)
    with pytest.raises(ValueError):
        plan_frame_purge(5, vote_threshold=2)
    with pytest.raises(ValueError):
        plan_frame_purge(5, frame_fraction=-0.1)


def test_plan_vote_threshold_zero_rejected():
    with pytest.raises(ValueError):
        plan_frame_purge(5, vote_threshold=0)


def test_spread_indices_single_index_does_not_divide_by_zero():
    from clean_video import _spread_indices

    assert _spread_indices(10, 1) == [0]
    assert _spread_indices(10, 0) == []


# ---------------------------------------------------------------------------
# video_purify -- availability gating (backend checked before ffmpeg)
# ---------------------------------------------------------------------------


def test_video_purify_unavailable_without_backend(tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 16)  # not a real video -- gating fails before decode
    dest = tmp_path / "out.mp4"

    result = video_purify(src, dest, remove_pixel="ctrlregen")
    assert result["available"] is False
    assert "CtrlRegen" in result["error"]  # backend error, even without ffmpeg
    assert not dest.exists()  # no partial output written


def test_video_purify_rejects_unknown_backend(tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 16)
    result = video_purify(src, dest=tmp_path / "out.mp4", remove_pixel="nope")
    assert result["available"] is False
    assert "unknown pixel remover" in result["error"]


@needs_ffmpeg
def test_video_purify_pipeline_preserves_audio_and_duration(tmp_path):
    # A VFR source (1s @10fps + 1s @30fps) with AAC audio. The pipeline must
    # honor an explicit backend dir (no env var), purify every frame, and remux
    # while preserving the source duration and copying the audio stream.
    fake_dir = tmp_path / "fake-backend"
    fake_dir.mkdir()

    src = tmp_path / "in.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=1:size=64x64:rate=10",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=1:size=64x64:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-map",
            "2:a",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-fps_mode",
            "vfr",
            str(src),
        ],
        check=True,
        capture_output=True,
    )
    in_duration = _probe_duration(src)

    # Stub the per-frame backend to a pass-through copy so the ffmpeg demux /
    # remux orchestration is exercised without a GPU checkout.
    import clean_video

    calls = {"n": 0}

    def fake_clean(frame, output, **kwargs):
        calls["n"] += 1
        return {"available": True, "bytes_out": Path(frame).stat().st_size}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(clean_video, "run_ctrlregen_clean", fake_clean)
    try:
        dest = tmp_path / "out.mp4"
        result = video_purify(src, dest, remove_pixel="ctrlregen", ctrlregen_dir=str(fake_dir))
    finally:
        monkeypatch.undo()

    assert result["available"] is True
    # 10 + 30 frames decoded and purified by the stub.
    assert result["frames_purified"] == result["frames_total"] == 40
    assert calls["n"] == 40
    assert dest.is_file() and dest.stat().st_size > 0

    out_duration = _probe_duration(dest)
    assert abs(out_duration - in_duration) <= 0.25  # timing not flattened away
    assert "aac" in _audio_codecs(dest)  # audio copied, not re-encoded
