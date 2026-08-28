"""Tests for clean_video.py (per-frame TrustMark video purification)."""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# plan_frame_purge -- the deterministic vote-collapse core
# ---------------------------------------------------------------------------


def test_plan_default_purifies_all_frames():
    plan = plan_frame_purge(10)
    assert plan["frames_total"] == 10
    assert plan["frames_to_purge"] == 10
    assert plan["fraction"] == 1.0
    assert plan["indices"] == list(range(10))


def test_plan_fraction_matches_requested_count():
    plan = plan_frame_purge(10, frame_fraction=0.5)
    assert plan["frames_to_purge"] == 5
    assert plan["fraction"] == 0.5
    assert len(plan["indices"]) == 5
    assert len(set(plan["indices"])) == 5  # no duplicates


def test_plan_min_fraction_crosses_vote_threshold():
    # To push the residual marked fraction below a 0.25 threshold we must purify
    # more than 1 - 0.25 = 0.75 of the frames. With the default (all frames) we
    # always cross it, so the planner reports the whole set.
    plan = plan_frame_purge(8, vote_threshold=0.25)
    assert plan["frames_to_purge"] == 8
    assert plan["fraction"] == 1.0


def test_plan_spreads_indices_uniformly():
    plan = plan_frame_purge(10, frame_fraction=0.3)
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


# ---------------------------------------------------------------------------
# video_purify -- availability gating (always runs; no backend in CI)
# ---------------------------------------------------------------------------


def test_video_purify_unavailable_without_backend(tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 16)  # not a real video -- gating fails before decode
    dest = tmp_path / "out.mp4"

    result = video_purify(src, dest, remove_pixel="ctrlregen")
    assert result["available"] is False
    assert "CtrlRegen" in result["error"]
    assert not dest.exists()  # no partial output written


def test_video_purify_rejects_unknown_backend(tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 16)
    result = video_purify(src, dest=tmp_path / "out.mp4", remove_pixel="nope")
    assert result["available"] is False
    assert "unknown pixel remover" in result["error"]


@needs_ffmpeg
def test_video_purify_pipeline_produces_playable_output(tmp_path, monkeypatch):
    fake_dir = tmp_path / "noai-watermark"
    fake_dir.mkdir()
    monkeypatch.setenv("NOAI_WATERMARK_DIR", str(fake_dir))

    src = tmp_path / "in.mp4"
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
            str(src),
        ],
        check=True,
        capture_output=True,
    )

    # Stub the per-frame backend to a pass-through copy so the ffmpeg demux /
    # remux orchestration is exercised without a GPU checkout.
    import clean_video

    calls = {"n": 0}

    def fake_clean(frame, output, **kwargs):
        calls["n"] += 1
        return {"available": True, "bytes_out": Path(frame).stat().st_size}

    monkeypatch.setattr(clean_video, "run_ctrlregen_clean", fake_clean)

    dest = tmp_path / "out.mp4"
    result = video_purify(src, dest, remove_pixel="ctrlregen")

    assert result["available"] is True
    assert result["frames_purified"] == result["frames_total"] == 10
    assert calls["n"] == 10
    assert dest.is_file() and dest.stat().st_size > 0

    # The remuxed output must be a readable video.
    probe = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(dest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0
