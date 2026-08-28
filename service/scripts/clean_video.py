#!/usr/bin/env python3
"""Optional per-frame pixel purification for video (MP4/MOV).

provcheck runs TrustMark-B per frame and votes across frames, so a single-frame
purification is not enough: enough frames must be cleared that the temporal vote
collapses. This module demuxes a video to frames, routes each selected frame
through the same pixel-domain remover used for images (CtrlRegen or
DiffusionPurification), and remuxes the purified frames with the original audio.

ffmpeg is a runtime dependency of the service image (installed in the
Dockerfile); the purification backend is still an optional external GPU checkout.
When ffmpeg or the backend is absent this reports ``available: False`` and
performs no work -- it never silently returns a partially-purified video.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import (  # noqa: E402
    subprocess_creationflags,
    subprocess_preexec_fn,
    which,
)
from image_meta import run_ctrlregen_clean, run_markdiffusion_purify  # noqa: E402

_BACKEND_LABELS = {"ctrlregen": "CtrlRegen", "diffusion": "DiffusionPurification"}
_BACKEND_ENV = {
    "ctrlregen": ("NOAI_WATERMARK_DIR", "CtrlRegen"),
    "diffusion": ("MARKDIFFUSION_DIR", "DiffusionPurification"),
}


def _ffmpeg_available() -> bool:
    """True when both ffmpeg and ffprobe are on PATH."""
    return which("ffmpeg") is not None and which("ffprobe") is not None


def _backend_configured(remove_pixel: str) -> tuple[bool, str]:
    env_var, label = _BACKEND_ENV.get(remove_pixel, (None, None))
    if env_var is None:
        return False, f"unknown pixel remover: {remove_pixel}"
    raw = os.environ.get(env_var)
    if not raw:
        return False, f"{label} not configured (set {env_var})"
    d = Path(raw).expanduser()
    if not d.is_dir():
        return False, f"{label} dir not found: {d}"
    return True, ""


def _spread_indices(total: int, k: int) -> list[int]:
    """Return *k* 0-based indices evenly spread across *total* frames, deduped."""
    if k <= 0 or total <= 0:
        return []
    if k >= total:
        return list(range(total))
    indices: list[int] = []
    for i in range(k):
        indices.append(round(i * (total - 1) / (k - 1)))
    seen: set[int] = set()
    out: list[int] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def plan_frame_purge(
    frame_count: int, *, vote_threshold: float = 0.5, frame_fraction: float | None = None
) -> dict[str, object]:
    """Decide which frames to purify so the temporal vote collapses.

    A frame we do not purify is assumed to still carry the mark, so to push the
    vote below the winning threshold we must leave fewer than ``vote_threshold``
    of the frames un-purified -- i.e. purify more than ``1 - vote_threshold`` of
    them. ``frame_fraction`` defaults to ``1.0`` (purify every frame), which
    collapses the vote whenever a single purification pass defeats TrustMark per
    frame. Selected indices are spread evenly so any temporal vote window sees
    enough purified frames regardless of window alignment.
    """
    if frame_count < 0:
        raise ValueError("frame_count must be >= 0")
    if not 0 <= vote_threshold <= 1:
        raise ValueError("vote_threshold must be in [0, 1]")
    if frame_count == 0:
        return {
            "frames_total": 0,
            "frames_to_purge": 0,
            "fraction": 0.0,
            "indices": [],
            "note": "empty video: no frames to purify",
        }

    fraction = 1.0 if frame_fraction is None else frame_fraction
    if fraction < 0:
        raise ValueError("frame_fraction must be >= 0")
    if fraction > 1:
        fraction = 1.0

    to_purge = max(0, min(frame_count, round(fraction * frame_count)))
    if to_purge == 0:
        return {
            "frames_total": frame_count,
            "frames_to_purge": 0,
            "fraction": 0.0,
            "indices": [],
            "note": "no frames purified (frame_fraction = 0)",
        }
    if to_purge >= frame_count:
        indices = list(range(frame_count))
        fraction = 1.0
        note = "all frames purified"
    else:
        indices = _spread_indices(frame_count, to_purge)
        fraction = to_purge / frame_count
        note = (
            f"purified {to_purge}/{frame_count} frames ({fraction:.3f}); "
            f"{frame_count - to_purge} frames left marked, below vote threshold {vote_threshold}"
        )
    return {
        "frames_total": frame_count,
        "frames_to_purge": to_purge,
        "fraction": fraction,
        "indices": indices,
        "note": note,
    }


def _run_ffmpeg(cmd: list[str], timeout: int, what: str) -> tuple[int, str]:
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=subprocess_preexec_fn,
            check=False,
            creationflags=subprocess_creationflags,
        )
    except subprocess.TimeoutExpired:
        return 1, f"{what} timed out after {timeout}s"
    except Exception as e:  # surface any subprocess failure
        return 1, f"{what} failed: {e}"
    return r.returncode, r.stderr or ""


def _probe_fps(path: Path) -> float:
    ffprobe = which("ffprobe")
    if not ffprobe:
        return 25.0
    try:
        r = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            preexec_fn=subprocess_preexec_fn,
            creationflags=subprocess_creationflags,
        )
        val = (r.stdout or "").strip()
        if "/" in val:
            num, den = val.split("/")
            if float(den) != 0:
                return float(num) / float(den)
        f = float(val)
        return f if f > 0 else 25.0
    except Exception:  # take the default when ffprobe is missing or odd
        return 25.0


def video_purify(
    path: Path,
    dest: Path,
    *,
    remove_pixel: str,
    vote_threshold: float = 0.5,
    frame_fraction: float | None = None,
    ctrlregen_dir: str | None = None,
    ctrlregen_strength: float = 0.25,
    ctrlregen_steps: int = 50,
    ctrlregen_device: str | None = None,
    ctrlregen_seed: int | None = None,
    markdiffusion_dir: str | None = None,
    markdiffusion_strength: float = 0.3,
    markdiffusion_model: str | None = None,
    markdiffusion_size: int = 512,
    markdiffusion_steps: int = 50,
    markdiffusion_device: str | None = None,
    timeout: int = 3600,
) -> dict[str, object]:
    """Purify video frames so provcheck's per-frame TrustMark temporal vote drops.

    Returns ``{"available": False, "error": ...}`` without side effects when ffmpeg
    or the backend is unavailable or a frame cannot be purified; a successful run
    returns ``{"available": True, ...}``.
    """
    if not _ffmpeg_available():
        return {
            "available": False,
            "error": "ffmpeg/ffprobe not available (install ffmpeg or use the service image)",
        }
    ok, err = _backend_configured(remove_pixel)
    if not ok:
        return {"available": False, "error": err}

    input_path = Path(path)
    output_path = Path(dest)
    if not input_path.is_file():
        return {"available": False, "error": f"not a file: {input_path}"}

    ffmpeg = which("ffmpeg")
    assert ffmpeg is not None

    with tempfile.TemporaryDirectory(prefix="wm-video-") as _tmpdir:
        tmp = Path(_tmpdir)
        frames_dir = tmp / "frames"
        frames_dir.mkdir()

        rc, stderr = _run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-fps_mode",
                "vfr",  # decode every frame (modern replacement for -vsync 0)
                "-start_number",
                "0",
                "-f",
                "image2",
                str(frames_dir / "frame_%06d.png"),
                "-nostdin",
            ],
            timeout,
            "ffmpeg frame extraction",
        )
        if rc != 0:
            return {"available": False, "error": (stderr or "").strip()[-2000:]}

        frames = sorted(frames_dir.glob("frame_*.png"))
        plan = plan_frame_purge(len(frames), vote_threshold=vote_threshold, frame_fraction=frame_fraction)
        indices = plan["indices"]
        assert isinstance(indices, list)

        frames_purified = 0
        for idx in indices:
            frame = frames_dir / f"frame_{idx:06d}.png"
            if remove_pixel == "ctrlregen":
                res = run_ctrlregen_clean(
                    frame,
                    frame,
                    upstream_dir=ctrlregen_dir,
                    strength=ctrlregen_strength,
                    steps=ctrlregen_steps,
                    device=ctrlregen_device,
                    seed=ctrlregen_seed,
                    timeout=timeout,
                )
            else:
                res = run_markdiffusion_purify(
                    frame,
                    frame,
                    upstream_dir=markdiffusion_dir,
                    strength=markdiffusion_strength,
                    model=markdiffusion_model,
                    size=markdiffusion_size,
                    steps=markdiffusion_steps,
                    device=markdiffusion_device,
                    timeout=timeout,
                )
            if not res.get("available"):
                label = _BACKEND_LABELS[remove_pixel]
                return {
                    "available": False,
                    "error": f"{label} failed on frame {idx}: {res.get('error', 'unknown error')}",
                }
            frames_purified += 1

        fps = _probe_fps(input_path)
        fd, tmp_out = tempfile.mkstemp(dir=output_path.parent, suffix=f".remux{output_path.suffix}")
        os.close(fd)
        try:
            rc, stderr = _run_ffmpeg(
                [
                    ffmpeg,
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(frames_dir / "frame_%06d.png"),
                    "-i",
                    str(input_path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0?",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-shortest",
                    str(tmp_out),
                    "-nostdin",
                ],
                timeout,
                "ffmpeg remux",
            )
            if rc != 0:
                return {"available": False, "error": (stderr or "").strip()[-2000:]}
            os.replace(tmp_out, output_path)
        finally:
            if os.path.exists(tmp_out):
                os.unlink(tmp_out)

    return {
        "available": True,
        "format": "mp4",
        "remove_pixel": remove_pixel,
        "frames_total": len(frames),
        "frames_purified": frames_purified,
        "frame_fraction": plan["fraction"],
        "vote_threshold": vote_threshold,
        "note": plan["note"],
        "bytes_out": output_path.stat().st_size,
    }


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Input video (MP4/MOV)")
    p.add_argument("-o", "--output", type=Path, help="Output path (default: *.video.*)")
    p.add_argument("--remove-pixel", choices=["ctrlregen", "diffusion"], required=True)
    p.add_argument("--vote-threshold", type=float, default=0.5, help="Temporal-vote threshold (default: 0.5)")
    p.add_argument("--frame-fraction", type=float, default=None, help="Fraction of frames to purify (default: 1.0)")
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    from common import cleaned_path

    output = args.output or cleaned_path(args.path, ".video")
    result = video_purify(
        args.path,
        output,
        remove_pixel=args.remove_pixel,
        vote_threshold=args.vote_threshold,
        frame_fraction=args.frame_fraction,
    )
    if args.json:
        import json

        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        if result.get("available"):
            print(
                f"{_BACKEND_LABELS[args.remove_pixel]}: purified "
                f"{result['frames_purified']}/{result['frames_total']} frames -> {output}"
            )
        else:
            print(f"unavailable: {result.get('error', 'unknown error')}", file=sys.stderr)
    return 0 if result.get("available") else 1


if __name__ == "__main__":
    raise SystemExit(main())
