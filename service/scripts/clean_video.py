#!/usr/bin/env python3
"""Optional per-frame pixel purification for video (MP4/MOV).

provcheck runs TrustMark-B per frame and votes across frames, so a single-frame
purification is not enough: enough frames must be cleared that the temporal vote
collapses. This module demuxes a video to frames, routes each selected frame
through the same pixel-domain remover used for images (CtrlRegen or
DiffusionPurification), and remuxes the purified frames with the original audio,
preserving source frame timing.

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
    safe_arg,
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


def _backend_configured(remove_pixel: str, explicit_dir: str | None = None) -> tuple[bool, str]:
    """Validate that the requested pixel backend is configured.

    *explicit_dir* (when supplied) overrides the environment variable; otherwise
    the matching env var is consulted, mirroring the per-frame remover helpers.
    """
    env_var, label = _BACKEND_ENV.get(remove_pixel, (None, None))
    if env_var is None:
        return False, f"unknown pixel remover: {remove_pixel}"
    raw = explicit_dir or os.environ.get(env_var)
    if not raw:
        return False, f"{label} not configured (set {env_var} or pass the explicit dir)"
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
    if k == 1:
        return [0]
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
    vote below the winning threshold we must leave strictly fewer than
    ``vote_threshold`` of the frames marked -- i.e. purify more than
    ``1 - vote_threshold`` of them. ``frame_fraction`` defaults to ``1.0``
    (purify every frame). If an explicit ``frame_fraction`` is too small to
    cross the threshold, the plan raises the purge count to the minimum that
    does (it never returns a plan that would leave the vote intact). Selected
    indices are spread evenly so any temporal vote window sees enough purified
    frames regardless of window alignment.
    """
    if frame_count < 0:
        raise ValueError("frame_count must be >= 0")
    if not 0 < vote_threshold <= 1:
        raise ValueError("vote_threshold must be in (0, 1]")
    if vote_threshold == 0:
        raise ValueError("vote_threshold=0 is unattainable: no finite purge leaves 0 frames marked")
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
    # Minimum purge that leaves the residual marked fraction strictly below the
    # threshold: to_purge > frame_count * (1 - vote_threshold).
    min_to_purge = min(frame_count, int(frame_count * (1 - vote_threshold)) + 1)
    raised = to_purge < min_to_purge
    to_purge = max(to_purge, min_to_purge)

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
        if raised:
            note = (
                f"raised to {to_purge}/{frame_count} frames ({fraction:.3f}) so that the "
                f"{frame_count - to_purge} un-purified frames stay below vote threshold {vote_threshold}"
            )
        else:
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
    """Probe the average source frame rate, falling back to a sane default."""
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
                safe_arg(str(path)),
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


def _frame_durations(path: Path, frame_count: int) -> list[float]:
    """Per-frame presentation durations (seconds) from the source PTS.

    Returns an empty list when durations are unavailable, so the caller falls
    back to a constant frame rate rather than silently mangling the timing.
    """
    if frame_count <= 0:
        return []
    ffprobe = which("ffprobe")
    if not ffprobe:
        return []
    try:
        r = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=pts_time",
                "-of",
                "csv=p=0",
                safe_arg(str(path)),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            preexec_fn=subprocess_preexec_fn,
            creationflags=subprocess_creationflags,
        )
    except Exception:  # probe failures fall back to uniform timing
        return []
    pts: list[float] = []
    for tok in (r.stdout or "").split():
        try:
            pts.append(float(tok))
        except ValueError:
            return []
    if len(pts) < frame_count:
        return []
    durations = [pts[i + 1] - pts[i] for i in range(frame_count - 1)]
    if any(d <= 0 for d in durations):
        return []
    durations.append(durations[-1])  # stable duration for the final frame
    return durations


def _write_frame_list(frames_dir: Path, durations: list[float]) -> Path:
    """Write a concat-demuxer list that preserves per-frame durations.

    The final frame is listed once more without a duration because the concat
    demuxer otherwise drops it.
    """
    list_path = frames_dir / "frames.txt"
    lines: list[str] = []
    last = len(durations) - 1
    for i, dur in enumerate(durations):
        lines.append(f"file '{frames_dir / f'frame_{i:06d}.png'}'")
        if i != last:
            lines.append(f"duration {dur:.6f}")
    lines.append(f"file '{frames_dir / f'frame_{last:06d}.png'}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_path


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

    Returns ``{"available": False, "error": ...}`` without side effects when the
    backend or ffmpeg is unavailable or a frame cannot be purified; a successful
    run returns ``{"available": True, ...}``.
    """
    ok, err = _backend_configured(
        remove_pixel,
        explicit_dir=ctrlregen_dir if remove_pixel == "ctrlregen" else markdiffusion_dir,
    )
    if not ok:
        return {"available": False, "error": err}
    if not _ffmpeg_available():
        return {
            "available": False,
            "error": "ffmpeg/ffprobe not available (install ffmpeg or use the service image)",
        }

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
                safe_arg(str(input_path)),
                "-fps_mode",
                "vfr",  # decode every frame (modern replacement for -vsync 0)
                "-start_number",
                "0",
                "-f",
                "image2",
                safe_arg(str(frames_dir / "frame_%06d.png")),
                "-nostdin",
            ],
            timeout,
            "ffmpeg frame extraction",
        )
        if rc != 0:
            return {"available": False, "error": (stderr or "").strip()[-2000:]}

        frames = sorted(frames_dir.glob("frame_*.png"))
        if not frames:
            return {"available": False, "error": "no video frames extracted (not a valid video?)"}
        plan = plan_frame_purge(
            len(frames), vote_threshold=vote_threshold, frame_fraction=frame_fraction
        )
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

        durations = _frame_durations(input_path, len(frames))
        if not durations:
            durations = [1.0 / _probe_fps(input_path)] * len(frames)
        frame_list = _write_frame_list(frames_dir, durations)

        fd, tmp_out = tempfile.mkstemp(dir=output_path.parent, suffix=f".remux{output_path.suffix}")
        os.close(fd)
        try:
            rc, stderr = _run_ffmpeg(
                [
                    ffmpeg,
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    safe_arg(str(frame_list)),
                    "-i",
                    safe_arg(str(input_path)),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0?",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "copy",
                    safe_arg(str(tmp_out)),
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
    p.add_argument(
        "--vote-threshold", type=float, default=0.5, help="Temporal-vote threshold (default: 0.5)"
    )
    p.add_argument(
        "--frame-fraction",
        type=float,
        default=None,
        help="Fraction of frames to purify (default: 1.0)",
    )
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
