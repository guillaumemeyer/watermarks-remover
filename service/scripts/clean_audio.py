#!/usr/bin/env python3
"""Optional destructive transform chain for audio provenance watermarks.

provcheck survives ordinary transcoding (its own survival range), so a plain
re-encode does not reliably remove silentcipher / AudioSeal / WavMark. Only a
"remix"-style degradation -- tempo + pitch + EQ plus a low-bitrate lossy
re-encode -- falls in the range provcheck documents as not surviving. This module
applies that chain with ffmpeg.

It is intentionally destructive (changes pitch, tempo, quality and duration).
Neural re-synthesis is a separate, heavier path and is out of scope here.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from clean_video import _ffmpeg_available, _run_ffmpeg  # noqa: E402
from common import safe_arg, subprocess_creationflags, subprocess_preexec_fn, which  # noqa: E402

# Formats detect_av_format reports for audio-only containers.
AUDIO_FORMATS = {"wav", "mp3", "flac"}

# Audio filename extensions, incl. audio-only MP4/MOV containers (m4a/aac/ogg/opus)
# that detect_av_format reports as "mp4".
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}

# Audio container -> codec for the destructive re-encode. Lossy codecs support a
# bitrate; pcm/flac are lossless (the transform is still applied, just not a
# low-bitrate re-encode).
_CODEC_BY_EXT = {
    ".m4a": "aac",
    ".aac": "aac",
    ".mp4": "aac",
    ".ogg": "libopus",
    ".opus": "libopus",
    ".mp3": "libmp3lame",
    ".wav": "pcm_s16le",
    ".flac": "flac",
}
_LOSSY_CODECS = {"aac", "libopus", "libmp3lame"}

# Parameters that provcheck's survival table shows still survive. To exit that
# envelope we must exceed them.
_TEMPO_LIMIT = 1.05  # beyond +/-5% tempo
_PITCH_LIMIT = 1.0  # beyond +/-1 semitone
_BITRATE_FLOOR_K = 128  # AAC/Opus below 128 kbps


def is_audio_format(fmt: str) -> bool:
    """True when *fmt* (a detect_av_format result) is an audio-only container."""
    return fmt in AUDIO_FORMATS


def is_audio_name(name: str | None) -> bool:
    """True when a filename's extension is a known audio container."""
    return (Path(name or "").suffix or "").lower() in AUDIO_EXTS


def _pitch_factor(semitones: float) -> float:
    return 2.0 ** (semitones / 12.0)


def _build_filter(tempo: float, pitch_semitones: float, sample_rate: int) -> str:
    """The destructive ffmpeg filter graph: tempo + pitch + EQ tilt."""
    sr = sample_rate or 44100
    return (
        f"atempo={tempo},"
        f"asetrate={int(sr * _pitch_factor(pitch_semitones))},aresample={sr},"
        "highpass=f=35,lowpass=f=14000,treble=g=-8"
    )


def _bitrate_kbps(bitrate: str) -> int:
    """Parse a canonical bitrate like ``96k`` -> 96 (kbps), or 0 if malformed."""
    match = re.fullmatch(r"([1-9]\d*)k", str(bitrate).strip())
    return int(match.group(1)) if match else 0


def plan_audio_degrade(
    tempo: float = 1.08, pitch_semitones: float = 2.0, reencode_bitrate: str = "96k"
) -> dict[str, object]:
    """Validate the destructive chain and describe which survival limits it exits."""
    if not 0.5 <= tempo <= 2.0:
        raise ValueError("tempo must be in [0.5, 2.0]")
    if abs(tempo - 1.0) <= (_TEMPO_LIMIT - 1.0):
        raise ValueError("tempo change must exceed +/-5% to exit the audio survival envelope")
    if abs(pitch_semitones) <= _PITCH_LIMIT:
        raise ValueError(
            "pitch shift must exceed +/-1 semitone to exit the audio survival envelope"
        )
    kbps = _bitrate_kbps(reencode_bitrate)
    if not kbps:
        raise ValueError("reencode bitrate must be a positive value such as 96k")
    if kbps >= _BITRATE_FLOOR_K:
        raise ValueError("lossy re-encode bitrate must be below 128 kbps")

    exceeded = ["tempo change >5%"]
    if abs(pitch_semitones) > _PITCH_LIMIT:
        exceeded.append(f"pitch shift >1 semitone ({pitch_semitones:+.1f})")
    if kbps and kbps < _BITRATE_FLOOR_K:
        exceeded.append(f"lossy re-encode below {_BITRATE_FLOOR_K} kbps ({kbps} kbps)")

    return {
        "tempo": tempo,
        "pitch_semitones": pitch_semitones,
        "bitrate": reencode_bitrate,
        "survival_limits_exceeded": exceeded,
        "note": (
            "destructive transform chain: tempo + pitch + EQ + lossy re-encode; "
            "falls in provcheck's documented not-surviving range, but is not "
            "vendor-detector-verified (we do not ship silentcipher/AudioSeal/WavMark decoders)"
        ),
    }


def _probe_sample_rate(path: Path) -> int:
    """Probe the first audio stream's sample rate (fallback 44100)."""
    ffprobe = which("ffprobe")
    if not ffprobe:
        return 44100
    try:
        r = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate",
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
    except Exception:  # take the default when ffprobe is missing or odd
        return 44100
    try:
        return int((r.stdout or "").strip())
    except ValueError:
        return 44100


def media_has_video(path: Path) -> bool | None:
    """Probe whether the media has a video stream.

    Returns ``None`` when the probe is inconclusive (ffprobe unavailable or
    failed), so the caller can distinguish "no video present" from "could not
    tell" and avoid running the ``-vn`` audio chain on a video-bearing payload.
    """
    ffprobe = which("ffprobe")
    if not ffprobe:
        return None
    try:
        r = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                safe_arg(str(path)),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            preexec_fn=subprocess_preexec_fn,
            creationflags=subprocess_creationflags,
        )
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return "video" in (r.stdout or "").splitlines()


def audio_purify(
    path: Path,
    dest: Path,
    *,
    tempo: float = 1.08,
    pitch_semitones: float = 2.0,
    reencode_bitrate: str = "96k",
    codec: str | None = None,
    timeout: int = 1800,
) -> dict[str, object]:
    """Apply the destructive transform chain to *path* and write to *dest*.

    Returns ``{"available": False, "error": ...}`` without side effects when ffmpeg
    is unavailable; a successful run returns ``{"available": True, ...}``.
    """
    plan = plan_audio_degrade(tempo, pitch_semitones, reencode_bitrate)
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

    sr = _probe_sample_rate(input_path)
    filter_graph = _build_filter(tempo, pitch_semitones, sr)
    codec = codec or _CODEC_BY_EXT.get(output_path.suffix.lower(), "aac")
    lossy = codec in _LOSSY_CODECS
    kbps = _bitrate_kbps(reencode_bitrate)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        safe_arg(str(input_path)),
        "-af",
        filter_graph,
        "-vn",
        "-c:a",
        codec,
    ]
    if lossy and kbps:
        cmd += ["-b:a", reencode_bitrate]
    cmd += [safe_arg(str(output_path)), "-nostdin"]

    rc, stderr = _run_ffmpeg(cmd, timeout, "ffmpeg audio degradation")
    if rc != 0:
        err = (stderr or "").strip()
        # ffmpeg's stderr is commonly led by its configure/build banner; keep the
        # trailing lines where the actual failure reason appears so it isn't buried.
        err_lines = [ln for ln in err.splitlines() if ln.strip()]
        err = "\n".join(err_lines[-8:]) if err_lines else err
        return {"available": False, "error": err[-2000:]}

    return {
        "available": True,
        "format": "audio",
        "codec": codec,
        "bitrate": reencode_bitrate if lossy else None,
        "tempo": tempo,
        "pitch_semitones": pitch_semitones,
        "filter": filter_graph,
        "note": plan["note"],
        "survival_limits_exceeded": plan["survival_limits_exceeded"],
        "bytes_out": output_path.stat().st_size,
    }


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Input audio (WAV/MP3/FLAC)")
    p.add_argument("-o", "--output", type=Path, help="Output path (default: *.audio.m4a)")
    p.add_argument("--tempo", type=float, default=1.08, help="Tempo factor (default: 1.08)")
    p.add_argument(
        "--pitch-semitones", type=float, default=2.0, help="Pitch shift in semitones (default: 2.0)"
    )
    p.add_argument(
        "--reencode-bitrate", type=str, default="96k", help="Lossy re-encode bitrate (default: 96k)"
    )
    p.add_argument("--codec", type=str, default=None, help="Output codec override")
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    output = args.output or args.path.with_name(f"{args.path.stem}.audio.m4a")
    result = audio_purify(
        args.path,
        output,
        tempo=args.tempo,
        pitch_semitones=args.pitch_semitones,
        reencode_bitrate=args.reencode_bitrate,
        codec=args.codec,
    )
    if args.json:
        import json

        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        if result.get("available"):
            print(
                f"audio destructive chain: {result['tempo']}x tempo, "
                f"{result['pitch_semitones']:+.1f} semitones, "
                f"{result['codec']} -> {output}"
            )
        else:
            print(f"unavailable: {result.get('error', 'unknown error')}", file=sys.stderr)
    return 0 if result.get("available") else 1


if __name__ == "__main__":
    raise SystemExit(main())
