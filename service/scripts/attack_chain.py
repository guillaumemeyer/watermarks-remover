#!/usr/bin/env python3
"""Attack chain coordinator for the WAVES robustness-evaluation toolchain.

Selects and sequences removal engines per watermark type, runs them in order,
and verifies each step with the forensic stealth detector.

Exit codes:
  0  chain completed
  1  runtime error
  2  bad input
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import (  # noqa: E402
    cleaned_path,
    emit_json,
    eprint,
    safe_write_bytes,
    subprocess_creationflags,
)

_FLUX3_ENGINES = frozenset({"clean_flux3.py"})
_FLUX3_TIERS = [(768, "768"), (1024, "1k"), (1536, "1.5k"), (2048, "2k"), (4096, "4k")]


def _pick_flux3_resolution(width: int, height: int) -> str:
    """Choose the Flux3 tier where the resize-back is at most ~25% upscale."""
    long_side = max(width, height)
    for threshold, tier in _FLUX3_TIERS:
        if threshold >= long_side * 0.75:
            return tier
    return "4k"


def _resolve_api_key(explicit: str | None) -> str:
    """Return the OpenRouter API key from arg, env, or ~/.claude/openrouter.env."""
    if explicit:
        return explicit
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    env_file = Path.home() / ".claude" / "openrouter.env"
    try:
        if env_file.is_file():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    except (OSError, UnicodeDecodeError) as exc:
        eprint(f"could not read {env_file}: {exc}")
    return ""


ATTACK_CHAINS: dict[str, list[dict[str, Any]]] = {
    "tree_ring": [
        # Across 5 runs 0.40 held L1 80.04-81.96 against the 78 threshold
        # (worst margin 2.0); 0.30 fell to 78.78. Grain stays within 1.12x
        # of the source image's own.
        {"engine": "clean_sdedit.py", "args": ["--strength", "0.40"]},
    ],
    "gaussian_shading": [
        {"engine": "clean_marknull.py", "args": ["--mode", "fast"]},
        {"engine": "clean_fmdiffwa.py", "args": ["--intensity", "0"]},
        {"engine": "clean_flux3.py", "args": ["--strength", "minimal"]},
    ],
    "stegastamp": [
        {"engine": "clean_fmdiffwa.py", "args": ["--intensity", "0"]},
        {
            "engine": "clean_synthid_bypass.py",
            "args": ["--pipeline", "vae+squeeze"],
        },
        {"engine": "clean_flux3.py", "args": ["--strength", "minimal"]},
    ],
    "stable_signature": [
        {"engine": "clean_nfpa.py", "args": []},
        {"engine": "clean_fmdiffwa.py", "args": ["--intensity", "0"]},
        {"engine": "clean_flux3.py", "args": ["--strength", "minimal"]},
    ],
    "synthid": [
        {"engine": "clean_flux3.py", "args": ["--strength", "normal"]},
    ],
    "hidden": [
        {"engine": "clean_flux3.py", "args": ["--strength", "normal"]},
    ],
    "trustmark": [
        {"engine": "clean_fmdiffwa.py", "args": ["--intensity", "0"]},
        {"engine": "clean_nfpa.py", "args": []},
        {"engine": "clean_flux3.py", "args": ["--strength", "minimal"]},
    ],
    "unknown": [
        {"engine": "clean_flux3.py", "args": ["--strength", "normal"]},
    ],
}

VALID_SCHEMES = frozenset(ATTACK_CHAINS.keys())


class ChainResult(NamedTuple):
    scheme: str
    steps_run: int
    steps_total: int
    stealth_score: float
    elapsed_s: float
    output_path: str
    complete: bool = True


def _run_engine(
    script: str,
    input_path: Path,
    output_path: Path,
    extra_args: list[str],
    device: str | None,
    timeout: int,
    api_key: str = "",
    input_size: tuple[int, int] | None = None,
    max_cost: float | None = None,
    provider: str | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / script),
        str(input_path),
        "-o",
        str(output_path),
        "--json",
    ]
    if device and script not in _FLUX3_ENGINES:
        cmd.extend(["--device", device])
    if script in _FLUX3_ENGINES:
        if input_size and "--resolution" not in extra_args:
            cmd.extend(["--resolution", _pick_flux3_resolution(*input_size)])
        if max_cost is not None and "--max-cost" not in extra_args:
            cmd.extend(["--max-cost", str(max_cost)])
        if provider and "--provider" not in extra_args:
            cmd.extend(["--provider", provider])
    cmd.extend(extra_args)
    env = None
    if script in _FLUX3_ENGINES and api_key:
        env = {**os.environ, "OPENROUTER_API_KEY": api_key}
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        creationflags=subprocess_creationflags,
        env=env,
    )


def _run_forensic(image_path: Path) -> float:
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "forensic_check.py"), str(image_path), "--json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            creationflags=subprocess_creationflags,
        )
    except subprocess.TimeoutExpired:
        eprint(f"forensic check timed out for {image_path}")
        return 1.0
    if result.returncode == 0:
        try:
            report = json.loads(result.stdout)
            return float(report.get("stealth_score", 1.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return 1.0


def run_chain(
    image_path: Path,
    scheme: str,
    output_path: Path | None = None,
    device: str | None = None,
    timeout: int = 600,
    stealth_threshold: float = 0.7,
    api_key: str | None = None,
    max_cost: float | None = None,
    flux3_resolution: str | None = None,
    flux3_strength: str | None = None,
    flux3_timeout: int | None = None,
    flux3_provider: str | None = None,
) -> ChainResult:
    if scheme not in VALID_SCHEMES:
        raise ValueError(f"unknown scheme: {scheme}; valid: {sorted(VALID_SCHEMES)}")

    chain = ATTACK_CHAINS[scheme]
    has_flux3 = any(s["engine"] in _FLUX3_ENGINES for s in chain)
    resolved_key = _resolve_api_key(api_key) if has_flux3 else ""
    t0 = time.monotonic()
    current = image_path
    steps_run = 0
    flux3_failed = False
    pillow_available = True
    original_size: tuple[int, int] | None = None
    try:
        from PIL import Image as _Img
        with _Img.open(image_path) as _im:
            original_size = _im.size
    except ImportError:
        pillow_available = False
        if has_flux3:
            eprint("Pillow not installed; skipping Flux3 (cannot guarantee dimension match)")
    except Exception as exc:
        eprint(f"could not read input size for post-chain resize: {exc}")

    import shutil
    import tempfile

    work_dir = Path(tempfile.mkdtemp(prefix="chain_"))

    try:
        for i, step in enumerate(chain):
            script = step["engine"]
            if not (SCRIPTS_DIR / script).is_file():
                eprint(f"skipping {script}: not installed")
                if script in _FLUX3_ENGINES:
                    flux3_failed = True
                continue
            if script in _FLUX3_ENGINES and not pillow_available:
                eprint(f"skipping {script}: Pillow needed for post-Flux3 resize")
                flux3_failed = True
                continue

            step_out = output_path if (i == len(chain) - 1 and output_path) else None
            if step_out is None:
                step_out = work_dir / f"{image_path.stem}.step{i}{image_path.suffix}"

            step_args = list(step["args"])
            if script in _FLUX3_ENGINES:
                if flux3_strength and "--strength" in step_args:
                    idx = step_args.index("--strength")
                    step_args[idx + 1] = flux3_strength
                if flux3_resolution:
                    step_args.extend(["--resolution", flux3_resolution])

            step_timeout = (flux3_timeout if flux3_timeout and script in _FLUX3_ENGINES
                            else timeout)
            try:
                result = _run_engine(
                    script, current, step_out, step_args, device, step_timeout,
                    api_key=resolved_key, input_size=original_size, max_cost=max_cost,
                    provider=flux3_provider if script in _FLUX3_ENGINES else None,
                )
            except subprocess.TimeoutExpired:
                eprint(f"{script} timed out after {step_timeout}s")
                remaining_has_flux3 = script in _FLUX3_ENGINES or any(
                    s["engine"] in _FLUX3_ENGINES for s in chain[i + 1 :]
                )
                if remaining_has_flux3:
                    flux3_failed = True
                break

            if result.returncode == 0 and step_out.is_file():
                current = step_out
                steps_run += 1
            elif result.returncode == 3:
                eprint(f"{script} unavailable (exit 3), skipping")
                if script in _FLUX3_ENGINES:
                    flux3_failed = True
                continue
            else:
                eprint(f"{script} failed (exit {result.returncode}): {result.stderr[:200]}")
                if script in _FLUX3_ENGINES:
                    flux3_failed = True
                continue

            remaining_has_flux3 = any(
                s["engine"] in _FLUX3_ENGINES for s in chain[i + 1 :]
            )
            if not remaining_has_flux3:
                stealth = _run_forensic(current)
                if stealth < stealth_threshold:
                    break

        if output_path and current != output_path:
            shutil.copy2(str(current), str(output_path))
            current = output_path
        elif not output_path and work_dir in Path(current).parents:
            fallback = image_path.parent / f"{image_path.stem}.chain{image_path.suffix}"
            shutil.copy2(str(current), str(fallback))
            current = fallback

        if original_size is not None:
            try:
                import io

                from PIL import Image as _Img
                with _Img.open(current) as out_im:
                    if out_im.size != original_size:
                        resized = out_im.resize(original_size, _Img.Resampling.LANCZOS)
                        buf = io.BytesIO()
                        resized.save(buf, format=out_im.format or "PNG", quality=95)
                        safe_write_bytes(current, buf.getvalue())
                        eprint(f"resized {out_im.size} -> {original_size} to match input")
            except Exception as exc:
                eprint(f"resize failed: {exc}")
                flux3_failed = True
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    final_stealth = _run_forensic(current)
    elapsed = time.monotonic() - t0

    return ChainResult(
        scheme=scheme,
        steps_run=steps_run,
        steps_total=len(chain),
        stealth_score=final_stealth,
        elapsed_s=round(elapsed, 2),
        output_path=str(current),
        complete=not flux3_failed and steps_run > 0,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Input image")
    p.add_argument("-o", "--output", type=Path, help="Output path")
    p.add_argument(
        "--scheme",
        choices=sorted(VALID_SCHEMES),
        default=None,
        help="Watermark scheme (default: auto-classify)",
    )
    p.add_argument("--device", type=str, default=None, help="Device for engines")
    p.add_argument("--timeout", type=int, default=600, help="Per-step timeout (s)")
    p.add_argument(
        "--stealth-threshold",
        type=float,
        default=0.7,
        help="Stop chain when stealth score drops below this (default: 0.7)",
    )
    p.add_argument(
        "--flux3-api-key",
        default=None,
        help="OpenRouter API key for FLUX 3 steps (default: $OPENROUTER_API_KEY or ~/.claude/openrouter.env)",
    )
    p.add_argument(
        "--flux3-max-cost",
        type=float,
        default=None,
        help="Per-image cost limit for FLUX 3 in USD (default: clean_flux3.py default)",
    )
    p.add_argument(
        "--flux3-resolution",
        choices=["768", "1k", "1.5k", "2k", "4k"],
        default=None,
        help="Override FLUX 3 resolution tier (default: auto from input size)",
    )
    p.add_argument(
        "--flux3-strength",
        choices=["minimal", "normal", "strong", "maximum"],
        default=None,
        help="Override FLUX 3 prompt strength (default: per-scheme chain definition)",
    )
    p.add_argument(
        "--flux3-timeout",
        type=int,
        default=None,
        help="FLUX 3 stage timeout in seconds (default: same as --timeout)",
    )
    p.add_argument(
        "--flux3-provider",
        choices=["openrouter", "bfl", "fal"],
        default=None,
        help="FLUX 3 provider (default: openrouter)",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    if not args.path.is_file():
        eprint(f"not a file: {args.path}")
        return 2

    scheme = args.scheme
    if scheme is None:
        try:
            clf_result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "classify_watermark.py"),
                    str(args.path),
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
                creationflags=subprocess_creationflags,
            )
            if clf_result.returncode == 0:
                clf_data = json.loads(clf_result.stdout)
                scheme = clf_data.get("scheme", "unknown")
            else:
                scheme = "unknown"
        except Exception:
            scheme = "unknown"

    output = args.output or cleaned_path(args.path, ".chain")

    try:
        result = run_chain(
            args.path,
            scheme,
            output_path=output,
            device=args.device,
            timeout=args.timeout,
            stealth_threshold=args.stealth_threshold,
            api_key=args.flux3_api_key,
            max_cost=args.flux3_max_cost,
            flux3_resolution=args.flux3_resolution,
            flux3_strength=args.flux3_strength,
            flux3_timeout=args.flux3_timeout,
            flux3_provider=args.flux3_provider,
        )
    except ValueError as e:
        eprint(str(e))
        return 2
    except Exception as e:
        eprint(f"chain error: {e}")
        return 1

    if args.json:
        emit_json(
            {
                "scheme": result.scheme,
                "steps_run": result.steps_run,
                "steps_total": result.steps_total,
                "stealth_score": result.stealth_score,
                "elapsed_s": result.elapsed_s,
                "output": result.output_path,
                "complete": result.complete,
            }
        )
    else:
        status = "complete" if result.complete else "INCOMPLETE"
        eprint(
            f"chain({result.scheme}): {result.steps_run}/{result.steps_total} steps, "
            f"stealth={result.stealth_score:.3f}, {result.elapsed_s:.1f}s [{status}]"
        )
    return 0 if result.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
