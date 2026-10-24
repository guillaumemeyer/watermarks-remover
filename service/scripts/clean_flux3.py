#!/usr/bin/env python3
"""Regenerate an image through FLUX 3 (OpenRouter) to drop latent watermarks, then strip metadata.

FLUX 3 image-to-image redraws every pixel, so a latent watermark in the input does not survive.
The regenerated image can carry provenance of its own (a C2PA manifest, BFL markers), so the
output always goes through image_meta.clean_image before it is written, and is scored by the
optional reverse-SynthID scorer (--synthid-dir, $REVERSE_SYNTHID_DIR or the scorer sidecar) to
catch a watermark FLUX 3 injected itself.

FLUX3_TEST_MOCK=1 skips the paid API call for tests: the "generated" image is
FLUX3_TEST_MOCK_BYTES (base64) or a copy of the input, and cost_usd is 0.0.

Exit codes:
  0  regenerated and written without residual metadata
  1  runtime error (request failed, unusable response, write failed, metadata left on output)
  2  bad input (missing/unsupported image, bad args, estimated cost over --max-cost)
  3  backend unavailable (no API key, provider not implemented, httpx missing)
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import cleaned_path, emit_json, eprint  # noqa: E402
from flux3_client import (  # noqa: E402
    PROMPT_STRATEGIES,
    Flux3Client,
    Flux3CostGuard,
    Flux3Error,
    Flux3Result,
    Flux3Unavailable,
    estimate_cost,
)
from image_meta import clean_image, detect_format  # noqa: E402

# Restricted to the priced tiers: an unknown tier is estimated at the 1k price, which would let
# an expensive request slip under --max-cost.
RESOLUTIONS = ("768", "1k", "1.5k", "2k", "4k")
PROVIDERS = ("openrouter", "bfl", "fal")
INPUT_FORMATS = ("png", "jpeg", "webp")
# Reverse-SynthID confidence above which FLUX 3 is taken to have watermarked its own output.
SYNTHID_THRESHOLD = 0.7


def _default_output(src: Path, src_fmt: str, out_fmt: str) -> Path:
    """path/x.ext -> path/x.flux3.ext, switching the extension when FLUX 3 changed the format."""
    out = cleaned_path(src, ".flux3")
    if out_fmt != src_fmt:
        out = out.with_suffix(".jpg" if out_fmt == "jpeg" else f".{out_fmt}")
    return out


def _synthid_status(entry: dict[str, Any] | None) -> tuple[str, bool | None]:
    """(synthid_check, flux3_watermark_detected); None means the output was not scored."""
    if entry is None:
        return "skipped", None
    confidence = entry.get("confidence")
    if (
        not entry.get("available")
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
    ):
        return "unavailable", None
    return "scored", confidence > SYNTHID_THRESHOLD


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Input image (PNG, JPEG or WebP)")
    p.add_argument("-o", "--output", type=Path, help="Output path (default: *.flux3.*)")
    p.add_argument(
        "--api-key", default=None, help="OpenRouter API key (default: $OPENROUTER_API_KEY)"
    )
    p.add_argument(
        "--prompt", default=None, help="Custom regeneration prompt; overrides --strength"
    )
    p.add_argument(
        "--resolution", choices=RESOLUTIONS, default="1k", help="Output resolution (default: 1k)"
    )
    p.add_argument(
        "--strength",
        choices=tuple(PROMPT_STRATEGIES),
        default="normal",
        help="Built-in prompt strategy, from minimal to maximum redraw (default: normal)",
    )
    p.add_argument(
        "--max-cost",
        type=float,
        default=0.50,
        help="Refuse when the estimated cost in USD exceeds this (default: 0.50)",
    )
    p.add_argument(
        "--provider",
        choices=PROVIDERS,
        default="openrouter",
        help="FLUX 3 provider; only openrouter is implemented so far (default: openrouter)",
    )
    p.add_argument(
        "--synthid-dir",
        default=None,
        help="reverse-SynthID checkout used to score the output (default: $REVERSE_SYNTHID_DIR)",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args(argv)

    if not args.path.is_file():
        eprint(f"not a file: {args.path}")
        return 2
    try:
        data = args.path.read_bytes()
    except OSError as e:
        eprint(f"cannot read {args.path}: {e}")
        return 2
    in_fmt = detect_format(data)
    if in_fmt not in INPUT_FORMATS:
        eprint(f"FLUX 3 input must be a PNG, JPEG or WebP image, not {in_fmt}: {args.path}")
        return 2

    estimated = estimate_cost(args.resolution)
    if not Flux3CostGuard(max_per_image=args.max_cost).check(args.resolution):
        eprint(
            f"estimated cost ${estimated:.2f} at {args.resolution} exceeds "
            f"--max-cost ${args.max_cost:.2f}"
        )
        return 2

    api_key = (args.api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
    if not api_key:
        eprint("FLUX 3 not configured: pass --api-key or set OPENROUTER_API_KEY")
        return 3

    prompt = args.prompt or PROMPT_STRATEGIES[args.strength]
    mock = os.environ.get("FLUX3_TEST_MOCK") == "1"
    try:
        client = Flux3Client(api_key, provider=args.provider)
        if mock:
            eprint("flux3: FLUX3_TEST_MOCK=1, skipping the API call")
            mock_b64 = os.environ.get("FLUX3_TEST_MOCK_BYTES")
            image = base64.b64decode(mock_b64, validate=True) if mock_b64 else data
            result = Flux3Result(image, 0.0, "mock")
        else:
            result = client.generate(data, prompt, resolution=args.resolution)
    except Flux3Unavailable as e:
        eprint(f"FLUX 3 unavailable: {e}")
        return 3
    except Flux3Error as e:
        eprint(f"FLUX 3 error: {e}")
        return 1
    except ValueError as e:
        eprint(f"bad input: {e}")
        return 2

    out_fmt = detect_format(result.image_bytes)
    output = args.output or _default_output(args.path, in_fmt, out_fmt)
    restrip: dict[str, Any] | None = None
    try:
        # The raw response still carries its provenance, so it only ever lands in a private
        # temp dir; clean_image writes the stripped bytes to the output atomically.
        with tempfile.TemporaryDirectory(prefix="flux3-") as tmp:
            raw = Path(tmp) / f"raw.{out_fmt}"
            raw.write_bytes(result.image_bytes)
            cleaned = clean_image(raw, output, synthid_dir=args.synthid_dir)
        synthid_check, detected = _synthid_status(cleaned["synthid_after"])
        if detected:
            eprint(
                f"warning: reverse-SynthID confidence {cleaned['synthid_after']['confidence']:.3f}"
                f" > {SYNTHID_THRESHOLD} on the FLUX 3 output; re-stripping metadata"
            )
            restrip = clean_image(output, output, synthid_dir=args.synthid_dir)
    except (OSError, ValueError) as e:
        eprint(f"cannot clean or write the FLUX 3 output: {e}")
        return 1

    final = restrip or cleaned
    report: dict[str, Any] = {
        "input": str(args.path),
        "output": str(output),
        "provider": args.provider,
        "mock": mock,
        "resolution": args.resolution,
        "strength": None if args.prompt else args.strength,
        "prompt": prompt,
        "estimated_cost_usd": estimated,
        "cost_usd": result.cost_usd,
        "max_cost_usd": args.max_cost,
        "format": cleaned["format"],
        "bytes_in": len(data),
        "bytes_out": final["bytes_out"],
        "actions": cleaned["actions"] + (restrip["actions"] if restrip else []),
        "still_has_c2pa": final["still_has_c2pa"],
        "still_has_ai_metadata": final["still_has_ai_metadata"],
        "post_findings": final["post_findings"],
        "synthid_check": synthid_check,
        "synthid": cleaned["synthid_after"],
        "flux3_watermark_detected": detected,
    }
    if restrip is not None:
        report["restrip_changed"] = restrip["changed"]

    if args.json:
        emit_json(report)
    else:
        print(
            f"FLUX 3 regenerated: {args.path} -> {output} "
            f"(cost ${result.cost_usd:.2f}, SynthID check {synthid_check})"
        )

    residual = final["still_has_c2pa"] or final["still_has_ai_metadata"]
    if residual:
        eprint(f"metadata still present on {output}: {final['post_findings']}")
    return 1 if residual else 0


if __name__ == "__main__":
    raise SystemExit(main())
