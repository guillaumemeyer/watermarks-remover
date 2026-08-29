#!/usr/bin/env python3
"""Unified clean: text Layer A, raster metadata, and document containers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from av_meta import clean_av
from common import (
    MAX_INPUT_BYTES,
    ROUTER_ADVICE,
    backup_path,
    cleaned_path,
    eprint,
    guard_binary,
    result_has_changes,
    safe_write_text,
)
from container_meta import clean_container, detect_container_format
from format_dispatch import classify
from image_meta import clean_image
from text_unicode import clean_text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--in-place", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--nfkc", action="store_true", help="Text: NFKC normalize")
    p.add_argument(
        "--deep-images",
        choices=("auto", "always", "lossless", "never"),
        default="auto",
        help=(
            "PDF: Ghostscript re-distill that reaches metadata inside embedded "
            "images. auto (default) only when AI/C2PA markers survive the "
            "document strip; always also clears non-AI EXIF; lossless never "
            "recompresses images; never skips the pass entirely"
        ),
    )
    p.add_argument("--aggressive-homoglyphs", action="store_true")
    p.add_argument(
        "--keep-non-ai-metadata",
        action="store_true",
        help="Images/audio/video: only drop C2PA/AI-looking segments",
    )
    p.add_argument(
        "--as",
        dest="force_type",
        choices=("auto", "text", "image", "container", "av"),
        default="auto",
    )
    p.add_argument(
        "--force-text",
        action="store_true",
        help="Clean as text even when the bytes look like a binary container",
    )
    p.add_argument(
        "-q",
        "--quiet",
        "--only-changed",
        dest="quiet",
        action="store_true",
        help="Only report files that were modified or carry warnings/residuals",
    )
    args = p.parse_args()

    if not args.path.is_file():
        eprint(f"not a file: {args.path}")
        return 2

    if args.path.stat().st_size > MAX_INPUT_BYTES:
        eprint(f"refusing input larger than {MAX_INPUT_BYTES} bytes: {args.path}")
        return 2

    kind = args.force_type if args.force_type != "auto" else classify(args.path)

    # classify() reports "unknown" for bytes that match no supported format.
    # Never mutate those in auto mode: a binary with valid UTF-8 runs would
    # be decoded and written back mangled. --as text / --force-text are the
    # explicit opt-ins and both route through the text pipeline below.
    if kind == "unknown":
        if args.force_type == "text" or args.force_text:
            kind = "text"
        else:
            eprint(f"refusing to classify {args.path}: unrecognized format")
            for line in ROUTER_ADVICE:
                eprint(line)
            return 2

    # In-place cleaning reads from a .bak copy whose suffix would make
    # markdown/HTML (detected by extension, not magic bytes) classify as
    # "unknown". Pin the format from the original path so --in-place and -o
    # route identically.
    container_fmt = None
    if kind == "container":
        container_fmt = detect_container_format(args.path, args.path.read_bytes())

    # classify() falls back to "text" for unrecognised bytes, so an unknown
    # binary would otherwise be decoded, scrubbed and written back mangled.
    # Sniff before --in-place takes a backup: refusing afterwards would leave a
    # .bak sidecar behind for a file this run never touches.
    raw = args.path.read_bytes() if kind == "text" else None
    if raw is not None:
        guard_binary(
            raw,
            str(args.path),
            allow_binary=args.force_text,
            advice=ROUTER_ADVICE,
        )

    if args.in_place:
        bak, created = backup_path(args.path)
        if not created and not args.quiet:
            eprint(f"backup {bak} already exists from an earlier run; keeping the original backup")
        # On a repeated in-place run the backup already holds the original
        # (pre-clean) bytes; process the current file so an already-clean
        # destination is not reported as modified against a stale .bak.
        src = bak if created else args.path
        dest = args.path
    else:
        src = args.path
        dest = args.output or cleaned_path(args.path)

    if kind == "text":
        text = raw.decode("utf-8", errors="surrogateescape")
        cleaned, stats = clean_text(
            text,
            nfkc=args.nfkc,
            aggressive_homoglyphs=args.aggressive_homoglyphs,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        safe_write_text(dest, cleaned)
        result = {
            "kind": "text",
            "input": str(args.path),
            "output": str(dest),
            "stats": stats,
            "changed": bool(stats["removed_count"] or stats["replaced_count"]),
        }
        changed = stats["removed_count"] or stats["replaced_count"]
        should_report = not args.quiet or changed
        if args.json:
            if should_report:
                print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            if should_report:
                eprint(
                    f"wrote {dest} removed={stats['removed_count']} replaced={stats['replaced_count']}"
                )
        return 0

    if kind == "image":
        try:
            result = clean_image(
                src,
                dest,
                strip_all_metadata=not args.keep_non_ai_metadata,
            )
        except Exception as e:
            eprint(f"error: {e}")
            return 1
        is_changed = (
            (src.read_bytes() != dest.read_bytes())
            if (src.is_file() and dest.is_file())
            else result_has_changes(result)
        )
        result = {"kind": "image", "changed": is_changed, **result}
        residual = result["still_has_c2pa"] or result["still_has_ai_metadata"]
        is_changed = result.get("changed", False) or (result["bytes_in"] != result["bytes_out"])
        should_report = not args.quiet or is_changed or residual
        if args.json:
            if should_report:
                print(json.dumps(result, indent=2))
        else:
            if should_report:
                if result["changed"]:
                    eprint(
                        f"wrote {result['output']} ({result['bytes_in']} -> {result['bytes_out']})"
                    )
                    for a in result["actions"]:
                        eprint(f"  - {a}")
                else:
                    eprint(f"already clean: {result['output']}")
            if residual:
                eprint("warning: residual C2PA/AI signals may remain")
        return 1 if residual else 0

    if kind == "av":
        try:
            result = clean_av(
                src,
                dest,
                strip_all_metadata=not args.keep_non_ai_metadata,
            )
        except Exception as e:
            eprint(f"error: {e}")
            return 1
        is_changed = (
            (src.read_bytes() != dest.read_bytes())
            if (src.is_file() and dest.is_file())
            else result_has_changes(result)
        )
        result = {"kind": "av", "changed": is_changed, **result}
        residual = result["still_has_c2pa"] or result["still_has_ai_metadata"]
        is_changed = result.get("changed", False) or (result["bytes_in"] != result["bytes_out"])
        should_report = not args.quiet or is_changed or residual
        if args.json:
            if should_report:
                print(json.dumps(result, indent=2))
        else:
            if should_report:
                if result["changed"]:
                    eprint(
                        f"wrote {result['output']} ({result['bytes_in']} -> {result['bytes_out']})"
                    )
                    for a in result["actions"]:
                        eprint(f"  - {a}")
                else:
                    eprint(f"already clean: {result['output']}")
            if residual:
                eprint("warning: residual C2PA/AI signals may remain")
        return 1 if residual else 0

    try:
        result = clean_container(src, dest, fmt=container_fmt, deep_images=args.deep_images)
    except Exception as e:
        eprint(f"error: {e}")
        return 1
    is_changed = (
        (src.read_bytes() != dest.read_bytes())
        if (src.is_file() and dest.is_file())
        else result_has_changes(result)
    )
    result = {"kind": "container", "changed": is_changed, **result}
    residual = result["still_has_c2pa"] or result["still_has_ai_metadata"]
    degraded = bool(result.get("meta", {}).get("degraded"))
    is_changed = result.get("changed", False) or (result["bytes_in"] != result["bytes_out"])
    should_report = not args.quiet or is_changed or residual or degraded
    if args.json:
        if should_report:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if should_report:
            if result["changed"]:
                eprint(f"wrote {result['output']} format={result['format']}")
                for a in result["actions"]:
                    eprint(f"  - {a}")
            else:
                eprint(f"already clean: {result['output']}")
        if residual:
            eprint("warning: residual C2PA/AI signals may remain")
            for f in result.get("post_findings") or []:
                eprint(f"  ! {f}")
    # A degraded (best-effort) PDF copy warns but is not a hard failure.
    return 1 if (residual and not degraded) else 0


if __name__ == "__main__":
    raise SystemExit(main())
