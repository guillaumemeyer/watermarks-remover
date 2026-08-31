#!/usr/bin/env python3
"""Strip invisible Unicode / normalize space homoglyphs (Layer A)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    backup_path,
    cleaned_path,
    eprint,
    read_text_input,
    write_text_output,
)
from text_unicode import clean_text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="?", default="-", help="Input text file, or - for stdin")
    p.add_argument("-o", "--output", help="Output path (default: stdout or *.cleaned.*)")
    p.add_argument("--nfkc", action="store_true", help="Apply Unicode NFKC after scrub")
    p.add_argument(
        "--aggressive-homoglyphs",
        action="store_true",
        help="Map Cyrillic/fullwidth Latin confusables to ASCII Latin",
    )
    p.add_argument(
        "--no-normalize-spaces",
        action="store_true",
        help="Do not rewrite exotic spaces to U+0020",
    )
    p.add_argument(
        "--strip-emoji-glue",
        action="store_true",
        help="Paranoid: strip all load-bearing invisibles too (emoji glue, script joiners, flag tags, same-script fillers/selectors, orthographic Cf)",
    )
    p.add_argument(
        "--strip-bidi",
        action="store_true",
        help="Also strip legitimate RTL/LTR directional marks and isolates",
    )
    p.add_argument("--stats", action="store_true", help="Print stats JSON to stderr")
    p.add_argument(
        "--force-text",
        action="store_true",
        help="Clean even when the input looks like a binary container "
        "(this rewrites the bytes and will corrupt the file)",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file (creates .bak backup)",
    )
    args = p.parse_args()

    text = read_text_input(args.path, allow_binary=args.force_text)
    cleaned, stats = clean_text(
        text,
        nfkc=args.nfkc,
        aggressive_homoglyphs=args.aggressive_homoglyphs,
        normalize_spaces=not args.no_normalize_spaces,
        strip_emoji_glue=args.strip_emoji_glue,
        strip_bidi=args.strip_bidi,
    )

    out = args.output
    if args.in_place:
        if args.path in (None, "-"):
            eprint("--in-place requires a file path")
            return 2
        src = Path(args.path)
        bak, created = backup_path(src)
        if not created:
            eprint(f"backup {bak} already exists from an earlier run; keeping the original backup")
        out = str(src)
    elif out is None and args.path not in (None, "-"):
        out = str(cleaned_path(Path(args.path)))

    write_text_output(cleaned, out)

    if args.stats:
        eprint(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        eprint(
            f"removed={stats['removed_count']} replaced={stats['replaced_count']} "
            f"len {stats['input_length']}->{stats['output_length']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
