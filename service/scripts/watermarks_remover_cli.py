#!/usr/bin/env python3
"""Friendly command-line entry point for clean_file.py."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def _expand_path(value: str) -> str:
    return str(Path(value).expanduser())


def main() -> int:
    args = sys.argv[1:]
    if not args:
        try:
            source = input("File to clean: ").strip()
            if not source:
                print("No input file provided.", file=sys.stderr)
                return 2
            output = input("Output file (leave blank for the default): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.", file=sys.stderr)
            return 2
        args = [_expand_path(source)]
        if output:
            args.extend(("--output", _expand_path(output)))
    else:
        # Expand user-home markers in path arguments while leaving every other
        # option untouched. argparse accepts the input path among options.
        expanded: list[str] = []
        path_next = False
        paths_only = False
        for arg in args:
            if paths_only:
                expanded.append(_expand_path(arg))
                continue
            elif arg == "--":
                expanded.append(arg)
                paths_only = True
                continue
            if path_next:
                expanded.append(_expand_path(arg))
                path_next = False
            elif arg in ("-o", "--output"):
                expanded.append(arg)
                path_next = True
            elif arg.startswith("--output="):
                expanded.append("--output=" + _expand_path(arg.partition("=")[2]))
            elif arg.startswith("-o") and len(arg) > 2:
                expanded.append("-o" + _expand_path(arg[2:]))
            elif not arg.startswith("-"):
                expanded.append(_expand_path(arg))
            else:
                expanded.append(arg)
        args = expanded

    sys.argv = ["watermarks-remover", *args]
    import clean_file

    return clean_file.main()


if __name__ == "__main__":
    raise SystemExit(main())
