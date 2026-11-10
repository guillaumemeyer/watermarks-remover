#!/usr/bin/env python3
"""Install the watermarks-remover command in ~/.local/bin."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    launcher_dir = Path(
        os.environ.get("WATERMARKS_REMOVER_BIN", Path.home() / ".local/bin")
    ).expanduser()
    launcher_dir.mkdir(parents=True, exist_ok=True)
    launcher_dir = launcher_dir.resolve()
    launcher = launcher_dir / "watermarks-remover"
    wrapper = root / "service/scripts/watermarks_remover_cli.py"
    content = (
        "#!/bin/sh\nexec "
        + shlex.quote(sys.executable)
        + " "
        + shlex.quote(str(wrapper))
        + ' "$@"\n'
    )
    launcher.write_text(content, encoding="utf-8")
    launcher.chmod(0o755)
    print(f"Installed {launcher}")
    if str(launcher_dir) not in os.environ.get("PATH", "").split(os.pathsep):
        print(f"Add {launcher_dir} to PATH to run watermarks-remover from any directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
