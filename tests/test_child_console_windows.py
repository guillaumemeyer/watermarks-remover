"""Child processes must not open a console window.

A parent without a console of its own -- pythonw, a service, anything
GUI-hosted -- makes Windows allocate a fresh console for every child it starts.
The window blinks and takes keyboard focus, and a single cleaned PDF spawns
exiftool, qpdf, ghostscript and c2patool in turn, so the flicker arrives in
bursts. `subprocess_creationflags` carries CREATE_NO_WINDOW on Windows and 0
everywhere else.

The static test is the one that matters over time: a spawn site added later
without the flag reintroduces the flicker in exactly one place, which is hard
to notice and easy to miss in review.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from common import subprocess_creationflags

SPAWNERS = {"run", "Popen", "check_output", "call"}


def _spawn_sites(source: str) -> list[ast.Call]:
    sites = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if isinstance(owner, ast.Name) and owner.id == "subprocess" and node.func.attr in SPAWNERS:
            sites.append(node)
    return sites


def _uses_the_shared_flag(node: ast.expr) -> bool:
    """True only for a bare `subprocess_creationflags`.

    Combinations are deliberately refused rather than merged in. On Windows
    CREATE_NEW_CONSOLE overrides CREATE_NO_WINDOW, so a `|` chain can carry the
    shared flag and still open a window; and constants like
    CREATE_NEW_PROCESS_GROUP do not exist off Windows, so a chain that reads
    fine here raises AttributeError there. A site that genuinely needs more
    flags is a decision worth making in the open, which means updating this
    guard rather than slipping past it.
    """
    return isinstance(node, ast.Name) and node.id == "subprocess_creationflags"


def test_every_spawn_site_suppresses_the_console():
    missing = []
    counted = 0
    for path in sorted(SCRIPTS.glob("*.py")):
        for call in _spawn_sites(path.read_text(encoding="utf-8")):
            counted += 1
            flag = next((kw for kw in call.keywords if kw.arg == "creationflags"), None)
            if flag is None or not _uses_the_shared_flag(flag.value):
                missing.append(f"{path.name}:{call.lineno}")

    assert counted, "no subprocess spawn sites found -- has the layout changed?"
    assert not missing, "spawn sites not passing subprocess_creationflags: " + ", ".join(missing)


def _flag_expression(expression: str) -> ast.expr:
    source = f"import subprocess\nsubprocess.run(['x'], creationflags={expression})\n"
    call = _spawn_sites(source)[0]
    return next(kw for kw in call.keywords if kw.arg == "creationflags").value


@pytest.mark.parametrize(
    "expression",
    [
        "0",
        "subprocess.CREATE_NEW_CONSOLE",
        "subprocess_creationflags + 1",
        "wrapper(subprocess_creationflags)",
        "subprocess_creationflags if nt else 0",
        # CREATE_NEW_CONSOLE wins over CREATE_NO_WINDOW: the flag is present and
        # the window opens anyway.
        "subprocess_creationflags | subprocess.CREATE_NEW_CONSOLE",
        # Fine on Windows, AttributeError on POSIX.
        "subprocess_creationflags | subprocess.CREATE_NEW_PROCESS_GROUP",
    ],
)
def test_the_static_check_rejects_what_is_not_the_shared_flag(expression):
    """Naming the flag is not the same as passing it."""
    assert _uses_the_shared_flag(_flag_expression(expression)) is False


def test_the_static_check_accepts_the_shared_flag():
    """The one accepted form, which is what every spawn site actually writes."""
    assert _uses_the_shared_flag(_flag_expression("subprocess_creationflags")) is True


def test_flag_value_matches_the_platform():
    if os.name == "nt":
        assert subprocess_creationflags == subprocess.CREATE_NO_WINDOW
    else:
        # subprocess treats 0 as "no flags", so POSIX callers pass it harmlessly.
        assert subprocess_creationflags == 0


@pytest.mark.skipif(os.name != "nt", reason="console windows are a Windows concept")
def test_a_console_less_parent_gives_its_child_no_window(tmp_path):
    """The behaviour itself: ask the child whether it owns a console window."""
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.is_file():
        pytest.skip("pythonw.exe not next to the interpreter")

    child = tmp_path / "child.py"
    child.write_text(
        textwrap.dedent(
            """
            import ctypes, sys, pathlib
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            pathlib.Path(sys.argv[1]).write_text(str(hwnd), encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )

    parent = tmp_path / "parent.py"
    parent.write_text(
        textwrap.dedent(
            f"""
            import subprocess, sys
            child, plain, hidden = sys.argv[1], sys.argv[2], sys.argv[3]
            subprocess.run([r"{sys.executable}", child, plain], capture_output=True)
            subprocess.run(
                [r"{sys.executable}", child, hidden],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            """
        ),
        encoding="utf-8",
    )

    plain, hidden = tmp_path / "plain.txt", tmp_path / "hidden.txt"
    subprocess.run(
        [str(pythonw), str(parent), str(child), str(plain), str(hidden)],
        check=True,
        timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    assert plain.read_text(encoding="utf-8") != "0", (
        "expected a console-less parent to hand its child a window without the flag; "
        "if this fails the test no longer proves anything"
    )
    assert hidden.read_text(encoding="utf-8") == "0"
