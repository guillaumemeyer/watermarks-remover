"""Every .PHONY target must exist and expand to a syntactically valid recipe.

A target whose recipe swallows the next target declaration (a missing `fi`,
`done`, or a stray trailing backslash) fails in two silent ways: the swallowed
target reports "Nothing to be done" instead of running, and the host target
dies with a shell syntax error. Both went unnoticed on main because no test
ever asked `make` what a target expands to.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
MAKE = shutil.which("make")
SH = shutil.which("sh")

# Targets whose recipes are intentionally empty or have no portable dry run.
_EXPECTED_NO_RECIPE: frozenset[str] = frozenset()

pytestmark = pytest.mark.skipif(
    os.name == "nt" or MAKE is None or SH is None,
    reason="requires POSIX make and sh",
)


def _phony_targets() -> list[str]:
    text = MAKEFILE.read_text(encoding="utf-8")
    match = re.search(r"^\.PHONY:(.*?)(?<!\\)\n(?!\s)", text, re.DOTALL | re.MULTILINE)
    assert match, "Makefile declares no .PHONY block"
    names = match.group(1).replace("\\\n", " ").split()
    assert names, ".PHONY block is empty"
    return names


def _declared_targets() -> set[str]:
    text = MAKEFILE.read_text(encoding="utf-8")
    return set(re.findall(r"^([A-Za-z0-9_.\-]+):(?!=)", text, re.MULTILINE))


def _dry_run(target: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [MAKE, "--dry-run", "--no-print-directory", target],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize("target", _phony_targets())
def test_phony_target_is_declared(target: str) -> None:
    """A .PHONY name with no rule is a target that silently does nothing."""
    assert target in _declared_targets(), (
        f"{target} is listed in .PHONY but has no rule in the Makefile; "
        "`make " + target + "` would report 'Nothing to be done'"
    )


@pytest.mark.parametrize("target", _phony_targets())
def test_phony_target_expands_to_a_recipe(target: str) -> None:
    """`make --dry-run` must print commands, not 'Nothing to be done'."""
    if target in _EXPECTED_NO_RECIPE:
        pytest.skip(f"{target} intentionally has no recipe")
    result = _dry_run(target)
    assert result.returncode == 0, (
        f"make --dry-run {target} failed ({result.returncode}):\n{result.stderr}"
    )
    assert "Nothing to be done" not in result.stdout + result.stderr, (
        f"{target} resolved to no recipe:\n{result.stdout}{result.stderr}"
    )
    assert result.stdout.strip(), f"{target} expanded to an empty recipe"


@pytest.mark.parametrize("target", _phony_targets())
def test_phony_target_recipe_is_valid_shell(target: str) -> None:
    """Each expanded recipe line must parse under `sh -n`.

    This is what catches a recipe that ate the next target declaration: the
    unterminated `if` only surfaces when a shell parses the whole line.
    """
    if target in _EXPECTED_NO_RECIPE:
        pytest.skip(f"{target} intentionally has no recipe")
    result = _dry_run(target)
    assert result.returncode == 0, f"make --dry-run {target} failed:\n{result.stderr}"
    check = subprocess.run(
        [SH, "-n"],
        input=result.stdout,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert check.returncode == 0, (
        f"recipe for {target} is not valid shell ({check.stderr.strip()}):\n{result.stdout}"
    )


def test_no_recipe_line_swallows_the_next_target() -> None:
    """A target declaration must never sit inside a continued recipe line."""
    offenders: list[tuple[int, str]] = []
    continued = False
    for number, line in enumerate(MAKEFILE.read_text(encoding="utf-8").splitlines(), 1):
        if continued and re.match(r"^[A-Za-z0-9_.\-]+:(?!=)", line):
            offenders.append((number, line))
        continued = line.endswith("\\")
    assert not offenders, "target declarations swallowed by a continued recipe line: " + ", ".join(
        f"line {n}: {t!r}" for n, t in offenders
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__]))
