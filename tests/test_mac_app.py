"""The macOS app copies `service/scripts` into the bundle; keep that list honest.

``mac/`` does not vendor a fork of the cleaners. ``ScriptBundle.requiredScripts``
and ``mac/Scripts/build_app.sh`` must match the import closure of
``inspect_file.py``, ``clean_file.py``, and ``rewrite_text.py``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAC = ROOT / "mac"
SCRIPTS = ROOT / "service" / "scripts"
SCRIPT_BUNDLE = MAC / "Sources" / "WatermarksMac" / "ScriptBundle.swift"
BUILD_SCRIPT = MAC / "Scripts" / "build_app.sh"
REWRITE_CALLER = MAC / "Sources" / "WatermarksMac" / "Reports.swift"

ENTRIES = ("inspect_file.py", "clean_file.py", "rewrite_text.py")

pytestmark = pytest.mark.skipif(not MAC.is_dir(), reason="mac/ app is not present")


def import_closure(entry: str) -> set[str]:
    local = {path.stem for path in SCRIPTS.glob("*.py")}
    seen: set[str] = set()

    def walk(module: str) -> None:
        if module in seen:
            return
        seen.add(module)
        source = SCRIPTS / f"{module}.py"
        if not source.exists():
            return
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in local:
                walk(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in local:
                        walk(alias.name)

    walk(Path(entry).stem)
    return {f"{name}.py" for name in seen}


def needed_scripts() -> set[str]:
    out: set[str] = set()
    for entry in ENTRIES:
        out |= import_closure(entry)
    return out


def swift_required_scripts() -> list[str]:
    source = SCRIPT_BUNDLE.read_text(encoding="utf-8")
    match = re.search(r"requiredScripts\s*=\s*\[(.*?)\]", source, re.DOTALL)
    assert match, "ScriptBundle.swift no longer declares requiredScripts"
    return re.findall(r'"([^"]+\.py)"', match.group(1))


def build_script_scripts() -> list[str]:
    source = BUILD_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"for script in \\\s*(.*?)\ndo", source, re.DOTALL)
    if match is None:
        match = re.search(r"for script in(.*?); do", source, re.DOTALL)
    assert match, "build_app.sh no longer copies a literal list of scripts"
    return re.findall(r"([\w.]+\.py)", match.group(1))


def test_swift_list_matches_the_import_closure() -> None:
    assert set(swift_required_scripts()) == needed_scripts()


def test_build_script_copies_the_same_files() -> None:
    assert set(build_script_scripts()) == set(swift_required_scripts())


def test_every_required_script_exists() -> None:
    for name in swift_required_scripts():
        assert (SCRIPTS / name).is_file(), f"service/scripts/{name} is missing"


def test_entry_points_are_inspect_clean_rewrite() -> None:
    source = SCRIPT_BUNDLE.read_text(encoding="utf-8")
    assert 'inspectEntry = "inspect_file.py"' in source
    assert 'cleanEntry = "clean_file.py"' in source
    assert 'rewriteEntry = "rewrite_text.py"' in source


def test_bundled_scripts_are_standard_library_only() -> None:
    allowed = {name.removesuffix(".py") for name in swift_required_scripts()}
    stdlib_modules = set(getattr(__import__("sys"), "stdlib_module_names", ()))
    for name in swift_required_scripts():
        tree = ast.parse((SCRIPTS / name).read_text(encoding="utf-8"))
        for node in tree.body:
            roots: list[str] = []
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = [node.module.split(".")[0]]
            for root in roots:
                assert root in allowed or root in stdlib_modules, (
                    f"{name} imports {root} at module level, which the system "
                    "python3 the app runs does not have"
                )


def test_app_sets_reasoning_effort_explicitly() -> None:
    source = REWRITE_CALLER.read_text(encoding="utf-8")
    assert '"--reasoning-effort"' in source
    assert '"off"' in source
