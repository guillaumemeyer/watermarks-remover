"""The macOS app ships the Layer B scripts, so its file list must stay honest.

``mac/`` bundles ``service/scripts`` inside Watermarker.app and runs them
verbatim. Three places name that set of files -- the Swift ``ScriptBundle``,
the bundle build script, and the updater's download list -- and all three have
to agree with what ``rewrite_text.py`` actually imports. A new import upstream
would otherwise ship an app that crashes on the first run with an unhelpful
ImportError.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAC = ROOT / "mac"
SCRIPTS = ROOT / "service" / "scripts"
SCRIPT_BUNDLE = MAC / "Sources" / "Watermarker" / "Services" / "ScriptBundle.swift"
REWRITE_SERVICE = MAC / "Sources" / "Watermarker" / "Services" / "RewriteService.swift"
BUILD_SCRIPT = MAC / "Scripts" / "build_app.sh"

ENTRY_POINT = "rewrite_text.py"

pytestmark = pytest.mark.skipif(not MAC.is_dir(), reason="mac/ app is not present")


def import_closure(entry: str) -> set[str]:
    """Every local module ``entry`` reaches through plain imports."""
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


def swift_required_scripts() -> list[str]:
    """The ``requiredScripts`` array from ScriptBundle.swift."""
    source = SCRIPT_BUNDLE.read_text(encoding="utf-8")
    match = re.search(r"requiredScripts\s*=\s*\[(.*?)\]", source, re.DOTALL)
    assert match, "ScriptBundle.swift no longer declares requiredScripts"
    return re.findall(r'"([^"]+\.py)"', match.group(1))


def build_script_scripts() -> list[str]:
    """The filenames build_app.sh copies into Resources/PythonScripts."""
    source = BUILD_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"for script in (.*?); do", source, re.DOTALL)
    assert match, "build_app.sh no longer copies a literal list of scripts"
    return re.findall(r"([\w.]+\.py)", match.group(1))


def test_swift_list_matches_the_import_closure() -> None:
    assert set(swift_required_scripts()) == import_closure(ENTRY_POINT)


def test_build_script_copies_the_same_files() -> None:
    assert set(build_script_scripts()) == set(swift_required_scripts())


def test_every_required_script_exists() -> None:
    for name in swift_required_scripts():
        assert (SCRIPTS / name).is_file(), f"service/scripts/{name} is missing"


def test_entry_point_is_the_rewrite_script() -> None:
    source = SCRIPT_BUNDLE.read_text(encoding="utf-8")
    assert f'entryPoint = "{ENTRY_POINT}"' in source


def test_bundled_scripts_are_standard_library_only() -> None:
    """The app runs the system python3, which has no third-party packages.

    Only the ``mlm`` tactic reaches for transformers, and it does so behind a
    lazy import so every other strategy works on a stock Mac. A new top-level
    third-party import would break the app for everyone.
    """
    allowed = {name.removesuffix(".py") for name in swift_required_scripts()}
    stdlib_modules = set(getattr(__import__("sys"), "stdlib_module_names", ()))
    for name in swift_required_scripts():
        tree = ast.parse((SCRIPTS / name).read_text(encoding="utf-8"))
        for node in tree.body:  # top level only; lazy imports live in functions
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
    """The app must not inherit the script's own --reasoning-effort default.

    ``rewrite_text.py`` defaults to ``none``, which OpenAI accepts and most
    other vendors reject with a 400 -- so leaving the flag off made the app
    work only with OpenAI models on OpenRouter. The flag is also read from
    ``WATERMARKS_REWRITE_REASONING_EFFORT``, which the app inherits from the
    user's shell, so passing it explicitly is the only way to be deterministic.
    """
    source = REWRITE_SERVICE.read_text(encoding="utf-8")
    assert '"--reasoning-effort"' in source


def test_app_defaults_reasoning_effort_to_off() -> None:
    """``off`` is the only value that omits the parameter for every model."""
    source = (MAC / "Sources" / "Watermarker" / "Models" / "SettingsStore.swift").read_text(
        encoding="utf-8"
    )
    assert 'Key.reasoningEffort, "off"' in source


def test_reasoning_effort_choices_match_the_script() -> None:
    """The picker cannot offer a value argparse would reject."""
    swift = (MAC / "Sources" / "Watermarker" / "Models" / "SettingsStore.swift").read_text(
        encoding="utf-8"
    )
    block = re.search(r"enum ReasoningEffort.*?\n    \}", swift, re.DOTALL)
    assert block, "SettingsStore.swift no longer declares ReasoningEffort"
    offered = set(re.findall(r'case \w+ = "([^"]+)"', block.group(0)))

    script = (SCRIPTS / "rewrite_text.py").read_text(encoding="utf-8")
    accepted = re.search(r'"--reasoning-effort",\s*choices=\((.*?)\),', script, re.DOTALL)
    assert accepted, "rewrite_text.py no longer declares --reasoning-effort choices"
    allowed = set(re.findall(r'"([^"]+)"', accepted.group(1)))
    assert offered <= allowed, f"picker offers values the script rejects: {offered - allowed}"
