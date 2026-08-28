"""The repository is also a Claude Code plugin and a single-plugin marketplace."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_MANIFEST = ROOT / ".claude-plugin" / "plugin.json"
HOOKS_MANIFEST = ROOT / "hooks" / "hooks.json"
MARKETPLACE_MANIFEST = ROOT / ".claude-plugin" / "marketplace.json"

KEBAB_CASE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
SEMVER = re.compile(r"^\d+\.\d+\.\d+")

# Names Anthropic reserves for official marketplaces; a third-party manifest
# using one stops loading entirely.
RESERVED_MARKETPLACE_NAMES = {
    "agent-skills",
    "anthropic-agent-skills",
    "anthropic-marketplace",
    "anthropic-plugins",
    "claude-code-marketplace",
    "claude-code-plugins",
    "claude-community",
    "claude-for-financial-services",
    "claude-for-legal",
    "claude-plugins-community",
    "claude-plugins-official",
    "financial-services-plugins",
    "first-party-plugins",
    "healthcare",
    "knowledge-work-plugins",
    "life-sciences",
}


@pytest.fixture(scope="module")
def plugin() -> dict:
    return json.loads(PLUGIN_MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def marketplace() -> dict:
    return json.loads(MARKETPLACE_MANIFEST.read_text(encoding="utf-8"))


def test_plugin_manifest_has_a_kebab_case_name(plugin):
    assert KEBAB_CASE.match(plugin["name"])
    assert plugin["description"]
    assert SEMVER.match(plugin["version"])


def test_marketplace_manifest_has_the_required_top_level_fields(marketplace):
    assert KEBAB_CASE.match(marketplace["name"])
    assert marketplace["name"] not in RESERVED_MARKETPLACE_NAMES
    assert marketplace["owner"]["name"]
    assert marketplace["plugins"]


def test_every_marketplace_entry_names_a_source_that_exists(marketplace):
    for entry in marketplace["plugins"]:
        assert KEBAB_CASE.match(entry["name"])
        source = entry["source"]
        assert isinstance(source, str), "remote sources need their own coverage"
        assert not source.startswith("../"), "sources must stay inside the marketplace root"
        resolved = (ROOT / source).resolve()
        assert resolved.is_dir()
        assert resolved == ROOT or ROOT in resolved.parents


def test_marketplace_entry_agrees_with_the_plugin_manifest(plugin, marketplace):
    # Both files pin a version; if they disagree, the plugin.json value wins at
    # install time and the marketplace listing silently shows the wrong one.
    entry = next(item for item in marketplace["plugins"] if item["name"] == plugin["name"])

    assert entry["version"] == plugin["version"]
    assert entry["license"] == plugin["license"]
    assert entry["repository"] == plugin["repository"]


def test_plugin_ships_every_skill_in_the_repository(plugin, marketplace):
    # The plugin has no "skills" field, so Claude Code scans the default
    # skills/ directory at the plugin root; both shipped skills must be there.
    entry = next(item for item in marketplace["plugins"] if item["name"] == plugin["name"])
    plugin_root = (ROOT / entry["source"]).resolve()
    assert "skills" not in plugin and "skills" not in entry

    shipped = {path.parent.name for path in (plugin_root / "skills").glob("*/SKILL.md")}

    assert shipped == {"remove-ai-marks", "clean-user-facing-text"}


def test_plugin_name_does_not_collide_with_a_bundled_skill_name(plugin):
    # Plugin skills are invoked as /<plugin>:<skill>; a plugin named after one
    # of its own skills produces /remove-ai-marks:remove-ai-marks.
    skills = {path.parent.name for path in (ROOT / "skills").glob("*/SKILL.md")}

    assert plugin["name"] not in skills


# --------------------------------------------------------------------------
# hooks
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hooks() -> dict:
    return json.loads(HOOKS_MANIFEST.read_text(encoding="utf-8"))


def test_post_tool_use_hook_covers_the_file_writing_tools(hooks):
    matchers = [entry["matcher"] for entry in hooks["hooks"]["PostToolUse"]]

    assert matchers, "no PostToolUse hook registered"
    covered = "|".join(matchers).split("|")
    assert {"Write", "Edit"} <= set(covered)


def test_hook_command_points_at_a_script_that_exists(hooks):
    # ${CLAUDE_PLUGIN_ROOT} is the plugin root, which for this repo-as-plugin
    # is the repository root.
    for entry in hooks["hooks"]["PostToolUse"]:
        for hook in entry["hooks"]:
            assert hook["type"] == "command"
            script = next(arg for arg in hook["args"] if arg.startswith("${CLAUDE_PLUGIN_ROOT}/"))
            resolved = ROOT / script.removeprefix("${CLAUDE_PLUGIN_ROOT}/")
            assert resolved.is_file(), resolved


def test_hook_mode_is_configurable_and_defaults_to_the_reporting_mode(plugin):
    sys.path.insert(0, str(ROOT / "service" / "scripts"))
    import hook_written_file

    option = plugin["userConfig"]["hook_mode"]
    assert option["default"] == hook_written_file.DEFAULT_MODE == "check"
    assert set(hook_written_file.MODES) == {"check", "clean"}


def test_hook_command_never_substitutes_an_unset_plugin_option(hooks):
    # Regression: Claude Code 2.1.235 refuses to run a hook whose command
    # references ${user_config.KEY} before the user has set that option in
    # /plugin manage -- a declared `default` does not satisfy it, and the hook
    # fails with "Plugin option ... isn't set" instead of running. Substituting
    # the mode there meant the hook silently never ran on a fresh install; it
    # reads CLAUDE_PLUGIN_OPTION_HOOK_MODE from the environment instead.
    for entry in hooks["hooks"]["PostToolUse"]:
        for hook in entry["hooks"]:
            rendered = json.dumps([hook.get("command"), *hook.get("args", [])])
            assert "user_config" not in rendered


def test_hook_command_is_cross_platform(hooks):
    # Windows does not ship `python3.exe` by default. Using `node` to launch the
    # platform-aware launcher (run_hook.js) guarantees cross-platform execution
    # in Claude Code (#239).
    for entry in hooks["hooks"]["PostToolUse"]:
        for hook in entry["hooks"]:
            assert hook["command"] == "node"
