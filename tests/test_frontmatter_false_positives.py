"""Frontmatter scanning must not mistake configuration for provenance.

The regression these cover: a Claude Code subagent definition carries

    tools: Read, Write, mcp__claude_ai_acme__mail__get-message

and the bare substring "claude" in that value made `tools` a value hit. Because
clean_markdown drops the *whole key* on a value hit, cleaning such a file
deleted the agent's tool grant outright -- a cosmetic scan turning into silent
data loss on every `.claude/agents/*.md` in a repo. `model:` had the same
problem via AI_FRONTMATTER_KEYS.

Both halves matter and are tested together: the false positives must stop, and
every real watermark must still be caught -- including one hiding inside an
agent definition, which is exactly where an exemption could be abused.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from container_meta import clean_markdown, inspect_markdown

AGENT_FM = (
    "---\n"
    "name: inbox-router\n"
    'description: "Triages the inbox and routes each mail to its project."\n'
    "tools: Read, Write, Edit, mcp__claude_ai_acme__mail__get-message\n"
    "model: opus\n"
    "---\n"
    "\nYou are the inbox router.\n"
)


def _flagged(text: str) -> bool:
    return inspect_markdown(text)[1]


def _findings(text: str) -> list[str]:
    return inspect_markdown(text)[2]


# --- false positives that must NOT fire ------------------------------------


def test_agent_tools_value_naming_claude_is_not_provenance() -> None:
    assert not _flagged(AGENT_FM), _findings(AGENT_FM)


def test_agent_frontmatter_survives_clean_untouched() -> None:
    cleaned, _actions = clean_markdown(AGENT_FM)
    # The tool grant and the model are what a naive value/key hit would delete.
    assert "mcp__claude_ai_acme__mail__get-message" in cleaned
    assert "model: opus" in cleaned
    assert cleaned == AGENT_FM


def test_prose_merely_mentioning_a_vendor_is_not_provenance() -> None:
    text = "---\ntitle: How to use Claude Code\nauthor: JJ\n---\n\nbody\n"
    assert not _flagged(text)


# --- true positives that must STILL fire ------------------------------------


@pytest.mark.parametrize(
    "key_line",
    [
        "ai_generated: true",
        "generator: Claude",
        "generated-with: Claude",
        "made_with: ChatGPT",
        "written-by: Gemini",
        "c2pa: manifest",
        "model: gpt-4",  # not agent-shaped frontmatter, so still provenance
    ],
)
def test_provenance_keys_still_caught(key_line: str) -> None:
    text = f"---\ntitle: Notes\n{key_line}\n---\n\nbody\n"
    assert _flagged(text), f"{key_line!r} should be flagged"


def test_watermark_inside_an_agent_definition_is_still_caught() -> None:
    """The exemption covers key names only -- values are always scanned.

    Otherwise adding `name`/`description`/`tools` would be a way to smuggle a
    watermark past the scanner.
    """
    text = AGENT_FM.replace(
        'description: "Triages the inbox and routes each mail to its project."',
        "description: Generated with Claude Code",
    )
    assert _flagged(text)
    assert any("description" in f for f in _findings(text))
    cleaned, actions = clean_markdown(text)
    assert "description" not in cleaned.split("---")[1]
    assert actions


def test_agent_shape_requires_all_three_keys() -> None:
    """Two of the three is ordinary frontmatter and gets no exemption."""
    text = "---\nname: notes\ndescription: about things\nmodel: gpt-4\n---\n\nbody\n"
    assert _flagged(text), "no tools key, so not an agent definition"
