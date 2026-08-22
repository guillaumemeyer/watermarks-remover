"""Smoke tests for the standalone Cursor text skill."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "clean-user-facing-text"


def test_lightweight_clean_text_cli():
    result = subprocess.run(
        [sys.executable, str(SKILL / "scripts" / "clean_text.py"), "-", "--stats"],
        input="Hello\u200b world\u3000again",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )

    assert result.stdout.rstrip("\n") == "Hello world again"
    assert '"removed_count": 1' in result.stderr
    assert '"replaced_count": 1' in result.stderr


def _run_vendored_clean_and_inspect(cps):
    payload = "a" + "".join(map(chr, cps)) + "b"

    cleaned = subprocess.run(
        [sys.executable, str(SKILL / "scripts" / "clean_text.py"), "-", "--stats"],
        input=payload,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    assert cleaned.stdout.rstrip("\n") == "ab"
    assert f'"removed_count": {len(cps)}' in cleaned.stderr

    inspected = subprocess.run(
        [sys.executable, str(SKILL / "scripts" / "inspect_text.py"), "-", "--json"],
        input=payload,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    assert inspected.returncode == 1  # suspicious input exits 1 by design
    report = json.loads(inspected.stdout)
    return {hit["kind"] for hit in report["hits"]}


def test_lightweight_clean_strips_reserved_default_ignorables():
    # The vendored engine must match the service engine on the unassigned
    # Default_Ignorable ranges (PropList.txt Other_Default_Ignorable_Code_Point).
    cps = [
        0x2065,
        0xE0000,
        *range(0xFFF0, 0xFFF9),
        *range(0xE0080, 0xE0100),
        *range(0xE01F0, 0xE1000),
    ]
    assert _run_vendored_clean_and_inspect(cps) == {"reserved_ignorable"}


def test_lightweight_clean_strips_noncharacters():
    # The vendored engine must match the service engine on the 66 Unicode
    # noncharacters (U+FDD0..U+FDEF plus U+nFFFE/U+nFFFF in every plane).
    cps = list(range(0xFDD0, 0xFDF0)) + [
        plane << 16 | low for plane in range(0x11) for low in (0xFFFE, 0xFFFF)
    ]
    assert _run_vendored_clean_and_inspect(cps) == {"noncharacter"}


def test_lightweight_skill_has_no_template_placeholders():
    skill_text = (SKILL / "SKILL.md").read_text(encoding="utf-8")

    assert "TODO" not in skill_text


def test_lightweight_skill_includes_required_references():
    assert (SKILL / "references" / "watermark-notes.md").is_file()
    assert (SKILL / "references" / "writing-in-your-voice.md").is_file()


def _run_installer(home: Path, *args: str, check: bool = True):
    env = os.environ.copy()
    env.pop("CURSOR_HOME", None)
    env.update({"HOME": str(home), "USERPROFILE": str(home)})
    return subprocess.run(
        [sys.executable, str(ROOT / "install_skill.py"), *args],
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def test_installer_uses_cursor_skill_location(tmp_path):
    _run_installer(tmp_path)

    installed_skill = tmp_path / ".cursor" / "skills" / SKILL.name
    assert (installed_skill / "SKILL.md").is_file()
    assert (installed_skill / "references" / "writing-in-your-voice.md").is_file()


def test_installer_preserves_existing_install_without_force(tmp_path):
    cursor_install = tmp_path / ".cursor" / "skills" / SKILL.name
    cursor_install.mkdir(parents=True)
    (cursor_install / "sentinel").write_text("keep", encoding="utf-8")

    result = _run_installer(tmp_path, check=False)

    assert result.returncode == 1
    assert (cursor_install / "sentinel").read_text(encoding="utf-8") == "keep"


def test_installer_force_creates_backup_and_replaces(tmp_path):
    destination = tmp_path / ".cursor" / "skills" / SKILL.name
    destination.mkdir(parents=True)
    (destination / "old").write_text("old", encoding="utf-8")

    _run_installer(tmp_path, "--force")

    backups = list(destination.parent.glob(f"{SKILL.name}.backup.*"))
    assert len(backups) == 1
    assert (backups[0] / "old").read_text(encoding="utf-8") == "old"
    assert (destination / "SKILL.md").is_file()


def test_vendored_text_unicode_is_identical_to_service_engine():
    # The Layer A engine is vendored byte-for-byte; only the CLI wrappers
    # (clean_text.py, inspect_text.py, common.py) may differ. Any engine
    # change must be applied to both copies in the same commit.
    service = (ROOT / "service" / "scripts" / "text_unicode.py").read_bytes()
    vendored = (SKILL / "scripts" / "text_unicode.py").read_bytes()
    assert service == vendored


def test_lightweight_preserves_legitimate_bidi_and_emoji_glue():
    # Regression for engine drift: the vendored copy once blanket-stripped
    # RTL isolates/marks and VS16 after arrows, corrupting legitimate text.
    for raw in ("السعر ⁦123 USD⁩‏", "Move ↔️"):
        result = subprocess.run(
            [sys.executable, str(SKILL / "scripts" / "clean_text.py"), "-"],
            input=raw,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        assert result.stdout.rstrip("\n") == raw


def test_lightweight_preserves_egyptian_format_controls():
    # The vendored engine must match the service engine on visible-layout
    # format controls: preserved next to their script, stripped when floating.
    kept = "\U00013079\U00013430\U000130a7"
    result = subprocess.run(
        [sys.executable, str(SKILL / "scripts" / "clean_text.py"), "-"],
        input=kept,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    assert result.stdout.rstrip("\n") == kept

    floating = "a\U00013430b"
    result = subprocess.run(
        [sys.executable, str(SKILL / "scripts" / "clean_text.py"), "-"],
        input=floating,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    assert result.stdout.rstrip("\n") == "ab"
