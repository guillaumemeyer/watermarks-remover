"""Installer coverage for the Claude Code, Cowork, and Cursor targets."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import install_skill

SKILLS = ("clean-user-facing-text", "remove-ai-marks")


def _run(home: Path, *args: str, check: bool = True, extra_env: dict[str, str] | None = None):
    env = os.environ.copy()
    for key in ("CURSOR_HOME", "CLAUDE_CONFIG_DIR"):
        env.pop(key, None)
    env.update({"HOME": str(home), "USERPROFILE": str(home)})
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, str(ROOT / "install_skill.py"), *args],
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


# --------------------------------------------------------------------------
# spec compliance
# --------------------------------------------------------------------------


@pytest.mark.parametrize("skill", SKILLS)
def test_shipped_skills_satisfy_the_upload_spec(skill):
    # claude.ai uploads, the Skills API, and package_skill.py reject any
    # frontmatter key outside the spec, so a shipped skill must stay inside it.
    fields = install_skill.validate_skill(ROOT / "skills" / skill)

    assert fields["name"] == skill
    assert set(fields) <= set(install_skill.SPEC_FRONTMATTER_FIELDS)
    assert 0 < len(fields["description"]) <= install_skill.MAX_DESCRIPTION_LEN


def _write_skill(directory: Path, frontmatter: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n\nBody.\n", encoding="utf-8")
    return directory


def test_validate_rejects_non_spec_frontmatter_key(tmp_path):
    skill = _write_skill(
        tmp_path / "demo-skill",
        "name: demo-skill\ndescription: Demo.\nargument-hint: [file]",
    )

    with pytest.raises(install_skill.SkillError, match="argument-hint"):
        install_skill.validate_skill(skill)


def test_validate_rejects_name_directory_mismatch(tmp_path):
    skill = _write_skill(tmp_path / "demo-skill", "name: other-skill\ndescription: Demo.")

    with pytest.raises(install_skill.SkillError, match="does not match"):
        install_skill.validate_skill(skill)


def test_validate_rejects_oversized_description(tmp_path):
    long_description = "x" * (install_skill.MAX_DESCRIPTION_LEN + 1)
    skill = _write_skill(
        tmp_path / "demo-skill", f"name: demo-skill\ndescription: {long_description}"
    )

    with pytest.raises(install_skill.SkillError, match="limit is"):
        install_skill.validate_skill(skill)


def test_folded_description_is_joined_into_one_line():
    fields = install_skill.parse_frontmatter(
        "---\nname: demo\ndescription: >\n  first line\n  second line\n---\n"
    )

    assert fields["description"] == "first line second line"


# --------------------------------------------------------------------------
# Claude Code
# --------------------------------------------------------------------------


@pytest.mark.parametrize("skill", SKILLS)
def test_claude_code_target_installs_into_claude_home(tmp_path, skill):
    _run(tmp_path, "--skill", skill, "--target", "claude-code")

    assert (tmp_path / ".claude" / "skills" / skill / "SKILL.md").is_file()


def test_claude_code_target_honours_claude_config_dir(tmp_path):
    config_dir = tmp_path / "elsewhere"

    _run(
        tmp_path,
        "--skill",
        "remove-ai-marks",
        "--target",
        "claude-code",
        extra_env={"CLAUDE_CONFIG_DIR": str(config_dir)},
    )

    assert (config_dir / "skills" / "remove-ai-marks" / "SKILL.md").is_file()


def test_claude_code_target_preserves_existing_install_without_force(tmp_path):
    existing = tmp_path / ".claude" / "skills" / "remove-ai-marks"
    existing.mkdir(parents=True)
    (existing / "sentinel").write_text("keep", encoding="utf-8")

    result = _run(tmp_path, "--skill", "remove-ai-marks", "--target", "claude-code", check=False)

    assert result.returncode == 1
    assert (existing / "sentinel").read_text(encoding="utf-8") == "keep"


def test_claude_code_force_backs_up_and_replaces(tmp_path):
    destination = tmp_path / ".claude" / "skills" / "remove-ai-marks"
    destination.mkdir(parents=True)
    (destination / "old").write_text("old", encoding="utf-8")

    _run(tmp_path, "--skill", "remove-ai-marks", "--target", "claude-code", "--force")

    backups = list(destination.parent.glob("remove-ai-marks.backup.*"))
    assert len(backups) == 1
    assert (backups[0] / "old").read_text(encoding="utf-8") == "old"
    assert (destination / "SKILL.md").is_file()


def test_claude_project_target_installs_into_project_directory(tmp_path):
    project = tmp_path / "repo"

    _run(
        tmp_path,
        "--skill",
        "remove-ai-marks",
        "--target",
        "claude-project",
        "--project-dir",
        str(project),
    )

    assert (project / ".claude" / "skills" / "remove-ai-marks" / "SKILL.md").is_file()


def test_link_installs_a_symlink_to_the_repository_skill(tmp_path):
    try:
        (tmp_path / "probe").symlink_to(ROOT, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available for this user")

    _run(tmp_path, "--skill", "remove-ai-marks", "--target", "claude-code", "--link")

    destination = tmp_path / ".claude" / "skills" / "remove-ai-marks"
    assert destination.is_symlink()
    assert destination.resolve() == (ROOT / "skills" / "remove-ai-marks").resolve()


# --------------------------------------------------------------------------
# Cowork bundle
# --------------------------------------------------------------------------


@pytest.mark.parametrize("skill", SKILLS)
def test_cowork_bundle_has_one_top_level_directory_named_after_the_skill(tmp_path, skill):
    bundle = tmp_path / f"{skill}.zip"

    _run(tmp_path, "--skill", skill, "--target", "cowork", "-o", str(bundle))

    names = zipfile.ZipFile(bundle).namelist()
    assert names, "bundle is empty"
    assert {name.split("/")[0] for name in names} == {skill}
    assert f"{skill}/SKILL.md" in names
    assert not [name for name in names if "__pycache__" in name or name.endswith(".pyc")]


def test_cowork_bundle_default_path_is_dist(tmp_path):
    default_bundle = ROOT / "dist" / "remove-ai-marks.zip"
    existed = default_bundle.exists()

    try:
        _run(tmp_path, "--skill", "remove-ai-marks", "--target", "cowork", "--force")
        assert default_bundle.is_file()
    finally:
        if not existed:
            default_bundle.unlink(missing_ok=True)


def test_cowork_bundle_is_reproducible(tmp_path):
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    _run(tmp_path, "--skill", "remove-ai-marks", "--target", "cowork", "-o", str(first))
    _run(tmp_path, "--skill", "remove-ai-marks", "--target", "cowork", "-o", str(second))

    assert first.read_bytes() == second.read_bytes()


def test_cowork_bundle_refuses_to_overwrite_without_force(tmp_path):
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"keep")

    result = _run(
        tmp_path, "--skill", "remove-ai-marks", "--target", "cowork", "-o", str(bundle), check=False
    )

    assert result.returncode == 1
    assert bundle.read_bytes() == b"keep"


def test_cowork_bundle_scripts_stay_executable(tmp_path):
    bundle = tmp_path / "text.zip"

    _run(tmp_path, "--skill", "clean-user-facing-text", "--target", "cowork", "-o", str(bundle))

    with zipfile.ZipFile(bundle) as archive:
        modes = {
            info.filename: (info.external_attr >> 16) & 0o777
            for info in archive.infolist()
            if info.filename.endswith(".py")
        }
    assert modes
    assert all(mode in (0o644, 0o755) for mode in modes.values())


# --------------------------------------------------------------------------
# CLI surface
# --------------------------------------------------------------------------


def test_list_reports_every_shipped_skill(tmp_path):
    result = _run(tmp_path, "--list")

    assert sorted(result.stdout.split()) == sorted(SKILLS)


def test_unknown_skill_fails_without_touching_the_filesystem(tmp_path):
    result = _run(tmp_path, "--skill", "nope", "--target", "claude-code", check=False)

    assert result.returncode == 2
    assert "unknown skill" in result.stderr
    assert not (tmp_path / ".claude").exists()
