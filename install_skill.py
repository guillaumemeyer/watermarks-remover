#!/usr/bin/env python3
"""Install this repository's agent skills into a supported host.

Targets:

  claude-code     ~/.claude/skills/<skill>            (personal, all projects)
  claude-project  <project>/.claude/skills/<skill>    (one repository)
  cowork          <skill>.zip to upload in Customize > Skills
  cursor          ~/.cursor/skills/<skill>            (default, historical)

Cowork (and cloud/routine) sessions do not read ~/.claude/skills on the local
machine: they load the skills enabled for the claude.ai account. So the Cowork
target builds an upload bundle instead of writing to a directory.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKILLS_DIR = ROOT / "skills"
DEFAULT_SKILL = "clean-user-facing-text"
DEFAULT_TARGET = "cursor"

# Fields the Agent Skills spec allows. claude.ai uploads, the Skills API and
# package_skill.py reject anything else with a hard error, so the packager and
# the installers refuse it here rather than at upload time.
SPEC_FRONTMATTER_FIELDS = (
    "allowed-tools",
    "compatibility",
    "description",
    "license",
    "metadata",
    "name",
)
NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
XML_TAG_PATTERN = re.compile(r"<[A-Za-z/!?][^>]*>")
RESERVED_NAME_WORDS = ("anthropic", "claude")
MAX_NAME_LEN = 64
MAX_DESCRIPTION_LEN = 1024
MAX_PACKAGE_BYTES = 30 * 1024 * 1024

EXCLUDED_NAMES = {"__pycache__", ".DS_Store", ".git", ".pytest_cache"}
EXCLUDED_SUFFIXES = (".pyc", ".pyo")

# Fixed timestamp so packaging the same skill twice yields identical bytes.
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class SkillError(RuntimeError):
    """A skill is missing, malformed, or violates the upload spec."""


# --------------------------------------------------------------------------
# discovery and validation
# --------------------------------------------------------------------------


def available_skills() -> list[str]:
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(p.name for p in SKILLS_DIR.iterdir() if (p / "SKILL.md").is_file())


def _split_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        raise SkillError("SKILL.md does not start with YAML frontmatter")
    rest = text.split("\n", 1)[1] if "\n" in text else ""
    end = re.search(r"^---\s*$", rest, re.MULTILINE)
    if end is None:
        raise SkillError("SKILL.md frontmatter is not terminated by ---")
    return rest[: end.start()]


def parse_frontmatter(text: str) -> dict[str, str]:
    """Parse the small YAML subset skills actually use.

    Handles plain scalars and folded/literal block scalars; nested mappings
    (``metadata``) are kept as raw text since only the top-level key names
    matter for validation.
    """
    block = _split_frontmatter(text)
    fields: dict[str, str] = {}
    lines = block.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[:1].isspace() or line.lstrip().startswith("- "):
            continue  # continuation of a value handled below
        match = re.match(r"^([A-Za-z0-9_.-]+):\s?(.*)$", line)
        if match is None:
            raise SkillError(f"cannot parse SKILL.md frontmatter line: {line!r}")
        key, inline = match.group(1), match.group(2).strip()
        body: list[str] = []
        while index < len(lines) and (not lines[index].strip() or lines[index][:1].isspace()):
            body.append(lines[index].strip())
            index += 1
        if inline in (">", ">-", ">+"):
            value = " ".join(part for part in body if part)
        elif inline in ("|", "|-", "|+"):
            value = "\n".join(body).strip()
        elif inline:
            value = inline
        else:
            value = "\n".join(part for part in body if part)
        fields[key] = value.strip()
    return fields


def _normalized(name: str) -> str:
    return name.replace("_", "-").lower()


def validate_skill(source: Path) -> dict[str, str]:
    """Check a skill directory against the Agent Skills spec. Returns fields."""
    skill_md = source / "SKILL.md"
    if not skill_md.is_file():
        raise SkillError(f"missing SKILL.md in {source}")

    fields = parse_frontmatter(skill_md.read_text(encoding="utf-8"))

    unexpected = sorted(set(fields) - set(SPEC_FRONTMATTER_FIELDS))
    if unexpected:
        raise SkillError(
            f"unexpected key(s) in SKILL.md frontmatter: {', '.join(unexpected)}. "
            f"Allowed properties are: {', '.join(SPEC_FRONTMATTER_FIELDS)}"
        )

    name = fields.get("name", "")
    if not name:
        raise SkillError("SKILL.md frontmatter is missing 'name'")
    if len(name) > MAX_NAME_LEN:
        raise SkillError(f"skill name exceeds {MAX_NAME_LEN} characters: {name}")
    if not NAME_PATTERN.match(name):
        raise SkillError(f"skill name must be lowercase letters, digits and hyphens: {name}")
    if any(word in name for word in RESERVED_NAME_WORDS):
        raise SkillError(
            f"skill name uses a reserved word ({', '.join(RESERVED_NAME_WORDS)}): {name}"
        )
    if _normalized(name) != _normalized(source.name):
        raise SkillError(f"directory {source.name} does not match frontmatter name {name}")

    description = fields.get("description", "")
    if not description:
        raise SkillError("SKILL.md frontmatter is missing 'description'")
    if len(description) > MAX_DESCRIPTION_LEN:
        raise SkillError(
            f"description is {len(description)} characters; the limit is {MAX_DESCRIPTION_LEN}"
        )
    if XML_TAG_PATTERN.search(description):
        raise SkillError("description must not contain XML tags")

    return fields


def resolve_skill(name: str) -> Path:
    source = SKILLS_DIR / name
    if not (source / "SKILL.md").is_file():
        raise SkillError(f"unknown skill {name!r}; available: {', '.join(available_skills())}")
    return source


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_NAMES for part in path.parts) or path.suffix in EXCLUDED_SUFFIXES


def skill_files(source: Path) -> list[Path]:
    """Every packageable file in the skill, sorted, relative to the skill root."""
    files = [
        path.relative_to(source)
        for path in source.rglob("*")
        if path.is_file() and not _is_excluded(path.relative_to(source))
    ]
    return sorted(files, key=lambda p: p.as_posix())


# --------------------------------------------------------------------------
# directory installs (Claude Code, Cursor)
# --------------------------------------------------------------------------


def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _stage(source: Path, skills_dir: Path) -> tuple[Path, Path]:
    skills_dir.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{source.name}.staging.", dir=skills_dir))
    staged_skill = staging_root / source.name
    try:
        shutil.copytree(
            source,
            staged_skill,
            ignore=shutil.ignore_patterns(*EXCLUDED_NAMES, "*.pyc", "*.pyo"),
        )
        if not (staged_skill / "SKILL.md").is_file():
            raise SkillError("staged skill is missing SKILL.md")
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return staging_root, staged_skill


def _backup_name(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.backup.{uuid.uuid4().hex[:12]}")


def install_directory(
    source: Path, destination: Path, force: bool, link: bool
) -> tuple[bool, Path | None]:
    if _present(destination) and not force:
        print(f"already exists: {destination}", file=sys.stderr)
        print(
            "No changes made. Re-run with --force to back up and replace.",
            file=sys.stderr,
        )
        return False, None

    if link:
        destination.parent.mkdir(parents=True, exist_ok=True)
        backup: Path | None = None
        if _present(destination):
            backup = _backup_name(destination)
            os.replace(destination, backup)
        try:
            destination.symlink_to(source, target_is_directory=True)
        except BaseException:
            if backup is not None and not _present(destination):
                os.replace(backup, destination)
            raise
        return True, backup

    staging_root, staged_skill = _stage(source, destination.parent)
    backup = None
    try:
        if _present(destination):
            backup = _backup_name(destination)
            os.replace(destination, backup)
        try:
            os.replace(staged_skill, destination)
        except BaseException:
            if backup is not None and not _present(destination):
                os.replace(backup, destination)
            raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    return True, backup


# --------------------------------------------------------------------------
# Cowork bundle
# --------------------------------------------------------------------------


def package_skill(source: Path, output: Path, force: bool) -> Path:
    """Write an upload bundle: one top-level directory named after the skill."""
    files = skill_files(source)
    total = sum((source / rel).stat().st_size for rel in files)
    if total > MAX_PACKAGE_BYTES:
        raise SkillError(
            f"skill is {total} bytes uncompressed; the upload limit is {MAX_PACKAGE_BYTES}"
        )

    if _present(output) and not force:
        print(f"already exists: {output}", file=sys.stderr)
        print("No changes made. Re-run with --force to overwrite.", file=sys.stderr)
        raise FileExistsError(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(temp_fd)
    temp_path = Path(temp_name)
    try:
        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as bundle:
            for rel in files:
                info = zipfile.ZipInfo(f"{source.name}/{rel.as_posix()}", date_time=ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                mode = 0o755 if os.access(source / rel, os.X_OK) else 0o644
                info.external_attr = mode << 16
                bundle.writestr(info, (source / rel).read_bytes())
        os.chmod(temp_path, 0o644)  # mkstemp defaults to 0600; this is a build artifact
        os.replace(temp_path, output)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return output


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _home(explicit: str | None, env_var: str, default: Path) -> Path:
    value = explicit or os.environ.get(env_var)
    return Path(value).expanduser() if value else default


def destination_for(args: argparse.Namespace, skill: str) -> Path:
    home = Path.home()
    if args.target == "cursor":
        return _home(args.cursor_home, "CURSOR_HOME", home / ".cursor") / "skills" / skill
    if args.target == "claude-code":
        return _home(args.claude_home, "CLAUDE_CONFIG_DIR", home / ".claude") / "skills" / skill
    if args.target == "claude-project":
        project = Path(args.project_dir or Path.cwd()).expanduser().resolve()
        return project / ".claude" / "skills" / skill
    raise SkillError(f"target {args.target} does not install into a directory")


HOST_HINTS = {
    "cursor": "Start a new Cursor session if the skill does not appear automatically.",
    "claude-code": (
        "Claude Code picks up new personal skills without a restart; run /skills to confirm."
    ),
    "claude-project": (
        "Claude Code loads project skills from .claude/skills in the working directory "
        "and its parents. Commit the directory to share it (cloud sessions read it too)."
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skill",
        default=DEFAULT_SKILL,
        help=f"Skill to install (default: {DEFAULT_SKILL}). Use --list to see all.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        choices=("cursor", "claude-code", "claude-project", "cowork"),
        help=f"Where to install (default: {DEFAULT_TARGET})",
    )
    parser.add_argument("--list", action="store_true", help="List available skills and exit")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Back up and replace an existing installation (or overwrite the bundle)",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="Symlink the repository skill instead of copying (directory targets only)",
    )
    parser.add_argument("--cursor-home", help="Override Cursor home (default: ~/.cursor)")
    parser.add_argument("--claude-home", help="Override Claude Code home (default: ~/.claude)")
    parser.add_argument(
        "--project-dir",
        help="Project root for --target claude-project (default: current directory)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Bundle path for --target cowork (default: dist/<skill>.zip)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        for name in available_skills():
            print(name)
        return 0

    try:
        source = resolve_skill(args.skill)
        validate_skill(source)
    except SkillError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.target == "cowork":
        if args.link:
            print("error: --link does not apply to the cowork target", file=sys.stderr)
            return 2
        default_output = ROOT / "dist" / f"{args.skill}.zip"
        output = Path(args.output).expanduser() if args.output else default_output
        try:
            bundle = package_skill(source, output, args.force)
        except FileExistsError:
            return 1
        except SkillError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        print(f"Cowork: wrote {bundle}")
        print(
            "Upload it in the Claude Desktop app under Customize > Skills > Add "
            "(or in the skills settings on claude.ai). Cowork, cloud and routine "
            "sessions load skills enabled for your account, not ~/.claude/skills."
        )
        return 0

    try:
        destination = destination_for(args, args.skill)
    except SkillError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    installed, backup = install_directory(source, destination, args.force, args.link)
    if not installed:
        return 1
    label = {
        "cursor": "Cursor",
        "claude-code": "Claude Code",
        "claude-project": "Claude Code (project)",
    }[args.target]
    if backup is not None:
        print(f"{label}: backed up existing skill to {backup}")
    verb = "linked" if args.link else "installed"
    print(f"{label}: {verb} {destination}")
    print(HOST_HINTS[args.target])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
