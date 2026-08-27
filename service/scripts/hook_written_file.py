#!/usr/bin/env python3
"""Inspect or clean a file an agent just wrote, driven by a PostToolUse hook.

A skill only ever *asks* a model to clean its output, and the model decides
whether to comply. A hook is executed by the harness, so it runs on every
matching tool call whether or not the model cooperates. That makes this the
deterministic half of the workflow — but only for files on disk: no hook can
rewrite the assistant's chat message before the user sees it (Claude Code's
Stop hook receives `last_assistant_message` read-only).

Reads a hook payload on stdin (Claude Code PostToolUse shape):

    {"tool_name": "Write", "tool_input": {"file_path": "..."}, "cwd": "..."}

Modes:
  check  report provenance marks, leave the file alone (default)
  clean  strip them in place, then report what changed

Exit codes follow the PostToolUse contract: 0 = nothing to say, 2 = stderr is
shown to the model. It never blocks a tool call, because PostToolUse fires
after the tool has already run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_lib import is_actionable, scan_file
from common import MAX_INPUT_BYTES, eprint, subprocess_creationflags

CLEAN_FILE_PY = Path(__file__).resolve().parent / "clean_file.py"

# Tools whose payload names a single file the agent just wrote. The hook
# matcher should filter these too; this is the defensive second check.
FILE_WRITING_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit", "Update"})

MODES = ("check", "clean")
DEFAULT_MODE = "check"

EXIT_QUIET = 0
EXIT_SHOW_MODEL = 2  # PostToolUse: stderr is surfaced to the model
EXIT_HOOK_ERROR = 1  # non-blocking "hook error" notice


def resolve_mode(requested: str | None) -> str:
    """CLI flag first, then the plugin option, then the env var, then check.

    The mode deliberately does *not* travel through ``${user_config.hook_mode}``
    in hooks.json. Claude Code 2.1.235 refuses to run a hook whose command
    references an option the user has never opened /plugin manage to set --
    "Plugin option \"hook_mode\" isn't set" -- and a declared `default` does
    not satisfy it. Substituting it would mean the hook silently never runs for
    anyone who installed the plugin and changed nothing, which is the worst
    possible default for a deterministic guard. Claude Code exports a set
    option as CLAUDE_PLUGIN_OPTION_<KEY>, so read that instead: unset simply
    falls through to the default.
    """
    for candidate in (
        requested,
        os.environ.get("CLAUDE_PLUGIN_OPTION_HOOK_MODE"),
        os.environ.get("WATERMARKS_HOOK_MODE"),
    ):
        value = (candidate or "").strip().lower()
        if not value:
            continue
        if value in MODES:
            return value
        eprint(f"watermarks-remover: unknown hook mode {value!r}; using {DEFAULT_MODE}")
        return DEFAULT_MODE
    return DEFAULT_MODE


def target_path(payload: dict) -> Path | None:
    """The file the tool call wrote, or None when the payload names no file."""
    if payload.get("tool_name") not in FILE_WRITING_TOOLS:
        return None

    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    raw = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not isinstance(raw, str) or not raw.strip():
        return None

    path = Path(raw).expanduser()
    if not path.is_absolute():
        # Hook payloads may carry a project-relative path; cwd is the session's.
        path = Path(payload.get("cwd") or Path.cwd()) / path
    return path


def _emit(system_message: str, additional_context: str | None = None) -> None:
    """Print the PostToolUse JSON the harness reads from stdout."""
    output: dict[str, object] = {
        "systemMessage": system_message,
        "hookSpecificOutput": {"hookEventName": "PostToolUse"},
    }
    if additional_context:
        output["additionalContext"] = additional_context
    print(json.dumps(output))


def run_check(path: Path) -> int:
    # scan_file/is_actionable are audit_dir.py's own per-file logic, so this
    # hook, the pre-commit gate, and the CI SARIF export agree on what counts.
    item = scan_file(path)
    if item.get("kind") == "unknown" or not is_actionable(item):
        return EXIT_QUIET

    findings = list(item.get("findings", []))
    if item.get("has_c2pa"):
        findings.append("C2PA manifest present")
    if item.get("has_ai_metadata"):
        findings.append("AI-generator metadata present")

    detail = "; ".join(findings) or "provenance marks detected"
    _emit(f"watermarks-remover: {path.name} carries provenance marks ({detail})")
    eprint(f"watermarks-remover: {path} carries AI/C2PA provenance marks:")
    for finding in findings:
        eprint(f"  - {finding}")
    eprint(
        f"Strip them with `python3 {CLEAN_FILE_PY} {path} --in-place`, "
        "or set WATERMARKS_HOOK_MODE=clean to have this hook do it."
    )
    return EXIT_SHOW_MODEL


def run_clean(path: Path) -> int:
    # Clean to a sibling temp file and swap only on a real difference: this
    # hook fires on every write, and rewriting identical bytes would churn
    # mtimes and retrigger file watchers on files that were already clean.
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        proc = subprocess.run(
            [sys.executable, str(CLEAN_FILE_PY), str(path), "-o", str(temp_path), "--json"],
            capture_output=True,
            text=True,
            check=False,
            creationflags=subprocess_creationflags,
        )
        if proc.returncode == 2:
            # Unrecognized format or oversized input; clean_file.py explained
            # why on stderr. Not this hook's problem — stay quiet.
            return EXIT_QUIET

        # clean_file.py returns 1 for two different things: a genuine failure,
        # and a successful clean that left residual signals (a degraded PDF, an
        # image whose C2PA scan still trips). They are told apart by whether it
        # printed its JSON report, so parse first and judge on that — treating
        # every exit 1 as failure threw away real cleans.
        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError:
            result = None
        if not isinstance(result, dict) or proc.returncode not in (0, 1):
            detail = proc.stderr.strip() or "clean produced no usable report"
            eprint(f"watermarks-remover: {path}: {detail}")
            return EXIT_HOOK_ERROR

        residual = result.get("still_has_c2pa") or result.get("still_has_ai_metadata")
        if temp_path.read_bytes() == path.read_bytes():
            if residual:
                findings = result.get("post_findings") or []
                detail = "; ".join(str(finding) for finding in findings) or _describe(result)
                _emit(
                    f"watermarks-remover: {path.name} was left unchanged; "
                    f"cleanup left residual or inconclusive signals ({detail})",
                    additional_context=(
                        f"The watermarks-remover hook could not confirm that {path} is clean. "
                        f"The file was left unchanged ({detail})."
                    ),
                )
            return EXIT_QUIET

        # mkstemp creates the temp file 0600; without this the swap would strip
        # the original's permissions, silently de-executabling a script.
        shutil.copymode(path, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)

    summary = _describe(result)
    if residual:
        summary += "; residual provenance signals may remain"
    _emit(
        f"watermarks-remover: cleaned {path.name} in place ({summary})",
        additional_context=(
            f"The watermarks-remover hook stripped provenance marks from {path} "
            f"after the write ({summary}). The file on disk no longer matches "
            "what was written; re-read it before editing again."
        ),
    )
    return EXIT_QUIET


def _describe(result: dict) -> str:
    stats = result.get("stats")
    if isinstance(stats, dict):
        removed = stats.get("removed_count") or 0
        replaced = stats.get("replaced_count") or 0
        parts = []
        if removed:
            parts.append(f"{removed} character(s) removed")
        if replaced:
            parts.append(f"{replaced} replaced")
        if parts:
            return ", ".join(parts)
    actions = result.get("actions")
    if isinstance(actions, list) and actions:
        return "; ".join(str(action) for action in actions)
    return "metadata stripped"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        default=None,
        help=(
            "check (report only, default) or clean (strip in place). Falls back to "
            "CLAUDE_PLUGIN_OPTION_HOOK_MODE, then WATERMARKS_HOOK_MODE, then check."
        ),
    )
    args = parser.parse_args(argv)

    raw = sys.stdin.read()
    if not raw.strip():
        return EXIT_QUIET
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        eprint("watermarks-remover: hook payload was not valid JSON")
        return EXIT_HOOK_ERROR
    if not isinstance(payload, dict):
        eprint("watermarks-remover: hook payload was not a JSON object")
        return EXIT_HOOK_ERROR

    path = target_path(payload)
    if path is None or not path.is_file():
        return EXIT_QUIET
    if path.stat().st_size > MAX_INPUT_BYTES:
        return EXIT_QUIET

    mode = resolve_mode(args.mode)
    try:
        return run_clean(path) if mode == "clean" else run_check(path)
    except OSError as error:
        # A hook must never take the session down with it.
        eprint(f"watermarks-remover: hook failed on {path}: {error}")
        return EXIT_HOOK_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
