"""PostToolUse hook: what the harness runs after an agent writes a file.

A skill asks the model to clean its output; this hook is executed by the
harness whether or not the model cooperates. These tests drive it exactly as
the harness does — payload on stdin, decisions in the exit code.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "service" / "scripts" / "hook_written_file.py"
MARKED = ROOT / "tests" / "fixtures" / "sample_watermarked.txt"

EXIT_QUIET = 0
EXIT_HOOK_ERROR = 1
EXIT_SHOW_MODEL = 2


def run_hook(payload, *args: str, env: dict[str, str] | None = None):
    full_env = os.environ.copy()
    # Both mode sources have to be cleared, or an ambient plugin option would
    # silently decide what mode these tests exercise.
    full_env.pop("WATERMARKS_HOOK_MODE", None)
    full_env.pop("CLAUDE_PLUGIN_OPTION_HOOK_MODE", None)
    full_env.update(env or {})
    return subprocess.run(
        [sys.executable, str(HOOK), *args],
        input=payload if isinstance(payload, str) else json.dumps(payload),
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=full_env,
        check=False,
    )


def write_event(path: Path, cwd: Path | None = None, tool: str = "Write") -> dict:
    return {
        "hook_event_name": "PostToolUse",
        "tool_name": tool,
        "tool_input": {"file_path": str(path)},
        "tool_response": {"status": "success"},
        "cwd": str(cwd or path.parent),
    }


@pytest.fixture
def marked_file(tmp_path) -> Path:
    path = tmp_path / "draft.md"
    path.write_bytes(MARKED.read_bytes())
    return path


@pytest.fixture
def clean_file(tmp_path) -> Path:
    path = tmp_path / "clean.md"
    path.write_text("Ordinary prose with nothing hidden in it.\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# check mode (the default)
# --------------------------------------------------------------------------


def test_check_reports_marks_to_the_model_without_touching_the_file(marked_file):
    before = marked_file.read_bytes()

    result = run_hook(write_event(marked_file))

    # Exit 2 is the only PostToolUse channel that reaches the model.
    assert result.returncode == EXIT_SHOW_MODEL
    assert "U+200B" in result.stderr
    assert json.loads(result.stdout)["systemMessage"].startswith("watermarks-remover:")
    assert marked_file.read_bytes() == before


def test_check_stays_silent_on_a_clean_file(clean_file):
    result = run_hook(write_event(clean_file))

    assert result.returncode == EXIT_QUIET
    assert result.stdout == ""


def test_check_is_the_default_when_no_mode_is_configured(marked_file):
    # ${user_config.hook_mode} substitutes to "" when the user never set it.
    result = run_hook(write_event(marked_file), "--mode", "")

    assert result.returncode == EXIT_SHOW_MODEL
    assert marked_file.read_bytes() == MARKED.read_bytes()


def test_unknown_mode_falls_back_to_check_instead_of_cleaning(marked_file):
    result = run_hook(write_event(marked_file), "--mode", "obliterate")

    assert result.returncode == EXIT_SHOW_MODEL
    assert marked_file.read_bytes() == MARKED.read_bytes()
    assert "unknown hook mode" in result.stderr


# --------------------------------------------------------------------------
# clean mode
# --------------------------------------------------------------------------


def test_clean_strips_marks_in_place_and_tells_the_model_the_file_moved(marked_file):
    result = run_hook(write_event(marked_file), "--mode", "clean")

    assert result.returncode == EXIT_QUIET
    payload = json.loads(result.stdout)
    assert "cleaned" in payload["systemMessage"]
    assert "re-read it" in payload["additionalContext"]

    cleaned = marked_file.read_text(encoding="utf-8")
    assert "​" not in cleaned and "­" not in cleaned


def test_clean_mode_can_come_from_the_environment(marked_file):
    result = run_hook(write_event(marked_file), env={"WATERMARKS_HOOK_MODE": "clean"})

    assert result.returncode == EXIT_QUIET
    assert "​" not in marked_file.read_text(encoding="utf-8")


def test_clean_leaves_an_already_clean_file_byte_identical_and_untouched(clean_file):
    before = clean_file.read_bytes()
    before_mtime = clean_file.stat().st_mtime_ns

    result = run_hook(write_event(clean_file), "--mode", "clean")

    # The hook fires on every write; rewriting identical bytes would churn
    # mtimes and retrigger file watchers across a whole session.
    assert result.returncode == EXIT_QUIET
    assert result.stdout == ""
    assert clean_file.read_bytes() == before
    assert clean_file.stat().st_mtime_ns == before_mtime


def test_clean_reports_inconclusive_truncated_mp4_without_rewriting(tmp_path):
    def box(fourcc: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload) + 8) + fourcc + payload

    ftyp = box(b"ftyp", b"isom" + struct.pack(">I", 0) + b"isomiso2mp41")
    mvhd = box(b"mvhd", b"\x00" * 20)
    udta = box(b"udta", b"    toolGenerated by OpenAI Sora" + b"\x00" * 32)
    target = tmp_path / "truncated.mp4"
    target.write_bytes((ftyp + box(b"moov", mvhd + udta))[:-16])
    before = target.read_bytes()
    before_mtime = target.stat().st_mtime_ns

    result = run_hook(write_event(target), "--mode", "clean")

    assert result.returncode == EXIT_QUIET
    assert "not fully inspected" in json.loads(result.stdout)["systemMessage"]
    assert target.read_bytes() == before
    assert target.stat().st_mtime_ns == before_mtime


def test_clean_does_not_leave_temp_files_behind(marked_file):
    run_hook(write_event(marked_file), "--mode", "clean")

    assert [p.name for p in marked_file.parent.iterdir()] == [marked_file.name]


# --------------------------------------------------------------------------
# payload handling
# --------------------------------------------------------------------------


def test_relative_file_path_resolves_against_the_payload_cwd(marked_file):
    payload = {
        "tool_name": "Edit",
        "tool_input": {"file_path": marked_file.name},
        "cwd": str(marked_file.parent),
    }

    result = run_hook(payload)

    assert result.returncode == EXIT_SHOW_MODEL


def test_notebook_edit_payload_uses_notebook_path(marked_file):
    notebook = marked_file.parent / "notes.md"
    notebook.write_bytes(MARKED.read_bytes())
    payload = {
        "tool_name": "NotebookEdit",
        "tool_input": {"notebook_path": str(notebook)},
        "cwd": str(notebook.parent),
    }

    result = run_hook(payload)

    assert result.returncode == EXIT_SHOW_MODEL


@pytest.mark.parametrize(
    "payload",
    [
        {"tool_name": "Bash", "tool_input": {"command": "ls"}},
        {"tool_name": "Write", "tool_input": {}},
        {"tool_name": "Write", "tool_input": {"file_path": "  "}},
        {"tool_name": "Write"},
        {},
    ],
)
def test_payloads_that_name_no_written_file_are_a_silent_no_op(payload):
    result = run_hook(payload)

    assert result.returncode == EXIT_QUIET
    assert result.stdout == ""


def test_missing_file_is_a_silent_no_op(tmp_path):
    result = run_hook(write_event(tmp_path / "never-written.md"))

    assert result.returncode == EXIT_QUIET
    assert result.stdout == ""


def test_empty_stdin_is_a_silent_no_op():
    result = run_hook("")

    assert result.returncode == EXIT_QUIET
    assert result.stdout == ""


@pytest.mark.parametrize("payload", ["not json at all", '["a", "list"]'])
def test_malformed_payload_reports_once_without_a_traceback(payload):
    result = run_hook(payload)

    # A non-blocking hook error: visible, but it must not derail the session.
    assert result.returncode == EXIT_HOOK_ERROR
    assert "Traceback" not in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1


def test_oversized_file_is_skipped_rather_than_scanned(tmp_path, marked_file):
    sys.path.insert(0, str(ROOT / "service" / "scripts"))
    from common import MAX_INPUT_BYTES

    big = tmp_path / "huge.md"
    big.write_bytes(b"x" * (MAX_INPUT_BYTES + 1))

    result = run_hook(write_event(big))

    assert result.returncode == EXIT_QUIET
    assert result.stdout == ""


# --------------------------------------------------------------------------
# clean-mode contract with clean_file.py
# --------------------------------------------------------------------------


def _fake_cleaner(tmp_path: Path, body: str) -> Path:
    """A stand-in for clean_file.py, so exit codes can be exercised directly."""
    script = tmp_path / "fake_clean.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\n"
        "out = sys.argv[sys.argv.index('-o') + 1]\n"
        "Path(out).write_text('cleaned\\n', encoding='utf-8')\n" + body,
        encoding="utf-8",
    )
    return script


def test_exit_one_with_a_report_is_a_clean_that_left_residual_signals(tmp_path, monkeypatch):
    # clean_file.py returns 1 both for failure and for "cleaned, residual
    # signals remain". Treating every 1 as failure threw away real cleans.
    sys.path.insert(0, str(ROOT / "service" / "scripts"))
    import hook_written_file

    target = tmp_path / "doc.md"
    target.write_bytes(MARKED.read_bytes())
    monkeypatch.setattr(
        hook_written_file,
        "CLEAN_FILE_PY",
        _fake_cleaner(
            tmp_path,
            'print(\'{"kind": "container", "still_has_c2pa": true}\')\nsys.exit(1)\n',
        ),
    )

    assert hook_written_file.run_clean(target) == EXIT_QUIET
    assert target.read_text(encoding="utf-8") == "cleaned\n"


def test_exit_one_without_a_report_is_a_failure_and_leaves_the_file_alone(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "service" / "scripts"))
    import hook_written_file

    target = tmp_path / "doc.md"
    original = MARKED.read_bytes()
    target.write_bytes(original)
    monkeypatch.setattr(
        hook_written_file,
        "CLEAN_FILE_PY",
        _fake_cleaner(tmp_path, "sys.stderr.write('boom\\n')\nsys.exit(1)\n"),
    )

    assert hook_written_file.run_clean(target) == EXIT_HOOK_ERROR
    assert target.read_bytes() == original


def test_unparseable_report_is_rejected_rather_than_guessed_at(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "service" / "scripts"))
    import hook_written_file

    target = tmp_path / "doc.md"
    original = MARKED.read_bytes()
    target.write_bytes(original)
    monkeypatch.setattr(
        hook_written_file,
        "CLEAN_FILE_PY",
        _fake_cleaner(tmp_path, "print('not json')\nsys.exit(0)\n"),
    )

    assert hook_written_file.run_clean(target) == EXIT_HOOK_ERROR
    assert target.read_bytes() == original


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
@pytest.mark.parametrize("mode", [0o644, 0o755])
def test_clean_preserves_the_files_permissions(tmp_path, mode):
    # The swap goes through mkstemp, which creates 0600; without copying the
    # mode across, cleaning a script would quietly strip its execute bit.
    target = tmp_path / "script.md"
    target.write_bytes(MARKED.read_bytes())
    target.chmod(mode)

    result = run_hook(write_event(target), "--mode", "clean")

    assert result.returncode == EXIT_QUIET
    assert target.stat().st_mode & 0o777 == mode
    assert "​" not in target.read_text(encoding="utf-8")


def test_plugin_option_sets_the_mode_like_the_env_var(marked_file):
    result = run_hook(write_event(marked_file), env={"CLAUDE_PLUGIN_OPTION_HOOK_MODE": "clean"})

    assert result.returncode == EXIT_QUIET
    assert "​" not in marked_file.read_text(encoding="utf-8")
