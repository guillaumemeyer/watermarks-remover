from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _install(home: Path) -> Path:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["WATERMARKS_REMOVER_BIN"] = str(home / "bin with spaces")
    subprocess.run(
        [sys.executable, str(ROOT / "install_cli.py")],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return home / "bin with spaces" / "watermarks-remover"


def test_installed_command_runs_outside_checkout_and_forwards_options(tmp_path: Path):
    launcher = _install(tmp_path / "home")
    other = tmp_path / "unrelated cwd"
    other.mkdir()
    source = tmp_path / "input with spaces.txt"
    output = tmp_path / "custom output.txt"
    source.write_text("Hello\u200bworld", encoding="utf-8")

    result = subprocess.run(
        [str(launcher), str(source), "-o", str(output), "--json", "--nfkc"],
        cwd=other,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8") == "Helloworld"
    assert '"output":' in result.stdout


def test_no_args_prompts_for_input_and_optional_output(tmp_path: Path):
    launcher = _install(tmp_path / "home")
    other = tmp_path / "unrelated cwd"
    other.mkdir()
    source = tmp_path / "source with spaces.txt"
    output = tmp_path / "destination with spaces.txt"
    source.write_text("Hello\u200bworld", encoding="utf-8")

    result = subprocess.run(
        [str(launcher)],
        cwd=other,
        input=f"{source}\n{output}\n",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8") == "Helloworld"
    assert "File to clean:" in result.stdout
    assert "Output file" in result.stdout


def test_no_args_eof_is_polite(tmp_path: Path):
    launcher = _install(tmp_path / "home")
    result = subprocess.run([str(launcher)], input="", capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "Cancelled." in result.stderr


def test_relative_paths_use_callers_cwd_and_default_output(tmp_path: Path):
    launcher = _install(tmp_path / "home")
    work = tmp_path / "work"
    work.mkdir()
    source = work / "input.txt"
    source.write_text("Hello\u200bworld", encoding="utf-8")

    result = subprocess.run(
        [str(launcher), "input.txt", "--json"],
        cwd=work,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (work / "input.cleaned.txt").read_text(encoding="utf-8") == "Helloworld"


def test_tilde_expands_in_equals_output_option(tmp_path: Path):
    home = tmp_path / "home"
    launcher = _install(home)
    source = home / "input.txt"
    output = home / "out with spaces.txt"
    source.write_text("Hello\u200bworld", encoding="utf-8")

    result = subprocess.run(
        [str(launcher), str(source), f"--output=~/{output.name}"],
        env={**os.environ, "HOME": str(home)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8") == "Helloworld"
