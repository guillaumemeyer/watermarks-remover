"""Regression tests for leading UTF-8 BOM handling across inspect, audit, clean, and hook."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_lib import is_actionable, scan_file
from common import read_text_input
from container_meta import clean_container, clean_markdown

UTF8_BOM = b"\xef\xbb\xbf"

HOOK = SCRIPTS / "hook_written_file.py"
CLEAN_FILE = SCRIPTS / "clean_file.py"
INSPECT_FILE = SCRIPTS / "inspect_file.py"
CLEAN_TEXT = SCRIPTS / "clean_text.py"


def _run_hook(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("WATERMARKS_HOOK_MODE", None)
    env.pop("CLAUDE_PLUGIN_OPTION_HOOK_MODE", None)
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Write",
        "tool_input": {"file_path": str(path)},
        "tool_response": {"status": "success"},
        "cwd": str(path.parent),
    }
    return subprocess.run(
        [sys.executable, str(HOOK), *args],
        input=json.dumps(payload),
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=env,
        check=False,
    )


@pytest.mark.parametrize(
    ("filename", "body"),
    [
        ("bom.ps1", 'Write-Host "Größe"\n'.encode()),
        ("bom.md", b"# Title\n"),
        ("bom.html", b"<html><body>Hello</body></html>\n"),
        ("bom.tex", b"\\documentclass{article}\n\\begin{document}Hi\\end{document}\n"),
    ],
)
def test_leading_utf8_bom_is_not_flagged_by_scan_file(tmp_path: Path, filename: str, body: bytes):
    path = tmp_path / filename
    path.write_bytes(UTF8_BOM + body)

    item = scan_file(path)

    assert not is_actionable(item), item
    assert item["findings"] == []
    assert item["suspicious_total"] == 0


def test_embedded_u_feff_after_leading_bom_is_still_flagged_once(tmp_path: Path):
    ps1 = tmp_path / "bom_inner.ps1"
    ps1.write_bytes(UTF8_BOM + b'Write-Host "a\xef\xbb\xbfb"\n')
    md = tmp_path / "bom_inner.md"
    md.write_bytes(UTF8_BOM + b"# Title\na\xef\xbb\xbfb\n")

    ps1_item = scan_file(ps1)
    assert is_actionable(ps1_item)
    assert ps1_item["suspicious_total"] == 1
    assert any("U+FEFF" in f and "x1" in f for f in ps1_item["findings"]), ps1_item["findings"]

    md_item = scan_file(md)
    assert is_actionable(md_item)
    assert md_item["suspicious_total"] == 1
    assert any("U+FEFF" in f and "x1" in f for f in md_item["findings"]), md_item["findings"]


def test_double_leading_bom_flags_and_strips_only_second_feff(tmp_path: Path):
    path = tmp_path / "double.ps1"
    path.write_bytes(UTF8_BOM + UTF8_BOM + b"Write-Host 'ok'\n")

    item = scan_file(path)
    assert is_actionable(item)
    assert item["suspicious_total"] == 1

    out = tmp_path / "double.cleaned.ps1"
    proc = subprocess.run(
        [sys.executable, str(CLEAN_FILE), str(path), "-o", str(out), "--json"],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    report = json.loads(proc.stdout)
    assert report["changed"] is True
    assert report["stats"]["removed_count"] == 1
    assert out.read_bytes() == UTF8_BOM + b"Write-Host 'ok'\n"


def test_post_tool_use_hook_check_and_clean_preserve_leading_utf8_bom(tmp_path: Path):
    clean_ps1 = tmp_path / "bom.ps1"
    clean_ps1_bytes = UTF8_BOM + 'Write-Host "Größe"\n'.encode()
    clean_ps1.write_bytes(clean_ps1_bytes)
    clean_ps1_mtime = clean_ps1.stat().st_mtime_ns

    clean_md = tmp_path / "bom.md"
    clean_md_bytes = UTF8_BOM + b"# Title\n"
    clean_md.write_bytes(clean_md_bytes)
    clean_md_mtime = clean_md.stat().st_mtime_ns

    # check mode stays quiet on clean BOM files
    assert _run_hook(clean_ps1).returncode == 0
    assert _run_hook(clean_md).returncode == 0

    # clean mode leaves clean BOM files byte-identical and untouched
    res_ps1 = _run_hook(clean_ps1, "--mode", "clean")
    assert res_ps1.returncode == 0
    assert res_ps1.stdout == ""
    assert clean_ps1.read_bytes() == clean_ps1_bytes
    assert clean_ps1.stat().st_mtime_ns == clean_ps1_mtime

    res_md = _run_hook(clean_md, "--mode", "clean")
    assert res_md.returncode == 0
    assert res_md.stdout == ""
    assert clean_md.read_bytes() == clean_md_bytes
    assert clean_md.stat().st_mtime_ns == clean_md_mtime

    # inner U+FEFF is reported (x1) in check mode and stripped while keeping leading BOM in clean mode
    inner_ps1 = tmp_path / "bom_inner.ps1"
    inner_ps1.write_bytes(UTF8_BOM + b'Write-Host "a\xef\xbb\xbfb"\n')

    check_res = _run_hook(inner_ps1)
    assert check_res.returncode == 2
    assert "U+FEFF" in check_res.stderr
    assert "x1" in check_res.stderr

    clean_res = _run_hook(inner_ps1, "--mode", "clean")
    assert clean_res.returncode == 0
    assert inner_ps1.read_bytes() == UTF8_BOM + b'Write-Host "ab"\n'


def test_clean_file_and_inspect_file_and_clean_text_preserve_leading_utf8_bom(tmp_path: Path):
    ps1 = tmp_path / "script.ps1"
    ps1.write_bytes(UTF8_BOM + b'Write-Host "a\xef\xbb\xbfb"\n')
    clean_ps1 = tmp_path / "clean.ps1"
    clean_ps1.write_bytes(UTF8_BOM + b'Write-Host "ok"\n')

    inspect_clean = subprocess.run(
        [sys.executable, str(INSPECT_FILE), str(clean_ps1), "--json"],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    assert inspect_clean.returncode == 0
    assert json.loads(inspect_clean.stdout)["suspicious_total"] == 0

    out_ps1 = tmp_path / "script.out.ps1"
    proc = subprocess.run(
        [sys.executable, str(CLEAN_FILE), str(ps1), "-o", str(out_ps1), "--json"],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    assert json.loads(proc.stdout)["changed"] is True
    assert out_ps1.read_bytes() == UTF8_BOM + b'Write-Host "ab"\n'

    out_text = tmp_path / "script.text.ps1"
    subprocess.run(
        [sys.executable, str(CLEAN_TEXT), str(ps1), "-o", str(out_text)],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    assert out_text.read_bytes() == UTF8_BOM + b'Write-Host "ab"\n'

    md = tmp_path / "doc.md"
    md.write_bytes(UTF8_BOM + b"# Title\na\xef\xbb\xbfb\n")
    out_md = tmp_path / "doc.out.md"
    proc_md = subprocess.run(
        [sys.executable, str(CLEAN_FILE), str(md), "-o", str(out_md), "--json"],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    assert json.loads(proc_md.stdout)["changed"] is True
    assert out_md.read_bytes() == UTF8_BOM + b"# Title\nab\n"


def test_read_text_input_strips_leading_utf8_bom_only(tmp_path: Path):
    path = tmp_path / "bom.ps1"
    path.write_bytes(UTF8_BOM + b'Write-Host "a\xef\xbb\xbfb"\n')

    text = read_text_input(str(path))
    assert text == 'Write-Host "a\ufeffb"\n'


def test_clean_markdown_and_container_preserve_untouched_bom_crlf_frontmatter(tmp_path: Path):
    raw = UTF8_BOM + b"---\r\ntitle: Notes\r\n---\r\n# Body\r\n"
    text = "---\r\ntitle: Notes\r\n---\r\n# Body\r\n"

    cleaned_text, actions = clean_markdown(text)
    assert cleaned_text == text
    assert actions == ["no AI frontmatter keys or embedded data URIs removed"]

    src = tmp_path / "notes.md"
    src.write_bytes(raw)
    orig_mtime = src.stat().st_mtime_ns
    out = tmp_path / "notes.cleaned.md"
    result = clean_container(src, out)

    assert result["changed"] is False
    assert out.read_bytes() == raw

    clean_res = _run_hook(src, "--mode", "clean")
    assert clean_res.returncode == 0
    assert clean_res.stdout == ""
    assert src.read_bytes() == raw
    assert src.stat().st_mtime_ns == orig_mtime


def test_clean_container_preserves_untouched_crlf_frontmatter_when_cleaning_body_comment(
    tmp_path: Path,
):
    raw = UTF8_BOM + b"---\r\ntitle: Notes\r\n---\r\n<!-- generated by chatgpt -->\r\n# Body\r\n"
    expected = UTF8_BOM + b"---\r\ntitle: Notes\r\n---\r\n\r\n# Body\r\n"

    src = tmp_path / "commented.md"
    src.write_bytes(raw)
    out = tmp_path / "commented.cleaned.md"
    result = clean_container(src, out)

    assert result["changed"] is True
    assert out.read_bytes() == expected
