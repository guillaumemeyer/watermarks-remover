"""--json must not suppress the residual-signal exit code (was: always 0)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import clean_file
import clean_image


def _container_result(dest: Path, residual: bool, degraded: bool = False) -> dict:
    return {
        "input": "x.md",
        "output": str(dest),
        "format": "markdown",
        "actions": ["clean"],
        "bytes_in": 1,
        "bytes_out": 1,
        "changed": False,
        "still_has_c2pa": residual,
        "still_has_ai_metadata": residual,
        "post_findings": ["marker:c2pa"] if residual else [],
        "meta": {"degraded": degraded},
    }


def _run_clean_file(monkeypatch, tmp_path, *, json_flag, residual, degraded=False):
    src = tmp_path / "x.md"
    src.write_text("---\ngenerator: Claude\n---\nhi\n", encoding="utf-8")
    dest = tmp_path / "x.cleaned.md"
    monkeypatch.setattr(
        clean_file, "clean_container", lambda *a, **k: _container_result(dest, residual, degraded)
    )
    argv = ["clean_file.py", str(src), "-o", str(dest)]
    if json_flag:
        argv.append("--json")
    monkeypatch.setattr(sys, "argv", argv)
    return clean_file.main()


def test_clean_file_json_and_human_agree_on_residual(monkeypatch, tmp_path):
    # The bug: --json returned 0 while human mode returned 1.
    assert _run_clean_file(monkeypatch, tmp_path, json_flag=False, residual=True) == 1
    assert _run_clean_file(monkeypatch, tmp_path, json_flag=True, residual=True) == 1


def test_clean_file_json_and_human_agree_on_clean(monkeypatch, tmp_path):
    assert _run_clean_file(monkeypatch, tmp_path, json_flag=False, residual=False) == 0
    assert _run_clean_file(monkeypatch, tmp_path, json_flag=True, residual=False) == 0


def test_clean_file_degraded_pdf_is_not_a_failure_in_either_mode(monkeypatch, tmp_path):
    assert (
        _run_clean_file(monkeypatch, tmp_path, json_flag=False, residual=True, degraded=True) == 0
    )
    assert _run_clean_file(monkeypatch, tmp_path, json_flag=True, residual=True, degraded=True) == 0


def _image_result(dest: Path, residual: bool) -> dict:
    return {
        "input": "x.png",
        "output": str(dest),
        "actions": ["strip"],
        "bytes_in": 1,
        "bytes_out": 1,
        "changed": False,
        "still_has_c2pa": residual,
        "still_has_ai_metadata": residual,
        "post_findings": ["byte-scan c2pa"] if residual else [],
        "synthid_before": None,
        "synthid_after": None,
        "pixel_removal": None,
    }


def _run_clean_image(monkeypatch, tmp_path, *, json_flag, residual):
    src = tmp_path / "x.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")
    dest = tmp_path / "x.cleaned.png"
    monkeypatch.setattr(clean_image, "clean_image", lambda *a, **k: _image_result(dest, residual))
    argv = ["clean_image.py", str(src), "-o", str(dest)]
    if json_flag:
        argv.append("--json")
    monkeypatch.setattr(sys, "argv", argv)
    return clean_image.main()


def test_clean_image_json_and_human_agree_on_residual(monkeypatch, tmp_path):
    assert _run_clean_image(monkeypatch, tmp_path, json_flag=False, residual=True) == 1
    assert _run_clean_image(monkeypatch, tmp_path, json_flag=True, residual=True) == 1
    assert _run_clean_image(monkeypatch, tmp_path, json_flag=True, residual=False) == 0
