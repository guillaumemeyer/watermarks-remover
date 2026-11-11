"""A leading UTF-8 BOM is an encoding signature, not a Layer A carrier (#442)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import clean_file
from audit_lib import is_actionable, scan_file
from container_meta import clean_container, inspect_container

BOM = b"\xef\xbb\xbf"
EMBEDDED = 'Write-Host "a\ufeffb"\n'.encode("utf-8")


def test_leading_utf8_bom_is_not_actionable(tmp_path: Path) -> None:
    ps1 = tmp_path / "bom.ps1"
    md = tmp_path / "bom.md"
    ps1.write_bytes(BOM + 'Write-Host "Größe"\n'.encode())
    md.write_bytes(BOM + b"# Title\n")

    for path in (ps1, md):
        item = scan_file(path)
        assert not is_actionable(item), path.name
        assert item["suspicious_total"] == 0
        assert item["findings"] == []


def test_embedded_feff_is_still_reported(tmp_path: Path) -> None:
    path = tmp_path / "bom_inner.ps1"
    path.write_bytes(BOM + EMBEDDED)
    item = scan_file(path)
    assert is_actionable(item)
    assert item["suspicious_total"] == 1
    assert any("U+FEFF" in f and "x1" in f for f in item["findings"])


def test_clean_file_keeps_a_leading_bom_and_strips_an_embedded_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "bom_inner.ps1"
    dest = tmp_path / "out.ps1"
    src.write_bytes(BOM + EMBEDDED)
    monkeypatch.setattr(sys, "argv", ["clean_file.py", str(src), "-o", str(dest)])
    assert clean_file.main() == 0
    out = dest.read_bytes()
    assert out.startswith(BOM)
    assert out.count(BOM) == 1
    assert b"ab" in out or b"ab" in out.replace(BOM, b"")
    # The inner U+FEFF is gone; the leading signature remains.
    body = out[len(BOM) :]
    assert b"\xef\xbb\xbf" not in body
    assert not is_actionable(scan_file(dest))


def test_clean_file_does_not_add_a_bom_when_none_was_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "plain.ps1"
    dest = tmp_path / "out.ps1"
    src.write_bytes(b'Write-Host "ok"\n')
    monkeypatch.setattr(sys, "argv", ["clean_file.py", str(src), "-o", str(dest)])
    assert clean_file.main() == 0
    assert not dest.read_bytes().startswith(BOM)


def test_markdown_inspect_and_clean_preserve_leading_bom(tmp_path: Path) -> None:
    src = tmp_path / "bom.md"
    dest = tmp_path / "out.md"
    src.write_bytes(BOM + b"# Title\n")
    report = inspect_container(src)
    assert report.layer_a_total == 0
    result = clean_container(src, dest, fmt="markdown")
    assert dest.read_bytes().startswith(BOM)
    assert dest.read_bytes()[len(BOM) :] == b"# Title\n"
    assert result["format"] == "markdown"
