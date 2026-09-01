"""Tests for the optional CtrlRegen pixel-watermark remover adapter."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import image_meta
from image_meta import run_ctrlregen_clean

CLEAN_SCRIPT = SCRIPTS / "clean_ctrlregen.py"

FAKE_PIL = """\
class Image:
    def __init__(self, size=(10, 20)):
        self.size = size
    @staticmethod
    def open(path):
        return Image()
    def convert(self, mode):
        return self
    def load(self):
        return self
    def save(self, fp, format=None, **kwargs):
        fp.write(b"FAKEIMAGE")
"""


def _fake_engine(fail_run: bool) -> str:
    run_body = 'raise RuntimeError("model missing")' if fail_run else "return image"
    return (
        "class CtrlRegenEngine:\n"
        "    def __init__(self, **kwargs):\n"
        "        pass\n"
        "    def run(self, image, strength=0.5, num_inference_steps=50, "
        "guidance_scale=2.0, seed=None):\n"
        f"        {run_body}\n"
        "\n"
        "def is_ctrlregen_available():\n"
        "    return True\n"
    )


def _make_fake_upstream(tmp_path: Path, *, fail_run: bool = False) -> Path:
    upstream = tmp_path / "noai-watermark"
    ctrlregen = upstream / "src" / "ctrlregen"
    pil = upstream / "src" / "PIL"
    ctrlregen.mkdir(parents=True)
    pil.mkdir(parents=True)
    (ctrlregen / "__init__.py").write_text("")
    (pil / "__init__.py").write_text(FAKE_PIL)
    (ctrlregen / "engine.py").write_text(_fake_engine(fail_run))
    return upstream


def _run_adapter(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("NOAI_WATERMARK_DIR", None)
    return subprocess.run(
        [sys.executable, str(CLEAN_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _minimal_png() -> bytes:
    def chunk(ctype: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(ctype)
        crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00")
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def test_cli_unavailable_without_upstream(tmp_path: Path):
    dummy = tmp_path / "img.png"
    dummy.write_bytes(b"not really an image")
    r = _run_adapter(str(dummy))
    assert r.returncode == 3
    assert "NOAI_WATERMARK_DIR" in (r.stderr or "")


def test_cli_bad_input_missing_file(tmp_path: Path):
    r = _run_adapter(str(tmp_path / "missing.png"))
    assert r.returncode == 2


def test_cli_bad_intensity(tmp_path: Path):
    dummy = tmp_path / "img.png"
    dummy.write_bytes(b"x")
    r = _run_adapter(str(dummy), "--intensity", "0")
    assert r.returncode == 2


def test_cli_unavailable_missing_src_dir(tmp_path: Path):
    dummy = tmp_path / "img.png"
    dummy.write_bytes(b"x")
    empty = tmp_path / "empty"
    empty.mkdir()
    r = _run_adapter(str(dummy), "--upstream-dir", str(empty))
    assert r.returncode == 3


def test_cli_json_success(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path)
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    out = tmp_path / "out.png"
    r = _run_adapter(
        str(img),
        "-o",
        str(out),
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["available"] is True
    assert payload["output"] == str(out)
    assert out.read_bytes() == b"FAKEIMAGE"


def test_cli_runtime_error(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path, fail_run=True)
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    r = _run_adapter(
        str(img),
        "-o",
        str(tmp_path / "out.png"),
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 1
    assert "model missing" in (r.stderr or "")


def test_cli_refuses_symlink_output(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path)
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    victim = tmp_path / "victim"
    victim.write_bytes(b"original")
    out = tmp_path / "out.png"
    try:
        out.symlink_to(victim)
    except OSError:
        pytest.skip("symlinks unavailable")
    r = _run_adapter(
        str(img),
        "-o",
        str(out),
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 1
    assert victim.read_bytes() == b"original"


def test_run_ctrlregen_clean_unconfigured_returns_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("NOAI_WATERMARK_DIR", raising=False)
    result = run_ctrlregen_clean(Path("x.png"), Path("y.png"))
    assert result["available"] is False
    assert "NOAI_WATERMARK_DIR" in result["error"]


def test_run_ctrlregen_clean_success_parses_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    payload = {"available": True, "output": str(tmp_path / "y.png"), "device": "cpu"}
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(image_meta.subprocess, "run", fake_run)
    result = run_ctrlregen_clean(
        Path("x.png"),
        Path("y.png"),
        upstream_dir=str(upstream),
        intensity=0.3,
        steps=40,
        device="cpu",
        seed=7,
        timeout=99,
    )

    assert result["available"] is True
    assert result["device"] == "cpu"
    assert "--json" in captured["cmd"]
    assert captured["kwargs"]["timeout"] == 99
    if os.name == "posix":
        assert captured["kwargs"]["preexec_fn"] is image_meta.ctrlregen_subprocess_preexec_fn


def test_run_ctrlregen_clean_unavailable_exit3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    upstream = tmp_path / "upstream"
    upstream.mkdir()

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=3, stdout="", stderr="deps missing")

    monkeypatch.setattr(image_meta.subprocess, "run", fake_run)
    result = run_ctrlregen_clean(Path("x.png"), Path("y.png"), upstream_dir=str(upstream))
    assert result["available"] is False
    assert "deps missing" in result["error"]


def test_run_ctrlregen_clean_prefers_venv_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    upstream = tmp_path / "upstream"
    if os.name == "nt":
        venv_python = upstream / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = upstream / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(image_meta.subprocess, "run", fake_run)
    run_ctrlregen_clean(Path("x.png"), Path("y.png"), upstream_dir=str(upstream))
    assert captured["cmd"][0] == str(venv_python)


def test_clean_image_ctrlregen_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NOAI_WATERMARK_DIR", raising=False)
    monkeypatch.delenv("REVERSE_SYNTHID_DIR", raising=False)
    src = tmp_path / "t.png"
    src.write_bytes(_minimal_png())
    dest = tmp_path / "t.cleaned.png"
    captured: dict = {}

    def fake_rc(path, output, **kwargs):
        captured["path"] = path
        captured["output"] = output
        captured["kwargs"] = kwargs
        return {"available": True, "device": "cpu"}

    monkeypatch.setattr(image_meta, "run_ctrlregen_clean", fake_rc)
    result = image_meta.clean_image(
        src,
        dest,
        remove_pixel="ctrlregen",
        ctrlregen_dir=str(tmp_path / "upstream"),
        ctrlregen_intensity=0.3,
    )

    assert result["pixel_removal"]["available"] is True
    assert any("CtrlRegen pixel removal" in a for a in result["actions"])
    assert captured["path"] == dest
    assert captured["output"] == dest
    assert captured["kwargs"]["intensity"] == 0.3
