"""Tests for the optional MarkDiffusion image-watermark harness adapter."""

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
from image_meta import run_markdiffusion_purify

HARNESS_SCRIPT = SCRIPTS / "markdiffusion_harness.py"

FAKE_PIL = """\
class Image:
    def __init__(self, mode="RGB", size=(10, 20)):
        self.mode = mode
        self.size = size
    @staticmethod
    def open(path):
        return Image()
    @staticmethod
    def new(mode, size):
        return Image(mode, size)
    def convert(self, mode):
        return self
    def save(self, fp, format=None, **kwargs):
        fp.write(b"FAKEPNG")
"""

FAKE_MARKDIFFUSION = """\
class DiffusionConfig:
    def __init__(self, scheduler=None, pipe=None, device="cpu", **kwargs):
        self.device = device
        self.image_size = kwargs.get("image_size", (512, 512))
        self.num_inference_steps = kwargs.get("num_inference_steps", 50)

class AutoWatermark:
    def __init__(self):
        self.config = SimpleNamespace(config_dict={"threshold": 50})
    @staticmethod
    def load(scheme, algorithm_config=None, diffusion_config=None, **kwargs):
        return AutoWatermark()
    def generate_watermarked_media(self, prompt, **kwargs):
        from PIL import Image
        return Image.new("RGB", (16, 16))
    def generate_unwatermarked_media(self, prompt):
        from PIL import Image
        return Image.new("RGB", (16, 16))
    def detect_watermark_in_media(self, image, prompt="", **kwargs):
        if kwargs.get("detector_type") == "p_value":
            return {"is_watermarked": True, "p_value": 0.0001}
        return {"is_watermarked": False, "l1_distance": 83.5}

class DiffusionPurification:
    def __init__(self, diffusion_config, purification_strength=0.3, prompt="", purifier_pipe=None):
        self.strength = purification_strength
    def edit(self, image, prompt=None):
        from PIL import Image
        return Image.new("RGB", (16, 16))

from types import SimpleNamespace
"""


def _make_fake_upstream(tmp_path: Path) -> Path:
    upstream = tmp_path / "markdiffusion"
    markdiffusion_pkg = upstream / "markdiffusion"
    for sub in (
        "",
        "utils",
        "watermark",
        "evaluation",
        "evaluation/tools",
    ):
        (markdiffusion_pkg / sub).mkdir(parents=True, exist_ok=True)
        (markdiffusion_pkg / sub / "__init__.py").write_text("")
    (upstream / "PIL").mkdir(parents=True)
    (upstream / "PIL" / "__init__.py").write_text(FAKE_PIL)
    (upstream / "markdiffusion" / "_fake.py").write_text(FAKE_MARKDIFFUSION)
    (markdiffusion_pkg / "watermark" / "__init__.py").write_text(
        "from markdiffusion._fake import AutoWatermark\n"
    )
    (markdiffusion_pkg / "utils" / "__init__.py").write_text(
        "from markdiffusion._fake import DiffusionConfig\n"
    )
    (markdiffusion_pkg / "evaluation" / "tools" / "image_editor.py").write_text(
        "from markdiffusion._fake import DiffusionPurification\n"
    )
    (markdiffusion_pkg / "evaluation" / "tools" / "__init__.py").write_text(
        "from markdiffusion._fake import DiffusionPurification\n"
    )
    return upstream


def _minimal_png() -> bytes:
    def chunk(ctype: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(ctype)
        crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00")
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _import_harness():
    import importlib.util

    spec = importlib.util.spec_from_file_location("mdh", str(HARNESS_SCRIPT))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- CLI exit-code paths that need no torch/diffusers -----------------------


def test_cli_unavailable_without_upstream(tmp_path: Path):
    env = os.environ.copy()
    env.pop("MARKDIFFUSION_DIR", None)
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    r = subprocess.run(
        [sys.executable, str(HARNESS_SCRIPT), "detect", str(img), "--json"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert r.returncode == 3
    assert "markdiffusion not importable" in (r.stderr or "")


def test_cli_bad_input_missing_file(tmp_path: Path):
    r = subprocess.run(
        [sys.executable, str(HARNESS_SCRIPT), "detect", str(tmp_path / "missing.png")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 2


def test_cli_bad_scheme(tmp_path: Path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    r = subprocess.run(
        [sys.executable, str(HARNESS_SCRIPT), "detect", str(img), "--scheme", "bogus"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 2
    assert "unknown scheme" in (r.stderr or "")


def test_cli_missing_purify_output(tmp_path: Path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    r = subprocess.run(
        [sys.executable, str(HARNESS_SCRIPT), "purify", str(img)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 2


# ---- Direct function tests (monkeypatched model loading) --------------------


def _build_args(mod, cmd: str, **overrides) -> object:
    import argparse

    args = argparse.Namespace()
    args.cmd = cmd
    args.upstream_dir = None
    args.model = "fake/model"
    args.device = "cpu"
    args.offline = False
    args.force_text = False
    args.config = None
    args.size = 512
    args.steps = 50
    args.guidance = 7.5
    args.seed = None
    args.json = True
    args.scheme = "tr"
    args.prompt = None
    args.detector_type = None
    args.path = None
    args.output = None
    args.watermarked_output = None
    args.unwatermarked_output = None
    args.purification_intensity = 0.3
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def test_cmd_detect_with_fake_upstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    mod = _import_harness()
    upstream = _make_fake_upstream(tmp_path)
    img = tmp_path / "img.png"
    img.write_bytes(_minimal_png())
    monkeypatch.setattr(
        mod,
        "_load_diffusion",
        lambda model, device, offline, size: (object(), object()),
    )
    args = _build_args(mod, "detect", path=str(img))
    rc = mod._cmd_detect(args, upstream, "TR")
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["available"] is True
    assert payload["scheme"] == "TR"
    assert payload["is_watermarked"] is False
    assert payload["score"] == 83.5
    assert payload["threshold"] == 50


def test_cmd_detect_p_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _import_harness()
    upstream = _make_fake_upstream(tmp_path)
    img = tmp_path / "img.png"
    img.write_bytes(_minimal_png())
    monkeypatch.setattr(
        mod,
        "_load_diffusion",
        lambda model, device, offline, size: (object(), object()),
    )
    args = _build_args(mod, "detect", path=str(img), detector_type="p_value")
    assert mod._cmd_detect(args, upstream, "tr") == 0


def test_cmd_watermark_with_fake_upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _import_harness()
    upstream = _make_fake_upstream(tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("a red fox", "utf-8")
    out = tmp_path / "wm.png"
    out2 = tmp_path / "plain.png"
    monkeypatch.setattr(
        mod,
        "_load_diffusion",
        lambda model, device, offline, size: (object(), object()),
    )
    args = _build_args(
        mod,
        "watermark",
        prompt=str(prompt),
        watermarked_output=str(out),
        unwatermarked_output=str(out2),
        scheme="ringid",
    )
    assert mod._cmd_watermark(args, upstream, "RI") == 0
    assert out.read_bytes() == b"FAKEPNG"
    assert out2.read_bytes() == b"FAKEPNG"


def test_cmd_purify_with_fake_upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _import_harness()
    upstream = _make_fake_upstream(tmp_path)
    img = tmp_path / "img.png"
    img.write_bytes(_minimal_png())
    out = tmp_path / "purified.png"
    monkeypatch.setattr(
        mod,
        "_load_diffusion",
        lambda model, device, offline, size: (object(), object()),
    )
    args = _build_args(mod, "purify", path=str(img), output=str(out))
    assert mod._cmd_purify(args, upstream) == 0
    assert out.read_bytes() == b"FAKEPNG"


def test_cmd_purify_runtime_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _import_harness()
    upstream = _make_fake_upstream(tmp_path)
    img = tmp_path / "img.png"
    img.write_bytes(_minimal_png())

    def _boom(*a, **k):
        raise RuntimeError("model missing")

    monkeypatch.setattr(mod, "_load_diffusion", _boom)
    args = _build_args(mod, "purify", path=str(img), output=str(tmp_path / "out.png"))
    assert mod._cmd_purify(args, upstream) == 1


# ---- run_markdiffusion_purify wiring ---------------------------------------


def test_run_purify_unconfigured_returns_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MARKDIFFUSION_DIR", raising=False)
    result = run_markdiffusion_purify(Path("x.png"), Path("y.png"))
    assert result["available"] is False


def test_run_purify_success_parses_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    payload = {"available": True, "output": str(tmp_path / "y.png"), "device": "cpu"}
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(image_meta.subprocess, "run", fake_run)
    result = run_markdiffusion_purify(
        Path("x.png"),
        Path("y.png"),
        upstream_dir=str(upstream),
        intensity=0.4,
        size=512,
        steps=40,
        device="cpu",
        timeout=99,
    )

    assert result["available"] is True
    assert result["device"] == "cpu"
    cmd = captured["cmd"]
    assert "purify" in cmd
    assert "--purification-intensity" in cmd and "0.4" in cmd
    assert "--upstream-dir" in cmd and str(upstream) in cmd
    assert captured["kwargs"]["timeout"] == 99
    if os.name == "posix":
        assert captured["kwargs"]["preexec_fn"] is image_meta.ctrlregen_subprocess_preexec_fn


def test_run_purify_runtime_error_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    upstream = tmp_path / "upstream"
    upstream.mkdir()

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(image_meta.subprocess, "run", fake_run)
    result = run_markdiffusion_purify(Path("x.png"), Path("y.png"), upstream_dir=str(upstream))
    assert result["available"] is False
    assert "boom" in result["error"]


def test_run_purify_prefers_venv_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
    run_markdiffusion_purify(Path("x.png"), Path("y.png"), upstream_dir=str(upstream))
    assert captured["cmd"][0] == str(venv_python)


def test_clean_image_diffusion_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MARKDIFFUSION_DIR", raising=False)
    monkeypatch.delenv("REVERSE_SYNTHID_DIR", raising=False)
    src = tmp_path / "t.png"
    src.write_bytes(_minimal_png())
    dest = tmp_path / "t.cleaned.png"
    captured: dict = {}

    def fake_purify(path, output, **kwargs):
        captured["path"] = path
        captured["output"] = output
        captured["kwargs"] = kwargs
        return {"available": True, "device": "cpu"}

    monkeypatch.setattr(image_meta, "run_markdiffusion_purify", fake_purify)
    result = image_meta.clean_image(
        src,
        dest,
        remove_pixel="diffusion",
        markdiffusion_dir=str(tmp_path / "upstream"),
        markdiffusion_intensity=0.3,
    )

    assert result["pixel_removal"]["available"] is True
    assert any("DiffusionPurification pixel removal" in a for a in result["actions"])
    assert captured["path"] == dest
    assert captured["output"] == dest
    assert captured["kwargs"]["intensity"] == 0.3
