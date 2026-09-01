"""Tests for the optional MarkLLM text-watermark harness adapter."""

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

DETECT_SCRIPT = SCRIPTS / "detect_text_watermark.py"

FAKE_TRANSFORMERS = (
    "import sys\n"
    "class _LM:\n"
    "    def to(self, device):\n"
    "        return self\n"
    "\n"
    "class AutoModelForCausalLM:\n"
    "    @staticmethod\n"
    "    def from_pretrained(name, **kwargs):\n"
    "        print('MARKLLM_PRETRAINED_KWARGS=' + repr(kwargs), file=sys.stderr)\n"
    "        return _LM()\n"
    "\n"
    "class AutoTokenizer:\n"
    "    @staticmethod\n"
    "    def from_pretrained(name, **kwargs):\n"
    "        return object()\n"
)

FAKE_TRANSFORMERS_CONFIG = (
    "class TransformersConfig:\n"
    "    def __init__(self, model, tokenizer, vocab_size=None, device='cuda', **kwargs):\n"
    "        self.device = device\n"
    "        self.model = model\n"
    "        self.tokenizer = tokenizer\n"
    "        self.vocab_size = vocab_size\n"
    "        self.gen_kwargs = {}\n"
    "        self.gen_kwargs.update(kwargs)\n"
)

KGW_CONFIG = '{"algorithm_name": "KGW", "z_threshold": 4.0}'
SYNTHID_CONFIG = '{"algorithm_name": "SynthID", "threshold": 0.52, "detector_type": "mean"}'


def _fake_auto_watermark(*, fail_detect: bool = False, fail_generate: bool = False) -> str:
    detect_body = (
        'raise RuntimeError("boom")'
        if fail_detect
        else 'return {"is_watermarked": True, "score": 3.5}'
    )
    gen_body = 'raise RuntimeError("boom")' if fail_generate else "return 'WATERMARKED SAMPLE'"
    return (
        "class _WM:\n"
        "    def __init__(self):\n"
        "        self.config = SimpleNamespace(gen_kwargs={})\n"
        "    def detect_watermark(self, text, return_dict=True):\n"
        f"        {detect_body}\n"
        "    def generate_watermarked_text(self, prompt):\n"
        f"        {gen_body}\n"
        "    def generate_unwatermarked_text(self, prompt):\n"
        "        return 'PLAIN SAMPLE'\n"
        "\n"
        "class AutoWatermark:\n"
        "    @staticmethod\n"
        "    def load(algorithm_name, algorithm_config=None, transformers_config=None):\n"
        "        return _WM()\n"
    )


def _make_fake_upstream(
    tmp_path: Path,
    *,
    with_config: bool = True,
    fail_detect: bool = False,
    fail_generate: bool = False,
    missing_watermark_dir: bool = False,
) -> Path:
    upstream = tmp_path / "MarkLLM"
    config_dir = upstream / "config"
    config_dir.mkdir(parents=True)
    if with_config:
        (config_dir / "KGW.json").write_text(KGW_CONFIG)
        (config_dir / "SynthID.json").write_text(SYNTHID_CONFIG)
    if not missing_watermark_dir:
        watermark = upstream / "watermark"
        watermark.mkdir(parents=True)
        (watermark / "__init__.py").write_text("")
        (watermark / "auto_watermark.py").write_text(
            "from types import SimpleNamespace\n"
            + _fake_auto_watermark(fail_detect=fail_detect, fail_generate=fail_generate)
        )
    utils_dir = upstream / "utils"
    utils_dir.mkdir(parents=True)
    (utils_dir / "__init__.py").write_text("")
    (utils_dir / "transformers_config.py").write_text(FAKE_TRANSFORMERS_CONFIG)
    transformers_dir = upstream / "transformers"
    transformers_dir.mkdir(parents=True)
    (transformers_dir / "__init__.py").write_text(FAKE_TRANSFORMERS)
    return upstream


def _run_adapter(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("MARKLLM_DIR", None)
    return subprocess.run(
        [sys.executable, str(DETECT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_cli_unavailable_without_upstream(tmp_path: Path):
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter("detect", str(f), "--scheme", "kgw")
    assert r.returncode == 3
    assert "MARKLLM_DIR" in (r.stderr or "")


def test_cli_unavailable_incomplete_checkout(tmp_path: Path):
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    empty = tmp_path / "empty"
    empty.mkdir()
    r = _run_adapter("detect", str(f), "--scheme", "kgw", "--upstream-dir", str(empty))
    assert r.returncode == 3

    upstream = _make_fake_upstream(tmp_path, missing_watermark_dir=True)
    r = _run_adapter("detect", str(f), "--scheme", "kgw", "--upstream-dir", str(upstream))
    assert r.returncode == 3


def test_cli_unavailable_missing_config(tmp_path: Path):
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    upstream = _make_fake_upstream(tmp_path, with_config=False)
    r = _run_adapter("detect", str(f), "--scheme", "kgw", "--upstream-dir", str(upstream))
    assert r.returncode == 3
    assert "config" in (r.stderr or "").lower()


def test_cli_unavailable_missing_deps(tmp_path: Path):
    # The watermark module imports a nonexistent dependency -> ImportError ->
    # exit 3 ("dependencies missing") before any model download.
    upstream = tmp_path / "MarkLLM"
    (upstream / "config").mkdir(parents=True)
    (upstream / "config" / "KGW.json").write_text(KGW_CONFIG)
    watermark = upstream / "watermark"
    watermark.mkdir()
    (watermark / "__init__.py").write_text("")
    (watermark / "auto_watermark.py").write_text("import does_not_exist_123\n")
    (upstream / "utils").mkdir()
    (upstream / "utils" / "__init__.py").write_text("")
    (upstream / "utils" / "transformers_config.py").write_text(FAKE_TRANSFORMERS_CONFIG)
    (upstream / "transformers").mkdir()
    (upstream / "transformers" / "__init__.py").write_text(FAKE_TRANSFORMERS)
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter("detect", str(f), "--scheme", "kgw", "--upstream-dir", str(upstream))
    assert r.returncode == 3
    assert "dependencies missing" in (r.stderr or "")


def test_cli_bad_input_missing_file(tmp_path: Path):
    r = _run_adapter("detect", str(tmp_path / "missing.txt"), "--scheme", "kgw")
    assert r.returncode == 2


def test_cli_bad_input_binary(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path)
    png = tmp_path / "img.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nnot really")
    r = _run_adapter("detect", str(png), "--scheme", "kgw", "--upstream-dir", str(upstream))
    assert r.returncode == 2
    assert "refusing" in (r.stderr or "")


def test_cli_bad_scheme(tmp_path: Path):
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter("detect", str(f), "--scheme", "nope")
    assert r.returncode == 2


def test_cli_detect_json_success(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path)
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "kgw",
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["available"] is True
    assert payload["scheme"] == "KGW"
    assert payload["is_watermarked"] is True
    assert payload["score"] == 3.5
    assert payload["threshold"] == 4.0
    assert payload["device"] == "cpu"


def test_cli_detect_synthid_alias(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path)
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "synthid-text",
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["scheme"] == "SynthID"
    assert payload["threshold"] == 0.52


def test_cli_detect_runtime_error(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path, fail_detect=True)
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "kgw",
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 1
    assert "boom" in (r.stderr or "")


def test_cli_detect_offline_flag(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path)
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "kgw",
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
        "--offline",
    )
    assert r.returncode == 0, r.stderr
    assert "local_files_only" in (r.stderr or "")
    assert "True" in (r.stderr or "")


def test_cli_config_too_large(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path)
    big = tmp_path / "huge.json"
    big.write_bytes(b"x" * (1024 * 1024 + 1))
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "kgw",
        "--config",
        str(big),
        "--upstream-dir",
        str(upstream),
    )
    assert r.returncode == 3
    assert "too large" in (r.stderr or "")


def test_cli_watermark_json_success(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("write about capybaras")
    wm_out = tmp_path / "wm.txt"
    uwm_out = tmp_path / "uwm.txt"
    r = _run_adapter(
        "watermark",
        str(prompt),
        "--scheme",
        "kgw",
        "-o",
        str(wm_out),
        "-o2",
        str(uwm_out),
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["available"] is True
    assert wm_out.read_text() == "WATERMARKED SAMPLE"
    assert uwm_out.read_text() == "PLAIN SAMPLE"


def test_cli_watermark_runtime_error(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path, fail_generate=True)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("write about capybaras")
    r = _run_adapter(
        "watermark",
        str(prompt),
        "--scheme",
        "kgw",
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 1
    assert "boom" in (r.stderr or "")


def test_rewrite_markllm_hook_records_before_after(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import rewrite_text

    class _FakeDetector:
        def __init__(self, **kwargs):
            pass

        def detect(self, text):
            return {"available": True, "is_watermarked": text == "ORIG", "score": 3.0}

    monkeypatch.setattr(rewrite_text, "MarkLLMTextDetector", _FakeDetector)
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: "REWRITTEN OUTPUT")
    out, info = rewrite_text.rewrite(
        "ORIG",
        backend="ollama",
        model="m",
        base_url="http://127.0.0.1:11434",
        api_key=None,
        tactic="paraphrase",
        lang="French",
        original_lang="English",
        timeout=10,
        layer_a_after=False,
        temperature=0.9,
        candidates=1,
        markllm_scheme="kgw",
        markllm_dir=str(tmp_path / "x"),
        markllm_model="opt-1.3b",
        markllm_timeout=5,
    )
    assert out == "REWRITTEN OUTPUT"
    mk = info["markllm"]
    assert mk["before"]["is_watermarked"] is True
    assert mk["after"]["is_watermarked"] is False
    assert mk["cleared"] is True
    assert "note" in mk
