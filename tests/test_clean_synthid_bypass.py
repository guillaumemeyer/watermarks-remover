"""Tests for the SynthID spectral bypass robustness pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_elastic_warp_preserves_dimensions():
    from clean_synthid_bypass import elastic_warp

    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    warped = elastic_warp(image, strength=0.01)
    assert warped.shape == image.shape


def test_channel_squeeze_reduces_range():
    from clean_synthid_bypass import channel_squeeze

    image = np.full((64, 64, 3), 200, dtype=np.uint8)
    squeezed = channel_squeeze(image, factor=0.9)
    assert squeezed.max() <= 200
    assert squeezed.shape == image.shape


def test_vae_roundtrip_changes_image():
    from clean_synthid_bypass import vae_roundtrip

    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = vae_roundtrip(image, device="cpu")
    assert result.shape == image.shape
    assert not np.array_equal(result, image)


def test_synthid_bypass_bad_input_exits_2():
    result = _run("clean_synthid_bypass.py", "/nonexistent.png")
    assert result.returncode == 2
    # Python itself exits 2 when the script is missing; pin the script's own message.
    assert "not a file" in result.stderr


def test_synthid_bypass_default_pipeline_writes_output(tmp_path: Path):
    from PIL import Image

    src = tmp_path / "in.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)).save(src)
    out = tmp_path / "out.png"

    result = _run("clean_synthid_bypass.py", str(src), "-o", str(out), "--json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["pipeline"] == ["warp", "squeeze", "jpeg"]
    assert payload["output_size"] == [128, 96]
    with Image.open(out) as cleaned:
        assert cleaned.size == (128, 96)


@pytest.mark.parametrize("pipeline", ["warp,blur", ""])
def test_synthid_bypass_bad_pipeline_exits_2(tmp_path: Path, pipeline: str):
    from PIL import Image

    src = tmp_path / "in.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(src)

    result = _run("clean_synthid_bypass.py", str(src), "--pipeline", pipeline)
    assert result.returncode == 2
    assert "pipeline" in result.stderr
