"""Tests for the re-watermarking attack engine."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

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


def _write_random_png(path: Path, w: int, h: int) -> None:
    from PIL import Image

    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    img.save(path)


def test_noise_ring_injects_frequency_pattern():
    from rewatermark_engine import inject_noise_ring

    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = inject_noise_ring(image, strength=0.5, radius=20)

    gray_before = np.mean(image.astype(np.float64), axis=2)
    gray_after = np.mean(result.astype(np.float64), axis=2)

    mag_before = np.abs(np.fft.fftshift(np.fft.fft2(gray_before)))
    mag_after = np.abs(np.fft.fftshift(np.fft.fft2(gray_after)))

    h, w = mag_before.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy : h - cy, -cx : w - cx]
    dist = np.sqrt(y * y + x * x)
    ring = (dist >= 18) & (dist <= 22)

    energy_before = float(np.mean(mag_before[ring]))
    energy_after = float(np.mean(mag_after[ring]))
    assert energy_after > energy_before


def test_fake_stega_modifies_lsb():
    from rewatermark_engine import inject_fake_stega

    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    result = inject_fake_stega(image, strength=0.3)

    lsb_original = image & 1
    lsb_result = result & 1
    changed = np.sum(lsb_original != lsb_result)
    assert changed > 0


def test_spectral_shift_adds_energy():
    from rewatermark_engine import inject_spectral_shift

    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = inject_spectral_shift(image, strength=0.5, target_freqs=[30])

    gray_before = np.mean(image.astype(np.float64), axis=2)
    gray_after = np.mean(result.astype(np.float64), axis=2)

    mag_before = np.abs(np.fft.fftshift(np.fft.fft2(gray_before)))
    mag_after = np.abs(np.fft.fftshift(np.fft.fft2(gray_after)))

    h, w = mag_before.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy : h - cy, -cx : w - cx]
    dist = np.sqrt(y * y + x * x)
    band = (dist >= 28) & (dist <= 32)

    assert float(np.mean(mag_after[band])) > float(np.mean(mag_before[band]))


def test_bad_input_exits_2():
    result = _run("clean_rewatermark.py", "/nonexistent.png")
    assert result.returncode == 2


def test_json_output(tmp_path):
    import json

    img = tmp_path / "test.png"
    _write_random_png(img, 128, 128)
    result = _run("clean_rewatermark.py", str(img), "--json")
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert "replacement_scheme" in report
    assert "injection_strength" in report
