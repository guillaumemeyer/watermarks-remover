"""Tests for the Tree-Ring watermark attack engine."""

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


def _inject_ring(image: np.ndarray, radius: int = 20) -> np.ndarray:
    """Inject a synthetic ring pattern for testing detection."""
    h, w = image.shape[:2]
    result = image.astype(np.float64).copy()

    for c in range(result.shape[2] if result.ndim == 3 else 1):
        channel = result[:, :, c] if result.ndim == 3 else result
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)

        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy : h - cy, -cx : w - cx]
        dist = np.sqrt(y * y + x * x)
        ring = np.exp(-0.5 * ((dist - radius) / 2) ** 2)
        fshift += ring * np.max(np.abs(fshift)) * 0.1

        channel_out = np.real(np.fft.ifft2(np.fft.ifftshift(fshift)))
        if result.ndim == 3:
            result[:, :, c] = channel_out
        else:
            result = channel_out

    return np.clip(result, 0, 255).astype(np.uint8)


def test_ring_detection_on_synthetic():
    from treering_attack import detect_ring_pattern

    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    ringed = _inject_ring(image, radius=20)

    detection = detect_ring_pattern(ringed, min_radius=15, max_radius=25)
    assert detection["confidence"] > 0
    assert 15 <= detection["peak_radius"] <= 25


def test_filter_mode_reduces_ring_energy():
    from treering_attack import detect_ring_pattern, notch_filter_attack

    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    ringed = _inject_ring(image, radius=20)

    before = detect_ring_pattern(ringed)
    filtered = notch_filter_attack(ringed, [20])
    after = detect_ring_pattern(filtered)

    assert after["ring_energy"] < before["ring_energy"]


def test_pgd_requires_torch():
    result = _run("clean_treering.py", "--help")
    assert result.returncode == 0


def test_bad_input_exits_2():
    result = _run("clean_treering.py", "/nonexistent.png")
    assert result.returncode == 2


def test_json_output(tmp_path):
    import json

    img = tmp_path / "test.png"
    _write_random_png(img, 128, 128)
    result = _run("clean_treering.py", str(img), "--mode", "filter", "--json")
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert "ring_energy_before" in report
    assert "ring_energy_after" in report
    assert "psnr" in report
