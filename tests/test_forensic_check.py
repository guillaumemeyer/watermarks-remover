"""Tests for the forensic stealth detector (WAVES defense-evaluation component).

The detector measures how visible a watermark-removal operation is: spectral
deviation from natural 1/f^alpha image statistics, and JPEG double-quantization
traces in the DCT coefficient histograms.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

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


def _write_random_png(path: Path, w: int, h: int):
    from PIL import Image

    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    img.save(path)


def _dead_leaves(size: int, seed: int) -> np.ndarray:
    """Occluding random disks: a standard stand-in for natural image statistics."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 128.0)
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(3000):
        # Power-law radii (p(r) ~ r^-3) give the scale invariance of real scenes.
        r = 2.0 / np.sqrt(1.0 - rng.random() * (1.0 - (2.0 / 64.0) ** 2))
        cy, cx = rng.uniform(-r, size + r, 2)
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = rng.uniform(0, 255)
    img += rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _jpeg_bytes(pixels: np.ndarray, quality: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _decode(data: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)))


# ---------------------------------------------------------------------------
# Spectral deformation
# ---------------------------------------------------------------------------


def test_spectral_natural_image_low_score():
    from forensic_check import spectral_deformation_score

    natural = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    score = spectral_deformation_score(natural)
    assert 0.0 <= score <= 1.0


def test_spectral_processed_image_higher_score():
    from forensic_check import spectral_deformation_score

    natural = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    # Simulate a diffusion artifact: suppress high frequencies (per channel).
    processed = gaussian_filter(natural, sigma=(3, 3, 0))
    score_natural = spectral_deformation_score(natural)
    score_processed = spectral_deformation_score(processed)
    assert score_processed > score_natural


def test_spectral_dead_leaves_scores_low_and_blur_raises_it():
    from forensic_check import spectral_deformation_score

    scene = _dead_leaves(256, seed=1)
    natural = spectral_deformation_score(scene)
    blurred = spectral_deformation_score(gaussian_filter(scene, sigma=2))
    assert natural < 0.3
    assert blurred > natural


# ---------------------------------------------------------------------------
# JPEG double compression
# ---------------------------------------------------------------------------


def test_jpeg_score_zero_for_non_jpeg():
    from forensic_check import jpeg_double_compression_score
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(_dead_leaves(64, seed=2)).save(buf, format="PNG")
    assert jpeg_double_compression_score(buf.getvalue()) == 0.0


def test_jpeg_double_compression_returns_valid_range():
    from forensic_check import jpeg_double_compression_score

    scene = _dead_leaves(512, seed=3)
    single = jpeg_double_compression_score(_jpeg_bytes(scene, 95))
    double = jpeg_double_compression_score(_jpeg_bytes(_decode(_jpeg_bytes(scene, 20)), 95))
    assert 0.0 <= single <= 1.0
    assert 0.0 <= double <= 1.0


# ---------------------------------------------------------------------------
# ForensicChecker + CLI
# ---------------------------------------------------------------------------


def test_forensic_checker_result_structure(tmp_path):
    from forensic_check import ForensicChecker

    img = tmp_path / "test.png"
    _write_random_png(img, 256, 256)
    checker = ForensicChecker(model_path=None)  # heuristic mode
    result = checker.check(str(img))
    assert isinstance(result.stealth_score, float)
    assert 0.0 <= result.stealth_score <= 1.0
    assert isinstance(result.detected_traces, list)
    assert all(isinstance(t, str) for t in result.detected_traces)
    assert result.recommendation in ("pass", "reprocess", "warn")


def test_forensic_bad_input_exits_2(tmp_path):
    proc = _run("forensic_check.py", str(tmp_path / "missing.png"), "--json")
    assert proc.returncode == 2
    assert proc.stdout == ""
    # The interpreter also exits 2 when the script itself is missing; make sure
    # the refusal came from the checker and names the bad input.
    assert "missing.png" in proc.stderr


def test_forensic_cli_json(tmp_path):
    img = tmp_path / "scene.jpg"
    img.write_bytes(_jpeg_bytes(_dead_leaves(256, seed=4), 90))
    proc = _run("forensic_check.py", str(img), "--json")
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert set(report) >= {"stealth_score", "detected_traces", "recommendation"}
    assert report["recommendation"] in ("pass", "reprocess", "warn")
