"""Tests for the watermark type classifier."""

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


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def test_spatial_features_shape():
    from classify_watermark import extract_spatial_features

    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    feats = extract_spatial_features(image)
    assert feats.ndim == 1
    assert feats.shape[0] > 0


def test_frequency_features_shape():
    from classify_watermark import extract_frequency_features

    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    feats = extract_frequency_features(image)
    assert feats.ndim == 1
    assert feats.shape[0] > 0


# ---------------------------------------------------------------------------
# Classifier interface
# ---------------------------------------------------------------------------


def test_classifier_returns_valid_scheme(tmp_path):
    from classify_watermark import VALID_SCHEMES, WatermarkClassifier

    img = tmp_path / "test.png"
    _write_random_png(img, 256, 256)
    clf = WatermarkClassifier(model_path=None)
    result = clf.classify(str(img))
    assert result.scheme in VALID_SCHEMES
    assert 0.0 <= result.confidence <= 1.0


def test_classifier_routes_stegastamp_to_ctrlregen():
    """Pixel-based watermarks must not be routed to latent-space-only engines."""
    from classify_watermark import LATENT_ONLY_ENGINES, PIXEL_BASED_SCHEMES

    for scheme in PIXEL_BASED_SCHEMES:
        assert scheme not in LATENT_ONLY_ENGINES, (
            f"{scheme} is pixel-based but mapped to a latent-only engine"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_classify_bad_input_exits_2():
    result = _run("classify_watermark.py", "/nonexistent.png")
    assert result.returncode == 2


def test_classify_json_output(tmp_path):
    img = tmp_path / "test.png"
    _write_random_png(img, 128, 128)
    result = _run("classify_watermark.py", str(img), "--json")
    assert result.returncode == 0
    import json

    report = json.loads(result.stdout)
    assert "scheme" in report
    assert "confidence" in report
