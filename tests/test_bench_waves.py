"""Tests for the WAVES benchmark adapter."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(BENCHMARKS))
sys.path.insert(0, str(SCRIPTS))


def test_waves_result_has_required_metrics():
    from bench_waves import WavesResult

    r = WavesResult(
        tpr=0.05, tpr_before=0.95, psnr=32.0, ssim=0.96, lpips=0.04, fid=15.0, n_images=100
    )
    assert 0.0 <= r.tpr <= 1.0
    assert 0.0 <= r.tpr_before <= 1.0
    assert r.psnr > 0
    assert r.n_images == 100


def test_waves_result_accepts_none_lpips_fid():
    from bench_waves import WavesResult

    r = WavesResult(tpr=0.1, tpr_before=0.8, psnr=30.0, ssim=0.9, lpips=None, fid=None, n_images=5)
    assert r.lpips is None
    assert r.fid is None
    assert r.tpr_before == 0.8
    assert r.n_images == 5


def test_detect_watermark_fallback_classifier(tmp_path: Path):
    """Built-in classifier returns False for a plain noise image."""
    import numpy as np
    from bench_waves import _detect_watermark
    from PIL import Image

    img = tmp_path / "noise.png"
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(img))

    result = _detect_watermark(img, decoder_cmd=None)
    assert isinstance(result, bool)


def test_compute_psnr_identical():
    import numpy as np
    from bench_waves import compute_psnr

    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    assert compute_psnr(img, img) >= 99.0


def test_compute_psnr_different():
    import numpy as np
    from bench_waves import compute_psnr

    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = np.full((64, 64, 3), 10, dtype=np.uint8)
    psnr = compute_psnr(a, b)
    assert 20 < psnr < 40


def test_compute_ssim_identical():
    import numpy as np
    from bench_waves import compute_ssim

    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    assert compute_ssim(img, img) > 0.99


def test_compute_ssim_different():
    import numpy as np
    from bench_waves import compute_ssim

    a = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    b = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    ssim = compute_ssim(a, b)
    assert -1.0 <= ssim <= 1.0
