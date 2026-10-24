"""Tests for the FMDiffWA frequency-domain watermark remover."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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


def test_fwm_suppresses_high_frequency():
    torch = pytest.importorskip("torch")
    from fmdiffwa_engine import FrequencyWatermarkModulator

    mod = FrequencyWatermarkModulator(patch_size=256)
    image = torch.randn(1, 3, 256, 256)
    modulated = mod.modulate(image)
    assert modulated.shape == image.shape
    fft_orig = torch.fft.fft2(image).abs().mean()
    fft_mod = torch.fft.fft2(modulated).abs().mean()
    assert fft_mod <= fft_orig * 1.1


def test_fwm_attenuates_injected_periodic_pattern():
    torch = pytest.importorskip("torch")
    from fmdiffwa_engine import FrequencyWatermarkModulator

    ys, xs = torch.meshgrid(torch.arange(256.0), torch.arange(256.0), indexing="ij")
    # Smooth content plus faint broadband noise, then a periodic mark at 40 cycles.
    content = 0.5 + 0.2 * torch.sin(2 * torch.pi * xs / 256) * torch.cos(2 * torch.pi * ys / 256)
    content = content + 0.01 * torch.randn(256, 256, generator=torch.Generator().manual_seed(0))
    mark = 0.02 * torch.sin(2 * torch.pi * 40 * xs / 256)
    image = (content + mark).expand(1, 3, 256, 256).contiguous()

    modulated = FrequencyWatermarkModulator(patch_size=256).modulate(image)

    def mark_energy(t):
        return torch.fft.rfft2(t)[..., 0, 40].abs().mean()

    assert mark_energy(modulated) < 0.5 * mark_energy(image)
    # Content outside the marked band survives: the residual is close to the mark itself.
    assert (modulated - image + mark).abs().mean() < 0.01


def test_fmdiffwa_handles_non_square_images():
    torch = pytest.importorskip("torch")
    from fmdiffwa_engine import FrequencyWatermarkModulator

    mod = FrequencyWatermarkModulator(patch_size=256)
    patches = mod._split_patches(torch.randn(1, 3, 512, 768), patch_size=256)
    assert len(patches) == 12  # 3x4 grid with guaranteed overlap


def test_fmdiffwa_overlapping_patches_merge_back_exactly():
    torch = pytest.importorskip("torch")
    from fmdiffwa_engine import FrequencyWatermarkModulator, merge_patches

    image = torch.rand(1, 3, 300, 520)
    patches = FrequencyWatermarkModulator(patch_size=256)._split_patches(image, patch_size=256)
    assert len(patches) == 6  # 300 -> 2 rows, 520 -> 3 columns, all overlapping
    assert all(p.shape == (1, 3, 256, 256) for p in patches)
    merged = merge_patches(patches, image.shape, patch_size=256)
    assert torch.allclose(merged, image, atol=1e-6)


def test_fmdiffwa_frequency_only_remove_keeps_size():
    pytest.importorskip("torch")
    from fmdiffwa_engine import FMDiffWAEngine
    from PIL import Image

    image = Image.new("RGB", (300, 200), (120, 80, 200))
    result = FMDiffWAEngine(patch_size=128, device="cpu").remove(image, intensity=0.0)
    assert result.size == image.size
    assert result.getpixel((150, 100)) == (120, 80, 200)


def test_fmdiffwa_bad_input_exits_2():
    result = _run("clean_fmdiffwa.py", "/nonexistent.png")
    assert result.returncode == 2
    assert "not a file" in result.stderr


def test_fmdiffwa_bad_intensity_exits_2(tmp_path: Path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    result = _run("clean_fmdiffwa.py", str(img), "--intensity", "1.5")
    assert result.returncode == 2
    assert "intensity" in result.stderr


def test_fmdiffwa_engine_available():
    pytest.importorskip("torch")
    from fmdiffwa_engine import FMDiffWAEngine

    assert FMDiffWAEngine.available() is True
