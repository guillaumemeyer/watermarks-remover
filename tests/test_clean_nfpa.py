"""Tests for the NFPA next-frame prediction watermark remover."""

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


def test_flow_matrix_shape():
    torch = pytest.importorskip("torch")
    from nfpa_engine import build_flow_matrix

    image = torch.randn(1, 3, 512, 512)
    flow = build_flow_matrix(image, strength=0.05)
    assert flow.shape == (1, 2, 512, 512)


def test_flow_matrix_magnitude_bounded():
    torch = pytest.importorskip("torch")
    from nfpa_engine import build_flow_matrix

    image = torch.randn(1, 3, 512, 512)
    flow = build_flow_matrix(image, strength=0.05)
    assert flow.abs().max() < 10.0


def test_flow_matrix_flat_image_has_no_flow():
    torch = pytest.importorskip("torch")
    from nfpa_engine import build_flow_matrix

    flow = build_flow_matrix(torch.full((1, 3, 64, 64), 0.5), strength=0.05)
    assert torch.isfinite(flow).all()
    assert flow.abs().max() == 0


def test_warp_preserves_statistics():
    torch = pytest.importorskip("torch")
    from nfpa_engine import build_flow_matrix, warp_noise

    noise = torch.randn(1, 4, 64, 64)
    flow = build_flow_matrix(torch.randn(1, 3, 512, 512), strength=0.02)
    warped = warp_noise(noise, flow)
    assert warped.shape == noise.shape
    assert abs(warped.mean().item()) < 0.5
    assert abs(warped.std().item() - 1.0) < 0.5


def test_warp_zero_flow_is_identity():
    torch = pytest.importorskip("torch")
    from nfpa_engine import warp_noise

    noise = torch.randn(1, 4, 64, 64)
    warped = warp_noise(noise, torch.zeros(1, 2, 512, 512))
    assert torch.allclose(warped, noise, atol=1e-4)


def test_warp_scales_image_flow_to_noise_resolution():
    torch = pytest.importorskip("torch")
    from nfpa_engine import warp_noise

    noise = torch.randn(1, 4, 64, 64)
    flow = torch.zeros(1, 2, 512, 512)
    flow[:, 0] = 8.0  # 8 image pixels == 1 latent pixel at 1/8 resolution
    warped = warp_noise(noise, flow)
    shifted = torch.corrcoef(torch.stack([warped[..., :-1].flatten(), noise[..., 1:].flatten()]))
    assert shifted[0, 1] > 0.99


def test_nfpa_bad_input_exits_2():
    result = _run("clean_nfpa.py", "/nonexistent.png")
    assert result.returncode == 2
    assert "not a file" in result.stderr


def test_nfpa_bad_flow_strength_exits_2(tmp_path: Path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    result = _run("clean_nfpa.py", str(img), "--flow-strength", "0")
    assert result.returncode == 2
    assert "flow-strength" in result.stderr
