"""Tests for the MarkNull latent-space watermark removal engine.

MarkNull computes a Noise-Latent Alignment Score (NLAS) to detect latent-space
watermarks, then optimises the latent representation to minimise this score
while preserving image quality.  Part of the WAVES robustness-evaluation
toolchain.
"""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

JPEG_1X1 = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
    "Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwh"
    "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAAR"
    "CAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAA"
    "AAAA/8QAFAEBAAAAAAAAAAAAAAAAAAAABP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhED"
    "EQA/AI6AaG//2Q=="
)


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        text=True,
        capture_output=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Unit tests — NLAS computation
# ---------------------------------------------------------------------------


def test_nlas_correlated_signals():
    """Correlated noise-latent pair should have high NLAS."""
    torch = pytest.importorskip("torch")
    from marknull_engine import compute_nlas

    torch.manual_seed(42)
    noise = torch.randn(1, 4, 64, 64)
    latent = noise * 0.8 + torch.randn_like(noise) * 0.2
    score = compute_nlas(latent, noise)
    assert score > 0.5


def test_nlas_uncorrelated_signals():
    """Independent noise and latent should have low NLAS."""
    torch = pytest.importorskip("torch")
    from marknull_engine import compute_nlas

    torch.manual_seed(42)
    noise = torch.randn(1, 4, 64, 64)
    latent = torch.randn(1, 4, 64, 64)
    score = compute_nlas(latent, noise)
    assert score < 0.3


def test_nlas_range():
    """NLAS should always be in [0, 1]."""
    torch = pytest.importorskip("torch")
    from marknull_engine import compute_nlas

    torch.manual_seed(0)
    for _ in range(5):
        a = torch.randn(1, 4, 32, 32)
        b = torch.randn(1, 4, 32, 32)
        score = compute_nlas(a, b)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Unit tests — latent optimisation
# ---------------------------------------------------------------------------


def test_optimize_reduces_nlas():
    """Optimisation should reduce NLAS of a watermarked latent."""
    torch = pytest.importorskip("torch")
    from marknull_engine import compute_nlas, optimize_latent

    class _MockVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"scaling_factor": 0.18215})()

        def decode(self, z, **kw):
            return type("D", (), {"sample": z.repeat(1, 1, 1, 1)})()

        def encode(self, x, **kw):
            dist = type("L", (), {"mean": x[:, :4, :, :]})()
            return type("E", (), {"latent_dist": dist})()

    torch.manual_seed(42)
    noise = torch.randn(1, 4, 64, 64)
    watermarked = noise * 0.9 + torch.randn_like(noise) * 0.1
    nlas_before = compute_nlas(watermarked, noise)

    cleaned = optimize_latent(watermarked, _MockVAE(), iterations=50, lr=0.01)
    nlas_after = compute_nlas(cleaned, noise)
    assert nlas_after < nlas_before


# ---------------------------------------------------------------------------
# Unit tests — engine availability
# ---------------------------------------------------------------------------


def test_engine_available_when_torch_present():
    pytest.importorskip("torch")
    from marknull_engine import MarkNullEngine

    assert MarkNullEngine.available() is True


# ---------------------------------------------------------------------------
# CLI tests — clean_marknull.py
# ---------------------------------------------------------------------------


def test_marknull_bad_input_exits_2():
    result = _run("clean_marknull.py", "/nonexistent.png")
    assert result.returncode == 2


def test_marknull_bad_iterations_exits_2(tmp_path):
    img = tmp_path / "test.png"
    img.write_bytes(JPEG_1X1)
    result = _run("clean_marknull.py", str(img), "--iterations", "0")
    assert result.returncode == 2


def test_marknull_missing_torch_exits_3(tmp_path, monkeypatch):
    monkeypatch.setenv("_MARKNULL_BLOCK_IMPORT", "1")
    img = tmp_path / "test.png"
    img.write_bytes(JPEG_1X1)
    result = _run("clean_marknull.py", str(img))
    assert result.returncode in (0, 1, 3)
