"""Tests for the Tree-Ring watermark decoder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from decode_treering import _score_ring_region, circle_mask, generate_key_pattern

SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from common import subprocess_creationflags


class TestCircleMask:
    def test_center_pixel_always_inside(self):
        mask = circle_mask(64, 5)
        assert mask[32, 32]

    def test_radius_zero_single_pixel(self):
        mask = circle_mask(64, 0)
        assert mask[32, 32]
        assert mask.sum() == 1

    def test_far_corner_outside(self):
        mask = circle_mask(64, 5)
        assert not mask[0, 0]

    def test_shape_matches_size(self):
        mask = circle_mask(128, 10)
        assert mask.shape == (128, 128)

    def test_mask_is_symmetric(self):
        mask = circle_mask(65, 8)
        assert np.array_equal(mask, mask[::-1, :])
        assert np.array_equal(mask, mask[:, ::-1])


class TestGenerateKeyPattern:
    def test_output_shape(self):
        pat = generate_key_pattern(4, 64, 64, 10, 999999)
        assert pat.shape == (1, 4, 64, 64)

    def test_deterministic_with_same_seed(self):
        p1 = generate_key_pattern(4, 64, 64, 10, 42)
        p2 = generate_key_pattern(4, 64, 64, 10, 42)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_produce_different_patterns(self):
        p1 = generate_key_pattern(4, 64, 64, 10, 42)
        p2 = generate_key_pattern(4, 64, 64, 10, 999)
        assert not np.array_equal(p1, p2)

    def test_ring_region_has_constant_values_per_radius(self):
        pat = generate_key_pattern(1, 64, 64, 5, 12345)
        mask_r5 = circle_mask(64, 5)
        mask_r4 = circle_mask(64, 4)
        ring = mask_r5 & ~mask_r4
        vals = pat[0, 0][ring]
        assert np.allclose(vals, vals[0])


class TestPhaseCoherence:
    """Test the phase-coherence scoring logic from detect_treering,
    extracted to avoid needing the SD model."""

    def test_identical_signals_high_coherence(self):
        rng = np.random.RandomState(0)
        signal = rng.randn(64, 64) + 1j * rng.randn(64, 64)
        cross = signal * signal.conj()
        unit = cross / np.abs(cross).clip(min=1e-8)
        coherence = np.abs(unit.mean())
        assert coherence > 0.99

    def test_random_signals_low_coherence(self):
        rng = np.random.RandomState(0)
        a = rng.randn(1000) + 1j * rng.randn(1000)
        b = rng.randn(1000) + 1j * rng.randn(1000)
        cross = a * b.conj()
        unit = cross / np.abs(cross).clip(min=1e-8)
        coherence = np.abs(unit.mean())
        assert coherence < 0.1


class TestScoreRingRegion:
    def test_identical_fft_gives_low_l1_high_coherence(self):
        rng = np.random.RandomState(7)
        fft = rng.randn(1, 4, 16, 16) + 1j * rng.randn(1, 4, 16, 16)
        mask = circle_mask(16, 3)[:16, :16]
        l1, coh = _score_ring_region(fft, fft, mask)
        assert l1 < 1e-6
        assert coh > 0.99

    def test_random_fft_gives_high_l1_low_coherence(self):
        rng = np.random.RandomState(7)
        a = rng.randn(1, 4, 16, 16) + 1j * rng.randn(1, 4, 16, 16)
        b = rng.randn(1, 4, 16, 16) + 1j * rng.randn(1, 4, 16, 16)
        mask = circle_mask(16, 3)[:16, :16]
        l1, coh = _score_ring_region(a, b, mask)
        assert l1 > 0.5
        assert coh < 0.3


class TestCLI:
    def test_missing_file_returns_exit2(self, tmp_path: Path):
        result = subprocess.run(
            [
                sys.executable,
                str(BENCHMARKS / "decode_treering.py"),
                str(tmp_path / "nope.png"),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            creationflags=subprocess_creationflags,
        )
        assert result.returncode == 2
        out = json.loads(result.stdout)
        assert "error" in out

    def test_json_output_has_required_keys(self, tmp_path: Path):
        from PIL import Image

        img = tmp_path / "test.png"
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(str(img))

        result = subprocess.run(
            [sys.executable, str(BENCHMARKS / "decode_treering.py"), str(img), "--json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            creationflags=subprocess_creationflags,
        )
        assert result.returncode in (0, 1), (
            f"unexpected returncode {result.returncode}: {result.stderr}"
        )
        out = json.loads(result.stdout)
        assert "detected" in out
        assert isinstance(out["detected"], bool)
        if result.returncode == 0:
            for key in ("score", "l1_dist", "phase_coherence"):
                assert key in out, f"missing key {key}"
                assert isinstance(out[key], (int, float))
