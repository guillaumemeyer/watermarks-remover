#!/usr/bin/env python3
"""Re-watermarking attack engine for the WAVES robustness-evaluation toolchain.

After removing an original watermark, injects a replacement signal to confuse
detectors.  Three replacement strategies: noise_ring (circular frequency
pattern), fake_stega (LSB bit embedding), spectral_shift (energy injection at
specific frequencies).

All strategies use numpy/scipy only (no torch dependency).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

REPLACEMENT_SCHEMES = ("noise_ring", "fake_stega", "spectral_shift")


class ReWatermarkResult(NamedTuple):
    output_path: str
    original_scheme: str
    replacement_scheme: str
    injection_strength: float


def inject_noise_ring(
    image: np.ndarray,
    strength: float = 0.3,
    radius: int = 20,
    width: int = 3,
) -> np.ndarray:
    """Inject a circular frequency-domain ring pattern."""
    h, w = image.shape[:2]
    result = image.astype(np.float64).copy()

    for c in range(result.shape[2] if result.ndim == 3 else 1):
        channel = result[:, :, c] if result.ndim == 3 else result
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)

        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy : h - cy, -cx : w - cx]
        dist = np.sqrt(y * y + x * x)
        ring_mask = np.exp(-0.5 * ((dist - radius) / max(width, 1)) ** 2)

        phase = np.random.uniform(0, 2 * np.pi, fshift.shape)
        injection = ring_mask * np.exp(1j * phase) * np.max(np.abs(fshift)) * strength * 0.01

        fshift += injection
        channel_out = np.real(np.fft.ifft2(np.fft.ifftshift(fshift)))
        if result.ndim == 3:
            result[:, :, c] = channel_out
        else:
            result = channel_out

    return np.clip(result, 0, 255).astype(np.uint8)


def inject_fake_stega(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Embed random bits in least significant bits of pixel values."""
    result = image.copy()
    n_bits = max(1, int(strength * 3))

    mask_clear = np.uint8((0xFF << n_bits) & 0xFF)
    rng = np.random.default_rng(42)
    noise = rng.integers(0, (1 << n_bits), size=result.shape, dtype=np.uint8)
    result = (result & mask_clear) | noise
    return result


def inject_spectral_shift(
    image: np.ndarray,
    strength: float = 0.3,
    target_freqs: list[int] | None = None,
) -> np.ndarray:
    """Add energy at specific frequency bands."""
    if target_freqs is None:
        target_freqs = [15, 30, 45]

    h, w = image.shape[:2]
    result = image.astype(np.float64).copy()

    for c in range(result.shape[2] if result.ndim == 3 else 1):
        channel = result[:, :, c] if result.ndim == 3 else result
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)

        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy : h - cy, -cx : w - cx]
        dist = np.sqrt(y * y + x * x)

        for freq in target_freqs:
            band = np.exp(-0.5 * ((dist - freq) / 2) ** 2)
            phase = np.random.uniform(0, 2 * np.pi, fshift.shape)
            injection = band * np.exp(1j * phase) * np.max(np.abs(fshift)) * strength * 0.005
            fshift += injection

        channel_out = np.real(np.fft.ifft2(np.fft.ifftshift(fshift)))
        if result.ndim == 3:
            result[:, :, c] = channel_out
        else:
            result = channel_out

    return np.clip(result, 0, 255).astype(np.uint8)


INJECTION_FNS = {
    "noise_ring": inject_noise_ring,
    "fake_stega": inject_fake_stega,
    "spectral_shift": inject_spectral_shift,
}


class ReWatermarkEngine:
    """Applies a replacement watermark signal after removal of the original."""

    def attack(
        self,
        image_path: str,
        original_scheme: str = "unknown",
        replacement_scheme: str = "noise_ring",
        strength: float = 0.3,
        output_path: str | None = None,
    ) -> ReWatermarkResult:
        import io

        from common import cleaned_path, safe_write_bytes
        from PIL import Image

        path = Path(image_path)
        img = Image.open(path).convert("RGB")
        arr = np.array(img)

        if replacement_scheme not in INJECTION_FNS:
            raise ValueError(
                f"unknown replacement: {replacement_scheme}; valid: {sorted(INJECTION_FNS)}"
            )

        fn = INJECTION_FNS[replacement_scheme]
        result_arr = fn(arr, strength=strength)

        out = Path(output_path) if output_path else cleaned_path(path, ".rewm")
        out_img = Image.fromarray(result_arr)

        buf = io.BytesIO()
        fmt = "JPEG" if path.suffix.lower() in (".jpg", ".jpeg") else "PNG"
        out_img.save(buf, format=fmt)
        safe_write_bytes(out, buf.getvalue())

        return ReWatermarkResult(
            output_path=str(out),
            original_scheme=original_scheme,
            replacement_scheme=replacement_scheme,
            injection_strength=strength,
        )
