#!/usr/bin/env python3
"""Tree-Ring watermark attack engine for the WAVES robustness-evaluation toolchain.

Tree-Ring watermarks embed a circular pattern in the Fourier domain of the
initial latent noise.  This engine detects the ring pattern and removes it
via frequency-domain notch filtering (numpy/scipy) or VAE+PGD optimization
(requires torch).

Exit codes:
  0  attack completed
  1  runtime error
  2  bad input
  3  engine unavailable (torch required for pgd mode)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))


class TreeRingResult(NamedTuple):
    output_path: str
    ring_energy_before: float
    ring_energy_after: float
    psnr: float
    iterations_used: int


def _lazy_torch():
    import torch

    return torch


def compute_radial_power(image: np.ndarray) -> np.ndarray:
    """Compute radial power spectrum of a grayscale image."""
    gray = (
        np.mean(image.astype(np.float64), axis=2) if image.ndim == 3 else image.astype(np.float64)
    )
    f = np.fft.fft2(gray)
    mag = np.abs(np.fft.fftshift(f)) ** 2

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)

    radial = np.zeros(max_r)
    y, x = np.ogrid[-cy : h - cy, -cx : w - cx]
    dist = np.sqrt(y * y + x * x).astype(int)

    for r in range(max_r):
        mask = dist == r
        if mask.any():
            radial[r] = np.mean(mag[mask])

    return radial


def detect_ring_pattern(
    image: np.ndarray,
    min_radius: int = 8,
    max_radius: int = 40,
) -> dict:
    """Detect Tree-Ring watermark pattern in the frequency domain."""
    radial = compute_radial_power(image)
    search = radial[min_radius : max_radius + 1]

    if len(search) < 3:
        return {"ring_energy": 0.0, "peak_radius": 0, "confidence": 0.0}

    log_radii = np.log10(np.arange(min_radius, max_radius + 1, dtype=np.float64))
    log_power = np.log10(np.where(search > 0, search, 1e-10))

    if len(log_radii) > 1:
        trend = np.polyval(np.polyfit(log_radii, log_power, 1), log_radii)
        residual = log_power - trend
    else:
        residual = log_power

    peak_idx = int(np.argmax(residual))
    peak_radius = min_radius + peak_idx
    ring_energy = float(search[peak_idx])

    std = float(np.std(residual))
    confidence = float(np.clip(residual[peak_idx] / (std + 1e-10) / 5.0, 0.0, 1.0))

    return {
        "ring_energy": ring_energy,
        "peak_radius": peak_radius,
        "confidence": confidence,
    }


def notch_filter_attack(
    image: np.ndarray,
    detected_radii: list[int] | None = None,
    width: int = 3,
) -> np.ndarray:
    """Remove ring patterns via frequency-domain notch filtering."""
    if detected_radii is None:
        detection = detect_ring_pattern(image)
        detected_radii = [detection["peak_radius"]]

    h, w_img = image.shape[:2]
    result = image.astype(np.float64).copy()

    cy, cx = h // 2, w_img // 2
    y, x = np.ogrid[-cy : h - cy, -cx : w_img - cx]
    dist = np.sqrt(y * y + x * x)

    notch = np.ones((h, w_img))
    for r in detected_radii:
        if r > 0:
            suppress = np.exp(-0.5 * ((dist - r) / max(width, 1)) ** 2)
            notch *= 1.0 - suppress * 0.9

    for c in range(result.shape[2] if result.ndim == 3 else 1):
        channel = result[:, :, c] if result.ndim == 3 else result
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        fshift *= notch
        channel_out = np.real(np.fft.ifft2(np.fft.ifftshift(fshift)))
        if result.ndim == 3:
            result[:, :, c] = channel_out
        else:
            result = channel_out

    return np.clip(result, 0, 255).astype(np.uint8)


def pgd_attack(
    image: np.ndarray,
    iterations: int = 100,
    epsilon: float = 0.03,
    step_size: float = 0.01,
) -> tuple[np.ndarray, int]:
    """PGD attack to minimize ring energy while preserving image quality.

    Requires torch. Returns (result_array, iterations_used).
    """
    t = _lazy_torch()

    img_float = image.astype(np.float64) / 255.0
    tensor = t.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).float()
    delta = t.zeros_like(tensor, requires_grad=True)

    for _i in range(iterations):
        perturbed = t.clamp(tensor + delta, 0, 1)

        gray = perturbed.mean(dim=1, keepdim=True).squeeze()
        f = t.fft.fft2(gray)
        mag = t.abs(t.fft.fftshift(f))

        h, w_t = mag.shape[-2:]
        cy, cx = h // 2, w_t // 2
        y = t.arange(-cy, h - cy, dtype=t.float32).unsqueeze(1)
        x_coord = t.arange(-cx, w_t - cx, dtype=t.float32).unsqueeze(0)
        dist = t.sqrt(y**2 + x_coord**2)

        ring_mask = ((dist >= 8) & (dist <= 40)).float()
        ring_energy = (mag * ring_mask).sum()

        ring_energy.backward()

        with t.no_grad():
            if delta.grad is not None:
                delta.data -= step_size * delta.grad.sign()
                delta.data = t.clamp(delta.data, -epsilon, epsilon)
                delta.grad.zero_()

    result = t.clamp(tensor + delta, 0, 1).squeeze().permute(1, 2, 0).detach().numpy()
    return (result * 255).astype(np.uint8), iterations


def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(255.0**2 / mse))


class TreeRingAttacker:
    """Combined Tree-Ring watermark attack engine."""

    def detect(self, image_path: str) -> dict:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        return detect_ring_pattern(np.array(img))

    def attack(
        self,
        image_path: str,
        mode: str = "filter",
        iterations: int = 100,
        epsilon: float = 0.03,
        step_size: float = 0.01,
        output_path: str | None = None,
    ) -> TreeRingResult:
        from common import cleaned_path, safe_write_bytes
        from PIL import Image

        path = Path(image_path)
        img = Image.open(path).convert("RGB")
        arr = np.array(img)

        detection = detect_ring_pattern(arr)
        ring_before = detection["ring_energy"]

        if mode == "pgd":
            result_arr, iters = pgd_attack(arr, iterations, epsilon, step_size)
        elif mode == "filter":
            result_arr = notch_filter_attack(arr, [detection["peak_radius"]])
            iters = 0
        else:
            raise ValueError(f"unknown mode: {mode}; valid: filter, pgd")

        detection_after = detect_ring_pattern(result_arr)
        ring_after = detection_after["ring_energy"]
        psnr = compute_psnr(arr, result_arr)

        import io

        out = Path(output_path) if output_path else cleaned_path(path, ".treering")
        buf = io.BytesIO()
        fmt = "JPEG" if path.suffix.lower() in (".jpg", ".jpeg") else "PNG"
        Image.fromarray(result_arr).save(buf, format=fmt)
        safe_write_bytes(out, buf.getvalue())

        return TreeRingResult(
            output_path=str(out),
            ring_energy_before=round(ring_before, 4),
            ring_energy_after=round(ring_after, 4),
            psnr=round(psnr, 2),
            iterations_used=iters,
        )
