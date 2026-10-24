"""FMDiffWA (frequency-modulated diffusion watermark attack) remover.

Reimplemented from arXiv 2604.22220; no reference code is vendored.
Frequency filtering runs on the full image at once (with reflection
padding to reduce perimeter artifacts). Radial bands whose energy rises
above the natural-image power-law trend, and isolated peaks that stand
out of their band, are treated as watermark-correlated and softly pulled
back to the expected level. An optional low-strength Stable Diffusion
img2img pass tiles the filtered result with guaranteed overlap and
cosine feathering to restore texture.

torch is required; diffusers and transformers are optional and imported
lazily. Without them the engine runs the frequency stage only.
"""

from __future__ import annotations

import importlib.util
import math
from typing import TYPE_CHECKING, Any

# Same SD 2.1 base weights as the MarkDiffusion harness and NFPA, so no second download.
from markdiffusion_harness import DEFAULT_MODEL, resolve_device

if TYPE_CHECKING:
    from PIL.Image import Image
    from torch import Tensor

# Radii below this (cycles/pixel) are never modified: DC and coarse layout carry content.
LOW_FREQ_CUTOFF = 0.02
# Robust z-score above which a band's energy is anomalous versus the power-law trend.
BAND_Z = 2.5
# z-score within a band above which a single coefficient is an anomalous peak.
PEAK_Z = 3.0
# Sigmoid width of the soft threshold, in z units.
SOFTNESS = 0.25
# Multiplier on the log-magnitude excess. 2 applies the Wiener gain
# P_expected / P_observed (MMSE estimate of the unmarked band); 1 would only bring
# the band's energy back to trend, leaving most of a mark that matches the content's power.
CUT_GAIN = 2.0
# Patches smaller than this give too few coefficients for a spectral estimate.
MIN_FILTER_SIZE = 16


def _axis_starts(size: int, patch: int, min_overlap: int | None = None) -> list[int]:
    """Patch origins along one axis; patches overlap so the last ends flush with the edge."""
    if size <= patch:
        return [0]
    if min_overlap is None:
        min_overlap = -(-patch // 4)  # ceil(patch/4), proportional
    min_overlap = min(min_overlap, patch - 1)
    step = patch - min_overlap
    count = max(2, math.ceil((size - min_overlap) / step))
    return [round(i * (size - patch) / (count - 1)) for i in range(count)]


def _axis_weights(starts: list[int], length: int, index: int) -> Tensor:
    """1-D feather weights: 1 inside, a raised-cosine ramp across each overlap."""
    import torch

    weights = torch.ones(length)
    if index > 0:
        overlap = starts[index - 1] + length - starts[index]
        if overlap > 0:
            ramp = 0.5 - 0.5 * torch.cos(torch.pi * (torch.arange(overlap) + 0.5) / overlap)
            weights[:overlap] = ramp
    if index < len(starts) - 1:
        overlap = starts[index] + length - starts[index + 1]
        if overlap > 0:
            ramp = 0.5 - 0.5 * torch.cos(torch.pi * (torch.arange(overlap) + 0.5) / overlap)
            weights[length - overlap :] *= ramp.flip(0)
    return weights


def _patch_windows(
    height: int, width: int, patch_size: int, min_overlap: int | None = None
) -> list[tuple[slice, slice, Tensor]]:
    """Row-major (rows, cols, 2-D feather weight) for every patch of the tiling."""
    ph, pw = min(patch_size, height), min(patch_size, width)
    ys, xs = _axis_starts(height, ph, min_overlap), _axis_starts(width, pw, min_overlap)
    windows = []
    for i, y in enumerate(ys):
        wy = _axis_weights(ys, ph, i)
        for j, x in enumerate(xs):
            wx = _axis_weights(xs, pw, j)
            windows.append((slice(y, y + ph), slice(x, x + pw), wy[:, None] * wx[None, :]))
    return windows


def merge_patches(patches: list[Tensor], shape: tuple[int, ...], patch_size: int) -> Tensor:
    """Blend patches from ``_split_patches`` back into an (N, C, H, W) image."""
    import torch

    n, c, h, w = shape
    ref = patches[0]
    out = torch.zeros(n, c, h, w, dtype=torch.float32, device=ref.device)
    norm = torch.zeros(1, 1, h, w, dtype=torch.float32, device=ref.device)
    for patch, (rows, cols, cpu_weight) in zip(
        patches, _patch_windows(h, w, patch_size), strict=True
    ):
        weight = cpu_weight.to(ref.device)
        out[..., rows, cols] += patch.float() * weight
        norm[..., rows, cols] += weight
    return (out / norm).to(ref.dtype)


def _band_mean(values: Tensor, band: Tensor, count: Tensor) -> Tensor:
    """Mean of (N, C, K) ``values`` per radial band index -> (N, C, B)."""
    import torch

    sums = torch.zeros(*values.shape[:2], count.numel(), device=values.device)
    return sums.index_add_(2, band, values) / count


def _suppress_patch(patch: Tensor) -> Tensor:
    """Softly pull anomalous spectral bands and peaks of one patch back to trend."""
    import torch

    h, w = patch.shape[-2:]
    if min(h, w) < MIN_FILTER_SIZE:
        return patch
    x = patch.float()
    spectrum = torch.fft.rfft2(x)
    log_mag = torch.log(spectrum.abs() + 1e-8).flatten(2)  # (N, C, K)

    fy = torch.fft.fftfreq(h, device=x.device)
    fx = torch.fft.rfftfreq(w, device=x.device)
    radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2).flatten()
    n_bands = max(8, min(h, w) // 8)
    rmax = radius.max()
    band = (radius / rmax * (n_bands - 1)).round().long()
    raw_count = torch.bincount(band, minlength=n_bands).float()
    count = raw_count.clamp(min=1)

    # Energy profile: mean and spread of log-magnitude per radial band.
    mean = _band_mean(log_mag, band, count)
    std = (_band_mean(log_mag**2, band, count) - mean**2).clamp(min=1e-12).sqrt()

    # Expected profile: natural images follow a power law, linear in log-log.
    centers = torch.arange(n_bands, device=x.device) * rmax / (n_bands - 1)
    valid = (raw_count > 0) & (centers >= LOW_FREQ_CUTOFF)
    logr = torch.log(centers[valid])
    y = mean[..., valid]
    dx = logr - logr.mean()
    slope = (dx * (y - y.mean(-1, keepdim=True))).sum(-1, keepdim=True) / (dx**2).sum()
    trend = y.mean(-1, keepdim=True) + slope * dx
    excess = y - trend
    med = excess.median(-1, keepdim=True).values
    mad = 1.4826 * (excess - med).abs().median(-1, keepdim=True).values + 1e-6
    band_w = torch.sigmoid(((excess - med) / mad - BAND_Z) / SOFTNESS)
    band_cut = torch.zeros_like(mean)
    band_cut[..., valid] = band_w * (excess - med).clamp(min=0)

    # Isolated peaks: coefficients far above their own band's distribution.
    above = log_mag - mean[..., band]
    peak_w = torch.sigmoid((above / std[..., band] - PEAK_Z) / SOFTNESS)
    log_mask = -CUT_GAIN * (band_cut[..., band] + peak_w * above.clamp(min=0))
    log_mask[..., radius < LOW_FREQ_CUTOFF] = 0

    mask = torch.exp(log_mask).view_as(spectrum)
    return torch.fft.irfft2(spectrum * mask, s=(h, w)).to(patch.dtype)


class FrequencyWatermarkModulator:
    def __init__(self, patch_size: int = 256) -> None:
        self.patch_size = patch_size

    def modulate(self, image: Tensor) -> Tensor:
        """Suppress watermark-correlated frequencies of the full (N, C, H, W) image at once."""
        import torch.nn.functional as F

        h, w = image.shape[-2:]
        pad = min(128, h // 4, w // 4)
        if pad >= 4:
            padded = F.pad(image, (pad, pad, pad, pad), mode="reflect")
            filtered = _suppress_patch(padded)
            return filtered[..., pad : pad + h, pad : pad + w]
        return _suppress_patch(image)

    def _split_patches(self, image: Tensor, patch_size: int) -> list[Tensor]:
        """Overlapping ``patch_size`` tiles of an (N, C, H, W) image, row-major."""
        h, w = image.shape[-2:]
        return [image[..., rows, cols] for rows, cols, _ in _patch_windows(h, w, patch_size)]


class FMDiffWAEngine:
    def __init__(
        self, model_id: str = DEFAULT_MODEL, patch_size: int = 256, device: str = "auto"
    ) -> None:
        self.model_id = model_id
        self.patch_size = patch_size
        self.device = resolve_device(device)
        self.modulator = FrequencyWatermarkModulator(patch_size)
        self.refined = False
        self._pipe: Any = None

    @staticmethod
    def available() -> bool:
        return all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy", "PIL"))

    @staticmethod
    def refinement_available() -> bool:
        return all(
            importlib.util.find_spec(name) is not None for name in ("diffusers", "transformers")
        )

    def _pipeline(self) -> Any:
        if self._pipe is None:
            import torch
            from diffusers import StableDiffusionImg2ImgPipeline

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
            ).to(self.device)
            self._pipe.set_progress_bar_config(disable=True)
        return self._pipe

    def _refine(self, patch: Tensor, intensity: float, steps: int) -> Tensor:
        """Low-strength img2img at the model's native resolution, resized back."""
        import torch.nn.functional as F

        pipe = self._pipeline()
        size = patch.shape[-2:]
        scale = pipe.unet.config.sample_size * pipe.vae_scale_factor / max(size)
        work = [max(8, round(s * scale / 8) * 8) for s in size]
        up = F.interpolate(patch, size=work, mode="bicubic", align_corners=False).clamp(0, 1)
        out = pipe(
            prompt="",
            image=up.to(pipe.dtype),
            strength=intensity,
            num_inference_steps=steps,
            guidance_scale=1.0,
            output_type="pt",
        ).images
        return F.interpolate(
            out.float(), size=tuple(size), mode="bicubic", align_corners=False, antialias=True
        ).clamp(0, 1)

    def remove(self, image: Image, intensity: float = 0.3, steps: int = 50) -> Image:
        """Frequency-suppress the full image, optionally diffusion-refine per tile, and merge.

        ``intensity`` is the img2img strength in [0, 1]; refinement runs only when
        diffusers + transformers are installed and ``steps * intensity >= 1``.
        ``self.refined`` records whether it ran.
        """
        import numpy as np
        import torch
        from PIL import Image as PILImage

        if not 0 <= intensity <= 1:
            raise ValueError(f"intensity must be in [0, 1]: {intensity}")
        rgb = image.convert("RGB")
        pixels = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0)
        pixels = pixels.permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.refined = int(steps * intensity) >= 1 and self.refinement_available()

        with torch.no_grad():
            filtered = self.modulator.modulate(pixels).clamp(0, 1)

            if self.refined:
                out = []
                for patch in self.modulator._split_patches(filtered, self.patch_size):
                    out.append(self._refine(patch, intensity, steps))
                merged = merge_patches(out, filtered.shape, self.patch_size)
            else:
                merged = filtered

        array = (merged[0].clamp(0, 1) * 255).round().to(torch.uint8)
        return PILImage.fromarray(array.permute(1, 2, 0).cpu().numpy())
