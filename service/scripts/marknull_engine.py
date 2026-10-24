"""MarkNull: latent-space watermark removal via NLAS optimisation.

Reimplemented from the MarkNull paper (arXiv 2608.10166).  Part of the WAVES
robustness-evaluation toolchain for measuring watermark fragility.

The core idea: latent-space watermarks (Tree-Ring, Gaussian Shading, etc.) embed
a signal correlated with the diffusion noise schedule.  The Noise-Latent
Alignment Score (NLAS) measures this correlation in the Fourier domain.
Gradient descent on the latent minimises NLAS while a reconstruction loss
preserves visual quality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

_torch = None
_diffusers = None


def _lazy_torch():
    global _torch  # noqa: PLW0603
    if _torch is None:
        import torch

        _torch = torch
    return _torch


def _lazy_diffusers():
    global _diffusers  # noqa: PLW0603
    if _diffusers is None:
        import diffusers

        _diffusers = diffusers
    return _diffusers


def resolve_device(raw: str | None = None) -> str:
    if raw and raw != "auto":
        return raw
    try:
        t = _lazy_torch()
        if t.cuda.is_available():
            return "cuda"
        mps = getattr(t.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # noqa: S110
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# NLAS — Noise-Latent Alignment Score
# ---------------------------------------------------------------------------


def compute_nlas(latent, noise) -> float:
    """Fourier-domain phase coherence between latent and noise.

    Returns a scalar in [0, 1]: ~0 for uncorrelated signals (random
    phases), ~1 when cross-spectrum phases are locked (watermark present).
    """
    torch = _lazy_torch()
    with torch.no_grad():
        scores: list[float] = []
        for c in range(latent.shape[1]):
            l_fft = torch.fft.fft2(latent[:, c])
            n_fft = torch.fft.fft2(noise[:, c])
            cross = l_fft * n_fft.conj()
            unit = cross / cross.abs().clamp(min=1e-8)
            scores.append(unit.mean().abs().item())
        raw = sum(scores) / len(scores)
        return float(max(0.0, min(1.0, raw)))


# ---------------------------------------------------------------------------
# Latent optimisation
# ---------------------------------------------------------------------------


def optimize_latent(latent, vae, *, iterations: int = 100, lr: float = 0.01):
    """Gradient descent on *latent* to minimise NLAS with a recon penalty.

    *vae* must expose ``decode(z)`` and ``encode(x)`` (Stable Diffusion AE
    interface).  Returns the optimised latent (detached).
    """
    torch = _lazy_torch()
    z = latent.clone().detach().requires_grad_(True)
    noise_ref = torch.randn_like(z)
    optimizer = torch.optim.Adam([z], lr=lr)
    scaling = getattr(vae.config, "scaling_factor", 0.18215)

    for _ in range(iterations):
        optimizer.zero_grad()
        nlas = _differentiable_nlas(z, noise_ref)
        decoded = vae.decode(z / scaling).sample
        re_encoded = vae.encode(decoded).latent_dist.mean * scaling
        recon_loss = torch.nn.functional.mse_loss(re_encoded, z)
        loss = nlas + 0.1 * recon_loss
        loss.backward()
        optimizer.step()

    return z.detach()


def _differentiable_nlas(latent, noise):
    """Differentiable NLAS for backprop (no ``torch.no_grad``)."""
    torch = _lazy_torch()
    scores = []
    for c in range(latent.shape[1]):
        l_fft = torch.fft.fft2(latent[:, c])
        n_fft = torch.fft.fft2(noise[:, c])
        cross = (l_fft * n_fft.conj()).abs().mean()
        norm = (l_fft.abs().mean() * n_fft.abs().mean()).clamp(min=1e-8)
        scores.append(cross / norm)
    return torch.stack(scores).mean()


# ---------------------------------------------------------------------------
# Engine class
# ---------------------------------------------------------------------------


class MarkNullEngine:
    """High-level API for MarkNull watermark removal."""

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1-base",
        device: str = "auto",
    ):
        self._model_id = model_id
        self._device = resolve_device(device)
        self._vae = None

    def _load_vae(self):
        if self._vae is not None:
            return self._vae
        diffusers = _lazy_diffusers()
        self._vae = diffusers.AutoencoderKL.from_pretrained(self._model_id, subfolder="vae").to(
            self._device
        )
        self._vae.eval()
        return self._vae

    def remove(
        self,
        image: Image,
        mode: str = "optimize",
        iterations: int = 100,
    ) -> Image:
        import numpy as np
        from PIL import Image as PILImage

        t = _lazy_torch()
        vae = self._load_vae()
        scaling = getattr(vae.config, "scaling_factor", 0.18215)

        img_t = (
            t.from_numpy(np.array(image.convert("RGB")).astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self._device)
        )
        img_t = img_t * 2.0 - 1.0

        with t.no_grad():
            latent = vae.encode(img_t).latent_dist.mean * scaling

        if mode == "optimize":
            latent = optimize_latent(latent, vae, iterations=iterations)

        with t.no_grad():
            decoded = vae.decode(latent / scaling).sample
            decoded = ((decoded.clamp(-1, 1) + 1) / 2 * 255).byte()
            decoded = decoded[0].permute(1, 2, 0).cpu().numpy()

        return PILImage.fromarray(decoded, "RGB")

    @staticmethod
    def available() -> bool:
        try:
            _lazy_torch()
            _lazy_diffusers()
            return True
        except Exception:
            return False
