"""NFPA (next-frame prediction attack) latent watermark remover.

Reimplemented from "The Future Unmarked: Watermark Removal via Next-Frame
Prediction" (NeurIPS 2025); no reference code is vendored. The watermarked
image is treated as the current frame: DDIM inversion recovers its initial
noise, an edge-driven flow field warps that noise as a small "next frame"
motion, and DDIM sampling from the warped noise regenerates the image. Latent
watermarks (Tree-Ring, Gaussian Shading, ...) live in the noise-latent
correlation, which the warp breaks while the image content survives.

torch, diffusers and transformers are optional and imported lazily, so
``NFPAEngine.available()`` works without them.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

# Same SD 2.1 base weights as the MarkDiffusion harness, so no second download.
from markdiffusion_harness import DEFAULT_MODEL, resolve_device

if TYPE_CHECKING:
    from PIL.Image import Image
    from torch import Tensor

# Largest displacement in image pixels (one latent pixel at 1/8 resolution).
MAX_FLOW_PIXELS = 8.0


def build_flow_matrix(image: Tensor, strength: float = 0.05) -> Tensor:
    """Edge-driven flow field of shape (N, 2, H, W) in image pixels (x, y).

    Pixels move along the local edge tangent by an amount proportional to the
    Sobel edge magnitude times ``strength`` (a fraction of the short side),
    soft-capped below ``MAX_FLOW_PIXELS``. Flat regions do not move.
    """
    import torch
    import torch.nn.functional as F

    image = image.float()
    if image.shape[1] == 3:
        weights = torch.tensor([0.299, 0.587, 0.114], device=image.device).view(1, 3, 1, 1)
        gray = (image * weights).sum(dim=1, keepdim=True)
    else:
        gray = image.mean(dim=1, keepdim=True)

    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    kernels = torch.stack([sobel_x, sobel_x.T]).unsqueeze(1).to(image.device)
    grads = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), kernels)
    gx, gy = grads[:, 0], grads[:, 1]
    magnitude = torch.sqrt(gx**2 + gy**2)

    peak = magnitude.amax(dim=(1, 2), keepdim=True)
    edges = magnitude / (peak + 1e-8)
    raw = edges * strength * min(image.shape[-2:])
    displacement = MAX_FLOW_PIXELS * torch.tanh(raw / MAX_FLOW_PIXELS)

    # Unit edge tangent (-gy, gx); zero where there is no gradient.
    scale = displacement / (magnitude + 1e-8)
    return torch.stack([-gy * scale, gx * scale], dim=1)


def warp_noise(noise: Tensor, flow: Tensor) -> Tensor:
    """Backward-warp ``noise`` (N, C, h, w) by ``flow`` (N or 1, 2, H, W).

    The flow is resampled to the noise resolution and its pixel displacements
    rescaled. Bilinear sampling shrinks the variance of i.i.d. noise, so each
    channel is restored to the input mean and std afterwards.
    """
    import torch
    import torch.nn.functional as F

    n, _, h, w = noise.shape
    src = noise.float()
    flow = flow.to(device=noise.device, dtype=torch.float32)
    flow_h, flow_w = flow.shape[-2:]
    if (flow_h, flow_w) != (h, w):
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
        flow = flow * torch.tensor([w / flow_w, h / flow_h], device=noise.device).view(1, 2, 1, 1)
    flow = flow.expand(n, -1, -1, -1)

    ys, xs = torch.meshgrid(
        torch.arange(h, device=noise.device, dtype=torch.float32),
        torch.arange(w, device=noise.device, dtype=torch.float32),
        indexing="ij",
    )
    sample_x = xs + flow[:, 0]
    sample_y = ys + flow[:, 1]
    # Pixel centers to [-1, 1] under align_corners=False.
    grid = torch.stack([(2 * sample_x + 1) / w - 1, (2 * sample_y + 1) / h - 1], dim=-1)
    warped = F.grid_sample(
        src, grid, mode="bilinear", padding_mode="reflection", align_corners=False
    )

    src_mean = src.mean(dim=(2, 3), keepdim=True)
    src_std = src.std(dim=(2, 3), keepdim=True)
    out_mean = warped.mean(dim=(2, 3), keepdim=True)
    out_std = warped.std(dim=(2, 3), keepdim=True)
    warped = (warped - out_mean) / (out_std + 1e-8) * src_std + src_mean
    return warped.to(noise.dtype)


def _ddim_loop(unet: Any, scheduler: Any, latents: Tensor, text: Tensor, steps: int) -> Tensor:
    """Run an unconditional DDIM pass (forward or inverse) over ``steps``."""
    scheduler.set_timesteps(steps, device=latents.device)
    for t in scheduler.timesteps:
        eps = unet(latents, t, encoder_hidden_states=text).sample
        latents = scheduler.step(eps, t, latents).prev_sample
    return latents


class NFPAEngine:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "auto") -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self._pipe: Any = None

    @staticmethod
    def available() -> bool:
        return all(
            importlib.util.find_spec(name) is not None
            for name in ("torch", "diffusers", "transformers")
        )

    def _pipeline(self) -> Any:
        if self._pipe is None:
            import torch
            from diffusers import StableDiffusionPipeline

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id, torch_dtype=dtype, safety_checker=None
            ).to(self.device)
        return self._pipe

    def remove(self, image: Image, flow_strength: float = 0.05, steps: int = 50) -> Image:
        import numpy as np
        import torch
        from diffusers import DDIMInverseScheduler, DDIMScheduler
        from PIL import Image as PILImage

        pipe = self._pipeline()
        rgb = image.convert("RGB")
        size = rgb.size
        # The SD VAE needs sides that are multiples of 8.
        work_size = (max(8, size[0] - size[0] % 8), max(8, size[1] - size[1] % 8))
        if work_size != size:
            rgb = rgb.resize(work_size, PILImage.Resampling.LANCZOS)
        pixels = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0)
        pixels = pixels.permute(2, 0, 1).unsqueeze(0).to(self.device, pipe.vae.dtype)
        scaling = pipe.vae.config.scaling_factor

        with torch.no_grad():
            tokens = pipe.tokenizer(
                [""],
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                return_tensors="pt",
            )
            text = pipe.text_encoder(tokens.input_ids.to(self.device))[0]

            latents = pipe.vae.encode(pixels).latent_dist.mean * scaling
            inverse = DDIMInverseScheduler.from_config(pipe.scheduler.config)
            noise = _ddim_loop(pipe.unet, inverse, latents, text, steps)

            flow = build_flow_matrix(pixels, strength=flow_strength)
            warped = warp_noise(noise, flow)

            forward = DDIMScheduler.from_config(pipe.scheduler.config)
            latents = _ddim_loop(pipe.unet, forward, warped, text, steps)
            decoded = pipe.vae.decode(latents / scaling).sample

        array = ((decoded[0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        result = PILImage.fromarray(array.permute(1, 2, 0).cpu().numpy())
        if result.size != size:
            result = result.resize(size, PILImage.Resampling.LANCZOS)
        return result
