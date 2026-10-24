"""FLUX 3 image regeneration client (OpenRouter) with a per-image and per-batch cost guard.

The core image is stdlib-only, so httpx (the HTTP client) and Pillow (needed only to
downscale an oversized input) are optional: the module imports without them and
`Flux3Client.generate` raises `Flux3Unavailable` when a missing one is actually needed.

Error contract for callers: `ValueError` = bad input, `Flux3Unavailable` = dependency or
provider missing, `Flux3Error` = the request failed or the response carried no image.
"""

from __future__ import annotations

import base64
import io
from typing import NamedTuple

from common import eprint

try:
    import httpx
except ImportError:
    httpx = None

OPENROUTER_IMAGES_URL = "https://openrouter.ai/api/v1/images"
MODEL = "black-forest-labs/flux-3-image"
REQUEST_TIMEOUT_S = 120.0
# OpenRouter answers 413 for oversized bodies, and a 4K PNG inlined as base64 can pass
# 30 MB. Inputs whose data URL exceeds this are downscaled before upload.
MAX_INLINE_BYTES = 15_000_000

PROMPT_STRATEGIES: dict[str, str] = {
    "minimal": "Enhance this image slightly for improved clarity and sharpness",
    "normal": "Faithfully reproduce this photograph maintaining all visual elements exactly",
    "strong": (
        "Create a fresh rendering of this scene with identical subject, composition, and lighting"
    ),
    "maximum": "Reimagine this image from scratch preserving the exact same scene and composition",
}

_COST_USD = {"768": 0.04, "1k": 0.05, "1.5k": 0.08, "2k": 0.10, "4k": 0.61}
_DEFAULT_COST_USD = 0.05
_LONG_SIDE_PX = {"768": 768, "1k": 1024, "1.5k": 1536, "2k": 2048, "4k": 4096}


class Flux3Error(RuntimeError):
    """The FLUX 3 request failed or its response carried no usable image."""


class Flux3Unavailable(RuntimeError):
    """A dependency (httpx, Pillow) or provider needed for the request is missing."""


class Flux3Result(NamedTuple):
    image_bytes: bytes
    cost_usd: float
    media_type: str


def estimate_cost(resolution: str) -> float:
    """Expected USD cost of one image at `resolution`; unknown resolutions price as 1k."""
    return _COST_USD.get(resolution.lower(), _DEFAULT_COST_USD)


class Flux3CostGuard:
    """Refuses a request whose estimated cost breaks the per-image or per-batch budget."""

    def __init__(self, max_per_image: float = 0.50, max_per_batch: float = 10.0) -> None:
        self.max_per_image = max_per_image
        self.max_per_batch = max_per_batch
        self.spent = 0.0

    def check(self, resolution: str) -> bool:
        cost = estimate_cost(resolution)
        # Round so accumulated float drift cannot refuse a request that lands exactly on budget.
        return cost <= self.max_per_image and round(self.spent + cost, 6) <= self.max_per_batch

    def record(self, cost: float) -> None:
        self.spent += cost


def _media_type(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("FLUX 3 input must be a PNG, JPEG or WebP image")


def _data_url_len(media_type: str, data: bytes) -> int:
    return len(f"data:{media_type};base64,") + 4 * ((len(data) + 2) // 3)


def _downscale(image_bytes: bytes, resolution: str) -> bytes:
    """Re-encode as JPEG, fitted to the output resolution and shrunk until it fits inline."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise Flux3Unavailable(
            "input is too large to send inline and Pillow is not installed to downscale it"
        ) from exc
    try:
        with Image.open(io.BytesIO(image_bytes)) as src:
            img = src.convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("FLUX 3 input could not be decoded for downscaling") from exc
    side = _LONG_SIDE_PX.get(resolution.lower(), 1024)
    while True:
        img.thumbnail((side, side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        out = buf.getvalue()
        if _data_url_len("image/jpeg", out) <= MAX_INLINE_BYTES:
            eprint(
                f"flux3: input over the {MAX_INLINE_BYTES // 1_000_000} MB inline limit; "
                f"sent as {img.width}x{img.height} JPEG ({len(out) / 1e6:.1f} MB)"
            )
            return out
        side = side * 3 // 4


class Flux3Client:
    def __init__(self, api_key: str, provider: str = "openrouter") -> None:
        if provider != "openrouter":
            raise Flux3Unavailable(
                f"FLUX 3 provider {provider!r} is not implemented; use 'openrouter'"
            )
        self.api_key = api_key
        self.provider = provider

    def generate(
        self,
        image_bytes: bytes,
        prompt: str,
        resolution: str = "1k",
        aspect_ratio: str = "auto",
    ) -> Flux3Result:
        """Regenerate `image_bytes` from `prompt` through FLUX 3 image-to-image."""
        if httpx is None:
            raise Flux3Unavailable("httpx is not installed; run: pip install httpx")
        media_type = _media_type(image_bytes)
        if _data_url_len(media_type, image_bytes) > MAX_INLINE_BYTES:
            image_bytes = _downscale(image_bytes, resolution)
            media_type = "image/jpeg"
        data_url = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode()}"
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "input_references": [{"type": "image_url", "image_url": {"url": data_url}}],
            "aspect_ratio": aspect_ratio,
            "resolution": resolution.upper(),
            "n": 1,
        }
        try:
            resp = httpx.post(
                OPENROUTER_IMAGES_URL,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=REQUEST_TIMEOUT_S,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise Flux3Error(f"FLUX 3 request failed: {exc}") from exc
        try:
            body = resp.json()
            item = body["data"][0]
            image = base64.b64decode(item["b64_json"], validate=True)
            cost = body.get("usage", {}).get("cost")
            cost_usd = estimate_cost(resolution) if cost is None else float(cost)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise Flux3Error(f"unexpected FLUX 3 response: {exc!r}") from exc
        if not image:
            raise Flux3Error("FLUX 3 response carried an empty image")
        return Flux3Result(image, cost_usd, item.get("media_type", "image/png"))
