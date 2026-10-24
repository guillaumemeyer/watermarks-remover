"""Tests for the SDEdit diffusion-resampling engine.

clean_sdedit.py removes latent-space watermarks (Tree-Ring, Gaussian Shading)
by renoising the image to an intermediate diffusion timestep and denoising it
back along a fresh noise path.  Part of the WAVES robustness-evaluation
toolchain.

These tests cover the CLI contract only: argument validation and exit codes.
They never load a diffusion model, so they run without a GPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "clean_sdedit.py"), *args],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


def _png(tmp_path: Path, name: str = "in.png") -> Path:
    """Write a small real PNG, or skip when Pillow is unavailable."""
    Image = pytest.importorskip("PIL.Image")
    p = tmp_path / name
    Image.new("RGB", (16, 16), (128, 128, 128)).save(p, format="PNG")
    return p


def test_missing_file_is_bad_input(tmp_path):
    r = _run(str(tmp_path / "nope.png"))
    assert r.returncode == 2, r.stderr


def test_non_image_file_is_bad_input(tmp_path):
    """A readable file that is not an image must exit 2, not fail later at 1."""
    bad = tmp_path / "notimage.png"
    bad.write_text("this is not a png", encoding="utf-8")
    r = _run(str(bad))
    assert r.returncode == 2, f"expected bad-input exit 2, got {r.returncode}: {r.stderr}"


def test_truncated_jpeg_is_bad_input(tmp_path):
    """Header-only validation passes a truncated JPEG; preflight must decode it."""
    Image = pytest.importorskip("PIL.Image")
    whole = tmp_path / "whole.jpg"
    Image.new("RGB", (64, 64), (200, 100, 50)).save(whole, format="JPEG", quality=95)
    cut = tmp_path / "cut.jpg"
    cut.write_bytes(whole.read_bytes()[:-8])

    r = _run(str(cut))
    assert r.returncode == 2, f"expected bad-input exit 2, got {r.returncode}: {r.stderr}"


def test_bad_input_beats_missing_deps(tmp_path):
    """Without torch/diffusers (as in CI), a bad image must still report 2, not 3."""
    bad = tmp_path / "notimage.png"
    bad.write_text("this is not a png", encoding="utf-8")

    blocker = tmp_path / "sitecustomize_blocker"
    blocker.mkdir()
    (blocker / "torch.py").write_text("raise ImportError('blocked for test')", encoding="utf-8")
    (blocker / "diffusers.py").write_text("raise ImportError('blocked for test')", encoding="utf-8")

    env = {**os.environ, "PYTHONPATH": str(blocker)}
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "clean_sdedit.py"), str(bad)],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
        env=env,
    )
    assert r.returncode == 2, (
        f"bad image without torch should exit 2, got {r.returncode}: {r.stderr}"
    )


@pytest.mark.parametrize(
    "corruption",
    ["unknown_chunk_type", "truncated_idat", "bad_bit_depth", "garbage_idat"],
)
def test_corrupt_png_is_bad_input(tmp_path, corruption):
    """Corrupt PNGs must exit 2, whichever Pillow error the damage produces."""
    Image = pytest.importorskip("PIL.Image")
    good = tmp_path / "good.png"
    Image.new("RGB", (64, 64), (10, 200, 30)).save(good, format="PNG")

    data = bytearray(good.read_bytes())
    idat = data.find(b"IDAT")
    ihdr = data.find(b"IHDR")
    assert idat > 0 and ihdr > 0, "test fixture should contain IHDR and IDAT chunks"

    if corruption == "unknown_chunk_type":
        data[idat : idat + 4] = b"IDXT"
    elif corruption == "truncated_idat":
        data = data[: idat + 20]
    elif corruption == "bad_bit_depth":
        data[ihdr + 12] = 99
    else:
        data[idat + 8 : idat + 24] = b"\x00" * 16

    bad = tmp_path / "corrupt.png"
    bad.write_bytes(bytes(data))

    r = _run(str(bad))
    assert r.returncode == 2, f"{corruption} should exit 2, got {r.returncode}: {r.stderr[:200]}"


def test_strength_out_of_range_is_bad_input(tmp_path):
    img = _png(tmp_path)
    for bad in ("0", "1.5", "-0.2"):
        r = _run(str(img), "--strength", bad)
        assert r.returncode == 2, f"--strength {bad} should be rejected: {r.stderr}"


def test_nonpositive_steps_is_bad_input(tmp_path):
    img = _png(tmp_path)
    r = _run(str(img), "--steps", "0")
    assert r.returncode == 2, r.stderr


def test_non_finite_guidance_is_bad_input(tmp_path):
    img = _png(tmp_path)
    r = _run(str(img), "--guidance", "nan")
    assert r.returncode == 2, r.stderr


def test_strength_times_steps_below_one_is_rejected(tmp_path):
    """strength*steps < 1 yields zero denoising steps, which silently no-ops."""
    img = _png(tmp_path)
    r = _run(str(img), "--strength", "0.001", "--steps", "50")
    assert r.returncode == 2, f"expected rejection of no-op config: {r.stderr}"


def test_encoding_matches_output_extension():
    from clean_sdedit import _encoding_for

    assert _encoding_for(Path("x.png")) == "PNG"
    assert _encoding_for(Path("x.jpg")) == "JPEG"
    assert _encoding_for(Path("x.JPEG")) == "JPEG"
    assert _encoding_for(Path("x.webp")) == "WEBP"
    assert _encoding_for(Path("x.tif")) == "TIFF"
    assert _encoding_for(Path("x.bmp")) == "BMP"
    assert _encoding_for(Path("x.bin")) == "PNG"


def test_webp_is_saved_lossless():
    """WEBP defaults to lossy q80, which would re-degrade the cleaned output."""
    from clean_sdedit import _save_options

    assert _save_options("WEBP") == {"lossless": True}
    assert _save_options("JPEG") == {"quality": 95}
    assert _save_options("PNG") == {}


def test_cuda_index_selects_half_precision(monkeypatch, tmp_path):
    """`cuda:0` must take the same fp16 path as bare `cuda`, not fall back to fp32."""
    import clean_sdedit

    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")

    seen: dict[str, object] = {}

    class _FakePipe:
        def to(self, device):
            seen["device"] = device
            return self

        def set_progress_bar_config(self, **kwargs):
            pass

        def __call__(self, **kwargs):
            raise RuntimeError("stop after construction")

    def _fake_from_pretrained(model_id, **kwargs):
        seen["dtype"] = kwargs.get("torch_dtype")
        return _FakePipe()

    import diffusers

    img = _png(tmp_path)

    monkeypatch.setattr(
        diffusers.StableDiffusionImg2ImgPipeline,
        "from_pretrained",
        staticmethod(_fake_from_pretrained),
    )

    for device in ("cuda", "cuda:0", "cuda:1"):
        seen.clear()
        with pytest.raises(RuntimeError):
            clean_sdedit.sdedit(
                img,
                strength=0.4,
                steps=50,
                guidance=3.0,
                prompt="x",
                model_id="unused",
                device=device,
                seed=None,
            )
        assert seen["dtype"] is torch.float16, f"{device} should select fp16, got {seen['dtype']}"

    seen.clear()
    with pytest.raises(RuntimeError):
        clean_sdedit.sdedit(
            img,
            strength=0.4,
            steps=50,
            guidance=3.0,
            prompt="x",
            model_id="unused",
            device="cpu",
            seed=None,
        )
    assert seen["dtype"] is torch.float32, f"cpu should select fp32, got {seen['dtype']}"
