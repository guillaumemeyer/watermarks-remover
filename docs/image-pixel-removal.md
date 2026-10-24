# Image Pixel-Watermark Removal

Statistical pixel-level watermarks embedded in AI-generated images
(Tree-Ring, Gaussian Shading, StegaStamp, SynthID-Image, etc.) are
invisible but detectable.  This module provides multiple removal engines,
a classifier, an attack-chain coordinator, and forensic stealth
verification.

## Supported Watermark Types

| Category | Schemes |
|----------|---------|
| Latent-space | Tree-Ring, RingID, ROBIN, WIND, SFW, Gaussian Shading, GaussMarker, PRC, SEAL |
| Pixel-domain | StegaStamp, Stable Signature, TrustMark, VINE, HiDDeN |
| Proprietary | SynthID-Image |

## Engines

| Engine | Script | Method |
|--------|--------|--------|
| FLUX 3 regeneration | `clean_flux3.py` | OpenRouter img2img regeneration via FLUX 3 |
| MarkNull | `clean_marknull.py` | NLAS latent-space optimization (SD 2.1) |
| NFPA | `clean_nfpa.py` | Next-frame prediction attack |
| FMDiffWA | `clean_fmdiffwa.py` | Frequency-domain diffusion watermark attack |
| SynthID bypass | `clean_synthid_bypass.py` | Spectral notch filtering for SynthID |
| Tree-Ring attack | `clean_treering.py` | Frequency notch + optional pixel-space PGD |
| SDEdit resampling | `clean_sdedit.py` | Local diffusion img2img renoise/denoise (SD 2.1) |
| Re-watermarking | `clean_rewatermark.py` | Injects a competing watermark signal |

`clean_sdedit.py` is the default engine for latent-space watermarks.  Unlike
the PGD mode of `clean_treering.py`, which adds a bounded pixel perturbation
that reads as grain, SDEdit resamples the image through the diffusion model so
the output carries model-native texture.  Measured on the Tree-Ring fixture,
PGD raised high-frequency energy to 2.6x the source image's own while SDEdit
stayed at 1.1x.

## Quick Start

```bash
# Single engine
python service/scripts/clean_image.py input.png output.png --remove-pixel marknull

# Auto mode: classify then chain
python service/scripts/clean_image.py input.png output.png \
    --remove-pixel auto --stealth-check

# Explicit chain with classification
python service/scripts/clean_image.py input.png output.png \
    --remove-pixel chain --classify

# FLUX 3 regeneration (needs OPENROUTER_API_KEY)
python service/scripts/clean_image.py input.png output.png \
    --remove-pixel flux3 --flux3-provider openrouter
```

## Attack Chain (auto mode)

When `--remove-pixel auto` is used (or `--remove-pixel chain --classify`), the system:

1. Runs `classify_watermark.py` to detect the watermark type
2. Looks up the optimal attack chain for that type
3. Executes engines in sequence with quality checks between stages
4. Optionally runs `forensic_check.py` to verify stealth (`--stealth-check`)

Chains are defined in `attack_chain.py` and cover 8 watermark schemes.

## Forensic Stealth

After removal, `forensic_check.py` scores the output on two axes:

- **Spectral deformation**: detects unnatural frequency artifacts
- **JPEG double compression**: detects re-encoding signatures

Outputs with a combined stealth score below the threshold are flagged.

## 4K / Large Image Support

`tile_utils.py` provides `split_tiles`, `merge_tiles`, and `process_tiled`
for processing large images without exhausting GPU memory.  Engines can
call `process_tiled(image, engine_fn)` to automatically split into
overlapping tiles, process each, and merge with boundary-aware cosine
feathering.  This is a utility module; individual engines opt in by
calling it explicitly.

## Evaluation

Evaluation scripts measure removal effectiveness (TPR: fraction of
watermarks still detected after removal) and image quality (PSNR, SSIM).
LPIPS and FID require torch and are reported as null when unavailable.

**PSNR/SSIM caveat for latent-space watermarks:** these metrics compare
clean reference images to removal outputs.  For schemes like Tree-Ring
that modify initial latent noise, the clean and watermarked images can
diverge significantly even before any removal is applied, so PSNR/SSIM
reflect both removal artifacts and the inherent latent divergence.
These metrics are most meaningful for pixel-domain watermarks (StegaStamp,
Stable Signature) where clean and watermarked images are visually similar.

**Important:** the built-in classifier (`classify_watermark.py`) is a
routing heuristic based on spatial/frequency features.  It cannot detect
real latent-space watermarks like Tree-Ring or Gaussian Shading.  For
meaningful TPR, supply a watermark-specific decoder via `--decoder`
(e.g., the ONNX decoders from the WAVES project).  Without `--decoder`,
the benchmark reports both baseline TPR (before removal) and post-removal
TPR so you can judge whether the detector is effective on your data.

- **WAVES benchmark** (`benchmarks/bench_waves.py`): per-engine watermark
  removal success rate (TPR via external decoder or built-in classifier)
  and image quality metrics (PSNR, SSIM, optional LPIPS/FID).
- **Forensic evaluation** (`benchmarks/bench_forensic.py`): per-engine
  forensic detectability rates.
- **Full matrix** (`benchmarks/run_full_eval.sh`): watermark x engine x
  intensity sweep.  Produces `results/eval_matrix.json`.

### Dataset setup

The benchmark expects this directory layout:

```
<waves-dir>/<dataset>/<watermark>/*.png   # watermarked images
<waves-dir>/<dataset>/clean/*.png         # clean reference images
```

The [WAVES project](https://github.com/umd-huang-lab/WAVES) does not
distribute a pre-built image dataset.  Its repository contains code to
generate watermarked images from source datasets (COCO, DiffusionDB,
DALL-E 3) using watermark model weights.  To generate authentic
Tree-Ring watermarked images, run Stable Diffusion with the Tree-Ring
latent noise injection from the WAVES `tree_ring/optim_utils.py`
module (requires a GPU and `diffusers`).

Once the dataset directory is prepared:

```bash
# Single engine/watermark pair (built-in classifier, reports baseline)
python benchmarks/bench_waves.py \
    --waves-dir /path/to/dataset --engine chain \
    --watermark tree_ring --output-dir results/ --json

# With external decoder for accurate TPR
# The --decoder command must accept: <img_path> --json
# and print {"detected": true/false} to stdout.
# Note: WAVES decode.py operates on directories, not single images,
# so it is NOT directly compatible — write a thin wrapper or use
# a watermark-specific ONNX decoder instead.
python benchmarks/bench_waves.py \
    --waves-dir /path/to/dataset --engine chain \
    --watermark tree_ring --output-dir results/ --json \
    --decoder python /path/to/your_decoder_wrapper.py

# Full sweep (requires torch + CUDA for GPU engines)
./benchmarks/run_full_eval.sh --waves-dir /path/to/dataset --output-dir results/
```

## Docker

MarkNull and forensic services have dedicated containers in `compose.yaml`:

```bash
docker compose up wr-marknull wr-forensic
```

## Dependencies

Core engines need `numpy`, `scipy`, and `Pillow`.  MarkNull and NFPA
additionally need `torch`, `diffusers`, and `transformers` (install
separately; not in `requirements-dev.txt` since they require GPU-specific
builds).  FLUX 3 needs `httpx`.  `httpx`, `Pillow`, `numpy`, and `scipy`
are listed in `requirements-dev.txt`.
