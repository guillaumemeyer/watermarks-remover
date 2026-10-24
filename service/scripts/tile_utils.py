"""Tiling utilities for processing large images (4K+) without exhausting GPU memory.

Splits images into overlapping tiles, processes each independently, and
reassembles with cosine feathering at overlap regions.
"""

from __future__ import annotations

import numpy as np


def split_tiles(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
) -> list[tuple[np.ndarray, int, int]]:
    """Split image into overlapping tiles.

    Returns list of (tile_array, row_offset, col_offset).
    """
    h, w = image.shape[:2]
    step = tile_size - overlap
    tiles = []

    for r in range(0, h, step):
        for c in range(0, w, step):
            r_end = min(r + tile_size, h)
            c_end = min(c + tile_size, w)
            r_start = max(0, r_end - tile_size)
            c_start = max(0, c_end - tile_size)
            tile = image[r_start:r_end, c_start:c_end]
            tiles.append((tile, r_start, c_start))

    return tiles


def _boundary_weight(size: int, overlap: int, at_start: bool, at_end: bool) -> np.ndarray:
    """1D weight: cosine ramp only on sides that overlap with another tile."""
    w = np.ones(size, dtype=np.float64)
    if overlap <= 0 or overlap >= size:
        return w
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, overlap)))
    if not at_start:
        w[:overlap] = ramp
    if not at_end:
        w[-overlap:] = ramp[::-1]
    return w


def merge_tiles(
    tiles: list[tuple[np.ndarray, int, int]],
    output_shape: tuple[int, ...],
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """Merge overlapping tiles with cosine feathering."""
    h, w = output_shape[:2]
    channels = output_shape[2] if len(output_shape) > 2 else 1

    if channels > 1:
        accum = np.zeros((h, w, channels), dtype=np.float64)
        weight_sum = np.zeros((h, w, channels), dtype=np.float64)
    else:
        accum = np.zeros((h, w), dtype=np.float64)
        weight_sum = np.zeros((h, w), dtype=np.float64)

    for tile, r_start, c_start in tiles:
        th, tw = tile.shape[:2]
        at_top = r_start == 0
        at_bottom = r_start + th >= h
        at_left = c_start == 0
        at_right = c_start + tw >= w
        wy = _boundary_weight(th, overlap, at_top, at_bottom)
        wx = _boundary_weight(tw, overlap, at_left, at_right)
        weight_2d = np.outer(wy, wx)

        if channels > 1:
            weight_3d = weight_2d[:, :, np.newaxis]
            accum[r_start : r_start + th, c_start : c_start + tw] += (
                tile.astype(np.float64) * weight_3d
            )
            weight_sum[r_start : r_start + th, c_start : c_start + tw] += weight_3d
        else:
            accum[r_start : r_start + th, c_start : c_start + tw] += (
                tile.astype(np.float64) * weight_2d
            )
            weight_sum[r_start : r_start + th, c_start : c_start + tw] += weight_2d

    weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    result = accum / weight_sum
    return np.clip(result, 0, 255).astype(np.uint8)


def process_tiled(
    image: np.ndarray,
    process_fn,
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """Split, process each tile, merge results."""
    tiles = split_tiles(image, tile_size, overlap)
    processed = []
    for tile, r, c in tiles:
        result = process_fn(tile)
        processed.append((result, r, c))
    return merge_tiles(processed, image.shape, tile_size, overlap)
