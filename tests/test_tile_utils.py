"""Tests for image tiling utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def test_split_tiles_covers_image():
    from tile_utils import split_tiles

    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    tiles = split_tiles(img, tile_size=128, overlap=32)
    assert len(tiles) > 0
    for tile, _r, _c in tiles:
        assert tile.shape[0] <= 128
        assert tile.shape[1] <= 128


def test_split_tiles_non_divisible():
    from tile_utils import split_tiles

    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    tiles = split_tiles(img, tile_size=128, overlap=32)
    assert len(tiles) > 0
    for tile, row, col in tiles:
        assert row + tile.shape[0] <= 300
        assert col + tile.shape[1] <= 400


def test_merge_tiles_preserves_identity():
    from tile_utils import merge_tiles, split_tiles

    img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    tiles = split_tiles(img, tile_size=128, overlap=32)
    result = merge_tiles(tiles, img.shape, tile_size=128, overlap=32)
    assert result.shape == img.shape
    assert np.allclose(result, img, atol=2)


def test_process_tiled_identity():
    from tile_utils import process_tiled

    img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    result = process_tiled(img, lambda x: x, tile_size=128, overlap=32)
    assert result.shape == img.shape
    assert np.allclose(result, img, atol=2)


def test_process_tiled_large_image():
    from tile_utils import process_tiled

    img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    result = process_tiled(img, lambda x: x, tile_size=512, overlap=64)
    assert result.shape == img.shape


def test_process_tiled_grayscale():
    from tile_utils import process_tiled

    img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    result = process_tiled(img, lambda x: x, tile_size=128, overlap=32)
    assert result.shape == img.shape
    assert np.allclose(result, img, atol=2)
