from __future__ import annotations

import numpy as np


def crop_tiles(image: np.ndarray, tile_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Non-overlapping crops, matching Aliby's ``tile.kind = crop``.

    Image is (C, H, W). Remainder pixels on the right/bottom are dropped.
    Returns tiles (N, C, tile, tile) and integer tile indices (N, 2) as (y, x).
    """
    if image.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {image.shape}")
    _c, height, width = image.shape
    n_y = height // tile_size
    n_x = width // tile_size
    if n_y == 0 or n_x == 0:
        raise ValueError(f"Image {height}x{width} is smaller than tile_size={tile_size}")
    tiles = []
    coords = []
    for iy in range(n_y):
        for ix in range(n_x):
            y0, x0 = iy * tile_size, ix * tile_size
            tiles.append(image[:, y0 : y0 + tile_size, x0 : x0 + tile_size])
            coords.append((iy, ix))
    return np.stack(tiles, axis=0), np.asarray(coords, dtype=np.int32)


def grid_coverage(height: int, width: int, tile_size: int) -> dict[str, float | int]:
    """Tiles and FOV fraction kept by Aliby ``kind: crop`` (right/bottom remainder dropped)."""
    if tile_size < 1:
        raise ValueError(f"tile_size must be >= 1, got {tile_size}")
    n_y = int(height) // int(tile_size)
    n_x = int(width) // int(tile_size)
    kept = n_y * int(tile_size) * n_x * int(tile_size)
    total = int(height) * int(width)
    return {
        "image_height": int(height),
        "image_width": int(width),
        "n_tiles_y": n_y,
        "n_tiles_x": n_x,
        "n_tiles": n_y * n_x,
        "fov_frac": (kept / total) if total else 0.0,
    }


def select_channels(image: np.ndarray, indices: list[int]) -> np.ndarray:
    return image[indices]


def reorder_channels(image: np.ndarray, src_names: list[str], dst_names: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(src_names)}
    missing = [n for n in dst_names if n not in index]
    if missing:
        raise ValueError(f"Cannot reorder to {dst_names}; missing {missing} from {src_names}")
    return image[[index[n] for n in dst_names]]
