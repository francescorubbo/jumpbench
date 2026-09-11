"""RGB contact sheets for crop dry-runs (no model weights)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def chw_to_rgb_u8(tile: np.ndarray) -> np.ndarray:
    """Percentile-stretch a (C,H,W) crop to an (H,W,3) uint8 RGB preview."""
    if tile.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {tile.shape}")
    channels, height, width = tile.shape
    arr = tile.astype(np.float32, copy=False)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    n = min(channels, 3)
    for i in range(n):
        channel = arr[i]
        lo, hi = np.percentile(channel, (1.0, 99.0)) if channel.size else (0.0, 1.0)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(channel.min()), float(channel.max())
            if hi <= lo:
                hi = lo + 1.0
        rgb[:, :, i] = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
    if channels == 1:
        rgb[:, :, 1] = rgb[:, :, 0]
        rgb[:, :, 2] = rgb[:, :, 0]
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def montage_rgb(tiles: np.ndarray, ncol: int = 8, pad: int = 2) -> np.ndarray:
    """Stack (N,C,H,W) crops into one RGB uint8 contact sheet."""
    if tiles.ndim != 4 or tiles.shape[0] == 0:
        raise ValueError("Need a non-empty (N,C,H,W) tile stack")
    n, _c, height, width = tiles.shape
    ncol = max(1, min(int(ncol), n))
    nrow = int(np.ceil(n / ncol))
    cell_h, cell_w = height + pad, width + pad
    sheet = np.full((nrow * cell_h + pad, ncol * cell_w + pad, 3), 30, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, ncol)
        y = pad + r * cell_h
        x = pad + c * cell_w
        sheet[y : y + height, x : x + width] = chw_to_rgb_u8(tile)
    return sheet


def write_png(path: Path, rgb: np.ndarray) -> Path:
    from imagecodecs import png_encode

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_encode(np.ascontiguousarray(rgb)))
    return path
