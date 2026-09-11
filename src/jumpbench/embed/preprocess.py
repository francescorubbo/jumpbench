from __future__ import annotations

import numpy as np


def clip_percentile(image: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """Clip each channel to [low, high] percentiles. Image is (C, H, W)."""
    out = image.astype(np.float32, copy=True)
    for c in range(out.shape[0]):
        lo, hi = np.percentile(out[c], [low, high])
        out[c] = np.clip(out[c], lo, hi)
    return out


def rescale_minmax(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    out = image.astype(np.float32, copy=True)
    for c in range(out.shape[0]):
        lo, hi = float(out[c].min()), float(out[c].max())
        out[c] = (out[c] - lo) / (hi - lo + eps)
    return out


def to_8bit(image: np.ndarray) -> np.ndarray:
    scaled = rescale_minmax(image)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.float32)


def standard(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-channel mean/std. Paper 'standard' is not ImageNet normalization."""
    out = image.astype(np.float32, copy=True)
    for c in range(out.shape[0]):
        mean = float(out[c].mean())
        std = float(out[c].std())
        out[c] = (out[c] - mean) / (std + eps)
    return out


def percentile_minmax(
    image: np.ndarray, low: float = 1.0, high: float = 99.0, eps: float = 1e-8
) -> np.ndarray:
    """Map each channel from [low, high] percentiles to [0, 1].

    Accepts (C, H, W) or a stack of crops (N, C, H, W); N is independent.
    """
    if image.ndim not in (3, 4):
        raise ValueError(f"Expected (C,H,W) or (N,C,H,W), got {image.shape}")
    squeeze = image.ndim == 3
    out = image.astype(np.float32, copy=True)
    if squeeze:
        out = out[None]
    n_tiles, n_channels, height, width = out.shape
    flat = out.reshape(n_tiles, n_channels, -1)
    lo = np.percentile(flat, low, axis=-1, keepdims=True)
    hi = np.percentile(flat, high, axis=-1, keepdims=True)
    scaled = np.clip((flat - lo) / (hi - lo + eps), 0.0, 1.0)
    scaled = scaled.reshape(n_tiles, n_channels, height, width)
    return scaled[0] if squeeze else scaled


OPS = {
    "clip_percentile": clip_percentile,
    "rescale_minmax": rescale_minmax,
    "minmax": rescale_minmax,
    "to_8bit": to_8bit,
    "8bit": to_8bit,
    "standard": standard,
    "percentile_minmax": percentile_minmax,
}


def apply_preprocess(image: np.ndarray, steps: list[dict] | None) -> np.ndarray:
    out = image
    for step in steps or []:
        op = step["op"]
        if op not in OPS:
            raise KeyError(f"Unknown preprocess op {op!r}. Known: {sorted(OPS)}")
        kwargs = {k: v for k, v in step.items() if k != "op"}
        out = OPS[op](out, **kwargs)
    return out


def apply_preprocess_tiles(tiles: np.ndarray, steps: list[dict] | None) -> np.ndarray:
    """Apply preprocess independently to each (C, H, W) crop in an (N, C, H, W) stack."""
    if tiles.ndim != 4:
        raise ValueError(f"Expected (N,C,H,W), got {tiles.shape}")
    if tiles.shape[0] == 0 or not steps:
        return tiles
    if len(steps) == 1 and steps[0].get("op") == "percentile_minmax":
        kwargs = {k: v for k, v in steps[0].items() if k != "op"}
        return percentile_minmax(tiles, **kwargs)
    return np.stack([apply_preprocess(tile, steps) for tile in tiles], axis=0)
