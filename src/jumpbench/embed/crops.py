"""Instance crops from Cellpose label masks. Windows that leave the FOV are skipped."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class InstanceCrops:
    tiles: np.ndarray
    object_label: np.ndarray
    centroid_y: np.ndarray
    centroid_x: np.ndarray
    n_skipped_edge: int

    @property
    def n_kept(self) -> int:
        return int(self.tiles.shape[0])


def squeeze_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[0] == 1:
        return mask[0]
    if mask.ndim != 2:
        raise ValueError(f"Expected (H,W) or (1,H,W) mask, got {mask.shape}")
    return mask


def window_inside(y0: int, y1: int, x0: int, x1: int, height: int, width: int) -> bool:
    return y0 >= 0 and x0 >= 0 and y1 <= height and x1 <= width and y1 > y0 and x1 > x0


def _empty(image: np.ndarray, crop_size: int, n_skipped: int) -> InstanceCrops:
    c = image.shape[0]
    return InstanceCrops(
        tiles=np.zeros((0, c, crop_size, crop_size), dtype=image.dtype),
        object_label=np.zeros((0,), dtype=np.int32),
        centroid_y=np.zeros((0,), dtype=np.float32),
        centroid_x=np.zeros((0,), dtype=np.float32),
        n_skipped_edge=n_skipped,
    )


def resize_square(patch: np.ndarray, size: int) -> np.ndarray:
    """Pad a CHW patch to square (zeros around the crop), then resize to size×size."""
    _c, height, width = patch.shape
    side = max(height, width)
    if side == height and side == width and side == size:
        return patch
    canvas = patch
    if height != side or width != side:
        canvas = np.zeros((_c, side, side), dtype=patch.dtype)
        y0 = (side - height) // 2
        x0 = (side - width) // 2
        canvas[:, y0 : y0 + height, x0 : x0 + width] = patch
    if side == size:
        return canvas
    zoom_xy = size / float(side)
    out = ndimage.zoom(canvas, (1.0, zoom_xy, zoom_xy), order=1)
    oh, ow = out.shape[1], out.shape[2]
    if oh == size and ow == size:
        return out.astype(patch.dtype, copy=False)
    result = np.zeros((_c, size, size), dtype=patch.dtype)
    hh, ww = min(size, oh), min(size, ow)
    result[:, :hh, :ww] = out[:, :hh, :ww]
    return result


def resize_tiles(tiles: np.ndarray, size: int) -> np.ndarray:
    """Resize an (N,C,H,W) stack to size×size. No-op when already that shape."""
    if tiles.ndim != 4:
        raise ValueError(f"Expected (N,C,H,W), got {tiles.shape}")
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    n_tiles, n_channels, height, width = tiles.shape
    if height == size and width == size:
        return tiles
    if n_tiles == 0:
        return np.zeros((0, n_channels, size, size), dtype=tiles.dtype)
    return np.stack([resize_square(tile, size) for tile in tiles], axis=0)


def _instance_geometry(
    mask: np.ndarray,
) -> list[tuple[int, float, float, tuple[int, int, int, int]]]:
    """(label, cy, cx, (y0, y1, x0, x1)) for each nonzero instance."""
    objects = ndimage.find_objects(mask)
    out: list[tuple[int, float, float, tuple[int, int, int, int]]] = []
    for i, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        y0, y1 = int(sl[0].start), int(sl[0].stop)
        x0, x1 = int(sl[1].start), int(sl[1].stop)
        region = mask[y0:y1, x0:x1] == i
        ys, xs = np.nonzero(region)
        if ys.size == 0:
            continue
        cy = y0 + float(ys.mean())
        cx = x0 + float(xs.mean())
        out.append((i, cy, cx, (y0, y1, x0, x1)))
    return out


def crop_cells_fixed(image: np.ndarray, mask: np.ndarray, crop_size: int) -> InstanceCrops:
    """Centroid-centered square crops. Skip windows that are not fully inside the FOV."""
    if image.ndim != 3:
        raise ValueError(f"Expected (C,H,W) image, got {image.shape}")
    mask = squeeze_mask(mask)
    _c, height, width = image.shape
    if mask.shape != (height, width):
        raise ValueError(f"Mask {mask.shape} does not match image spatial size {(height, width)}")
    half = crop_size // 2
    tiles = []
    labels = []
    cys = []
    cxs = []
    skipped = 0
    for label, cy, cx, _bbox in _instance_geometry(mask):
        icy = int(round(cy))
        icx = int(round(cx))
        y0 = icy - half
        x0 = icx - half
        y1 = y0 + crop_size
        x1 = x0 + crop_size
        if not window_inside(y0, y1, x0, x1, height, width):
            skipped += 1
            continue
        tiles.append(image[:, y0:y1, x0:x1])
        labels.append(label)
        cys.append(cy)
        cxs.append(cx)
    if not tiles:
        return _empty(image, crop_size, skipped)
    return InstanceCrops(
        tiles=np.stack(tiles, axis=0),
        object_label=np.asarray(labels, dtype=np.int32),
        centroid_y=np.asarray(cys, dtype=np.float32),
        centroid_x=np.asarray(cxs, dtype=np.float32),
        n_skipped_edge=skipped,
    )


def crop_cells_bbox(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    margin: int = 16,
) -> InstanceCrops:
    """BBox plus margin. Skip windows that leave the FOV; pad/resize kept patches to crop_size."""
    if image.ndim != 3:
        raise ValueError(f"Expected (C,H,W) image, got {image.shape}")
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    mask = squeeze_mask(mask)
    _c, height, width = image.shape
    if mask.shape != (height, width):
        raise ValueError(f"Mask {mask.shape} does not match image spatial size {(height, width)}")
    tiles = []
    labels = []
    cys = []
    cxs = []
    skipped = 0
    for label, cy, cx, (y0, y1, x0, x1) in _instance_geometry(mask):
        y0 -= margin
        y1 += margin
        x0 -= margin
        x1 += margin
        if not window_inside(y0, y1, x0, x1, height, width):
            skipped += 1
            continue
        patch = image[:, y0:y1, x0:x1]
        tiles.append(resize_square(patch, crop_size))
        labels.append(label)
        cys.append(cy)
        cxs.append(cx)
    if not tiles:
        return _empty(image, crop_size, skipped)
    return InstanceCrops(
        tiles=np.stack(tiles, axis=0),
        object_label=np.asarray(labels, dtype=np.int32),
        centroid_y=np.asarray(cys, dtype=np.float32),
        centroid_x=np.asarray(cxs, dtype=np.float32),
        n_skipped_edge=skipped,
    )
