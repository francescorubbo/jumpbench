from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from jumpbench.config import load_data_config
from jumpbench.data.metadata import parse_site_key

CHANNEL_FILES = ("AGP", "DNA", "ER", "Mito", "RNA")


def tiff_stem(site_key: str, channel: str) -> str:
    return f"{site_key}__{channel}.tif"


def tiff_path(images_root: Path, site_key: str, channel: str) -> Path:
    return Path(images_root) / tiff_stem(site_key, channel)


def load_site_tiffs(
    images_root: Path,
    site_key: str,
    channels: tuple[str, ...] = CHANNEL_FILES,
) -> np.ndarray:
    """Load a site as uint16 array with shape (C, H, W) in JUMP-lite channel order."""
    arrays = []
    for channel in channels:
        path = tiff_path(images_root, site_key, channel)
        if not path.exists():
            raise FileNotFoundError(path)
        arrays.append(np.asarray(tifffile.imread(path)))
    stacked = np.stack(arrays, axis=0)
    if stacked.dtype != np.uint16:
        stacked = stacked.astype(np.uint16, copy=False)
    return stacked


def list_local_sites(images_root: Path) -> list[str]:
    keys = set()
    for path in Path(images_root).glob("*.tif"):
        stem = path.stem
        if "__" not in stem:
            continue
        site, _channel = stem.rsplit("__", 1)
        try:
            parse_site_key(site)
        except ValueError:
            continue
        keys.add(site)
    return sorted(keys)


def default_images_root() -> Path:
    from jumpbench.paths import resolve

    cfg = load_data_config()
    return resolve(cfg["layout"]["images"])
