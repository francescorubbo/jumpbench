"""Load one site as (C,H,W) uint16 from local files or streamed Orig TIFFs."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from io import BytesIO
from pathlib import Path

import numpy as np
import polars as pl
import tifffile

from jumpbench.data.images import CHANNEL_FILES, load_site_images
from jumpbench.data.s3util import GALLERY_BUCKET, read_s3_bytes

IMAGE_SOURCES = ("local", "s3")


def decode_tiff_bytes(payload: bytes) -> np.ndarray:
    return np.asarray(tifffile.imread(BytesIO(payload)))


def uri_map(index: pl.DataFrame) -> dict[str, dict[str, str]]:
    """site_key -> {channel: s3_key}."""
    frame = index
    if "s3_key" not in frame.columns:
        if "uri" not in frame.columns:
            raise ValueError("TIFF index needs s3_key or uri")
        frame = frame.with_columns(
            pl.col("uri").str.replace(r"^s3://cellpainting-gallery/", "").alias("s3_key")
        )
    out: dict[str, dict[str, str]] = {}
    for row in frame.select(["Metadata_Site_Key", "channel", "s3_key"]).iter_rows(named=True):
        out.setdefault(str(row["Metadata_Site_Key"]), {})[str(row["channel"])] = str(row["s3_key"])
    return out


class S3TiffLoader:
    """GET Orig TIFFs from Cell Painting Gallery; never write them to disk."""

    def __init__(self, index: pl.DataFrame):
        self._keys = uri_map(index)

    def load(
        self,
        site_key: str,
        channels: tuple[str, ...] = CHANNEL_FILES,
    ) -> np.ndarray:
        chmap = self._keys.get(site_key)
        if chmap is None:
            raise FileNotFoundError(f"No Orig TIFF URI for {site_key}")
        arrays = []
        for channel in channels:
            key = chmap.get(channel)
            if not key:
                raise FileNotFoundError(f"No Orig TIFF URI for {site_key} {channel}")
            arrays.append(decode_tiff_bytes(read_s3_bytes(GALLERY_BUCKET, key)))
        stacked = np.stack(arrays, axis=0)
        if stacked.dtype != np.uint16:
            stacked = stacked.astype(np.uint16, copy=False)
        return stacked


def site_load_fn(
    image_source: str,
    images_root: Path,
    index: pl.DataFrame | None = None,
    *,
    local_load: Callable[[Path, str], np.ndarray] | None = None,
) -> Callable[[str], np.ndarray]:
    if image_source not in IMAGE_SOURCES:
        raise ValueError(f"image_source must be one of {IMAGE_SOURCES}, got {image_source!r}")
    if image_source == "local":
        root = Path(images_root)
        load_local = local_load or load_site_images

        def _local(site_key: str) -> np.ndarray:
            return load_local(root, site_key)

        return _local
    if index is None:
        raise ValueError("S3 image_source requires a TIFF URI index")
    return S3TiffLoader(index).load


def iter_loaded_sites(
    keys: Sequence[str],
    load_fn: Callable[[str], np.ndarray],
    *,
    prefetch: int = 1,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (site_key, image), overlapping loads when prefetch > 1."""
    if prefetch < 1:
        raise ValueError(f"prefetch must be >= 1, got {prefetch}")
    if prefetch == 1 or len(keys) <= 1:
        for key in keys:
            yield key, load_fn(key)
        return
    workers = min(prefetch, len(keys))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator = iter(keys)
        inflight: dict = {}

        def _submit() -> None:
            try:
                nxt = next(iterator)
            except StopIteration:
                return
            inflight[pool.submit(load_fn, nxt)] = nxt

        for _ in range(workers):
            _submit()
        while inflight:
            finished, _pending = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in finished:
                key = inflight.pop(fut)
                image = fut.result()
                _submit()
                yield key, image
