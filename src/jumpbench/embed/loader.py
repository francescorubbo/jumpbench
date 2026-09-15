"""Load one site as (C,H,W) uint16 from local files or streamed CPG objects."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import tifffile
from botocore.exceptions import ClientError

from jumpbench.config import load_data_config
from jumpbench.data.images import CHANNEL_FILES, load_site_images
from jumpbench.data.s3util import GALLERY_BUCKET, read_s3_bytes

IMAGE_SOURCES = ("local", "s3", "s3_mq")
S3_SOURCES = ("s3", "s3_mq")
MQ_META_KEY = ".zarray"
MQ_CHUNK_KEY = "0.0.0"


def decode_tiff_bytes(payload: bytes) -> np.ndarray:
    return np.asarray(tifffile.imread(BytesIO(payload)))


def mq_store_prefix(cfg: dict[str, Any] | None = None) -> str:
    """S3 key prefix of JUMP-lite jpegxl_lossy_mq.zarr (no trailing slash)."""
    gallery = (cfg or load_data_config())["cellpainting_gallery"]
    root = str(gallery["jump_lite_root"]).rstrip("/")
    rel = str(gallery["images"]["mq"]).strip("/")
    return f"{root}/{rel}"


def mq_store_uri(cfg: dict[str, Any] | None = None) -> str:
    return f"s3://{GALLERY_BUCKET}/{mq_store_prefix(cfg)}"


def mq_site_keys(site_key: str, cfg: dict[str, Any] | None = None) -> dict[str, str]:
    base = f"{mq_store_prefix(cfg)}/{site_key}"
    return {"metadata": f"{base}/{MQ_META_KEY}", "chunk": f"{base}/{MQ_CHUNK_KEY}"}


def _is_missing(exc: BaseException) -> bool:
    if not isinstance(exc, ClientError):
        return False
    error = exc.response.get("Error", {}) if getattr(exc, "response", None) else {}
    code = str(error.get("Code", ""))
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404


def decode_jump_lite_mq_array(zarray: bytes, chunk: bytes) -> np.ndarray:
    """Decode a one-chunk Zarr v2 JPEG XL site array from object bytes."""
    import imagecodecs

    meta = json.loads(zarray)
    decoded = np.asarray(imagecodecs.jpegxl_decode(chunk))
    expected = tuple(meta["shape"])
    if decoded.shape != expected:
        decoded = decoded.astype(np.dtype(meta["dtype"]), copy=False).reshape(expected)
    if decoded.dtype != np.uint16:
        decoded = decoded.astype(np.uint16, copy=False)
    if decoded.ndim != 3:
        raise ValueError(f"Expected (C,H,W) JUMP-lite array, got shape {decoded.shape}")
    return decoded


class S3JumpLiteMqLoader:
    """GET JUMP-lite MQ zarr sites from CPG; never write them to disk."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        self._cfg = cfg

    def close(self) -> None:
        return None

    def load(self, site_key: str) -> np.ndarray:
        keys = mq_site_keys(site_key, self._cfg)
        try:
            zarray = read_s3_bytes(GALLERY_BUCKET, keys["metadata"])
            chunk = read_s3_bytes(GALLERY_BUCKET, keys["chunk"])
        except ClientError as exc:
            if _is_missing(exc):
                raise FileNotFoundError(f"No JUMP-lite MQ zarr array for {site_key}") from exc
            raise
        return decode_jump_lite_mq_array(zarray, chunk)


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

    def __init__(self, index: pl.DataFrame, channel_jobs: int = 5):
        self._keys = uri_map(index)
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(channel_jobs)), thread_name_prefix="s3ch"
        )

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    def _get_channel(self, s3_key: str) -> np.ndarray:
        return decode_tiff_bytes(read_s3_bytes(GALLERY_BUCKET, s3_key))

    def load(
        self,
        site_key: str,
        channels: tuple[str, ...] = CHANNEL_FILES,
    ) -> np.ndarray:
        chmap = self._keys.get(site_key)
        if chmap is None:
            raise FileNotFoundError(f"No Orig TIFF URI for {site_key}")
        keys: list[str] = []
        for channel in channels:
            key = chmap.get(channel)
            if not key:
                raise FileNotFoundError(f"No Orig TIFF URI for {site_key} {channel}")
            keys.append(key)
        if len(keys) == 1:
            arrays = [self._get_channel(keys[0])]
        else:
            arrays = list(self._pool.map(self._get_channel, keys))
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
) -> tuple[Callable[[str], np.ndarray], Callable[[], None]]:
    if image_source not in IMAGE_SOURCES:
        raise ValueError(f"image_source must be one of {IMAGE_SOURCES}, got {image_source!r}")
    if image_source == "local":
        root = Path(images_root)
        load_local = local_load or load_site_images

        def _local(site_key: str) -> np.ndarray:
            return load_local(root, site_key)

        return _local, lambda: None
    if image_source == "s3_mq":
        loader = S3JumpLiteMqLoader()
        return loader.load, loader.close
    if index is None:
        raise ValueError("S3 image_source requires a TIFF URI index")
    loader = S3TiffLoader(index)
    return loader.load, loader.close


def iter_loaded_sites(
    keys: Sequence[str],
    load_fn: Callable[[str], np.ndarray],
    *,
    prefetch: int = 1,
    on_missing: Callable[[str], None] | None = None,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (site_key, image), overlapping loads when prefetch > 1.

    When ``on_missing`` is set, ``FileNotFoundError`` from ``load_fn`` is
    reported via that callback and the site is skipped (CPG MQ/TIFF holes).
    """
    if prefetch < 1:
        raise ValueError(f"prefetch must be >= 1, got {prefetch}")

    def _handle_missing(key: str, exc: FileNotFoundError) -> None:
        if on_missing is None:
            raise exc
        on_missing(key)

    if prefetch == 1 or len(keys) <= 1:
        for key in keys:
            try:
                image = load_fn(key)
            except FileNotFoundError as exc:
                _handle_missing(key, exc)
                continue
            yield key, image
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
                try:
                    image = fut.result()
                except FileNotFoundError as exc:
                    _handle_missing(key, exc)
                    _submit()
                    continue
                _submit()
                yield key, image
