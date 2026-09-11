"""JUMP-lite Cellpose instance masks on public CPG S3 (one Zarr v3 array per site)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from botocore.exceptions import ClientError

from jumpbench.config import load_data_config
from jumpbench.data.metadata import (
    JOIN_WELL,
    attach_site_keys,
    filter_sites,
    filter_wells,
    load_perturbations,
    load_sites,
)
from jumpbench.data.s3util import GALLERY_BUCKET, read_s3_bytes
from jumpbench.profiles.cellprofiler import filter_crispr_wells

MASK_OBJECTS = ("cells", "nuclei")
MASK_CODECS = ("jpegxl_lossy_mq", "zstd")
DEFAULT_MASK_CODEC = "jpegxl_lossy_mq"
CHUNK_KEY = "c/0/0/0"
META_KEY = "zarr.json"


def _object_relpath(object_type: str, cfg: dict[str, Any] | None = None) -> str:
    if object_type not in MASK_OBJECTS:
        raise ValueError(f"object_type must be one of {MASK_OBJECTS}, got {object_type!r}")
    gallery = (cfg or load_data_config())["cellpainting_gallery"]
    key = "masks_cell" if object_type == "cells" else "masks_nuclei"
    return str(gallery[key])


def mask_store_prefix(
    object_type: str = "cells",
    codec: str = DEFAULT_MASK_CODEC,
    cfg: dict[str, Any] | None = None,
) -> str:
    """S3 key prefix of the Zarr store (no trailing slash)."""
    if codec not in MASK_CODECS:
        raise ValueError(f"codec must be one of {MASK_CODECS}, got {codec!r}")
    cfg = cfg or load_data_config()
    gallery = cfg["cellpainting_gallery"]
    root = str(gallery["jump_lite_root"]).rstrip("/")
    rel = _object_relpath(object_type, cfg).strip("/")
    return f"{root}/{rel}/{codec}.zarr"


def mask_s3_keys(
    site_key: str,
    object_type: str = "cells",
    codec: str = DEFAULT_MASK_CODEC,
    cfg: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Object keys for one site array: metadata JSON and the single chunk."""
    prefix = mask_store_prefix(object_type, codec, cfg)
    base = f"{prefix}/{site_key}"
    return {"metadata": f"{base}/{META_KEY}", "chunk": f"{base}/{CHUNK_KEY}"}


def mask_sites(
    subset: str = "crispr",
    *,
    max_wells: int | None = None,
    sources: list[str] | None = None,
    plates: list[str] | None = None,
    site_keys: list[str] | None = None,
) -> pl.DataFrame:
    """Frozen JUMP-lite 4-site keys, optionally restricted to CRISPR wells."""
    sites = attach_site_keys(load_sites())
    if subset == "crispr":
        wells = filter_crispr_wells(load_perturbations())
        wells = filter_wells(
            wells,
            max_wells=max_wells,
            sources=sources,
            plates=plates,
            site_keys=site_keys,
        )
        out = sites.join(wells.select(JOIN_WELL), on=JOIN_WELL, how="inner")
        if site_keys:
            out = out.filter(pl.col("Metadata_Site_Key").is_in(list(site_keys)))
        return out
    if subset == "all":
        return filter_sites(
            sites,
            max_wells=max_wells,
            sources=sources,
            plates=plates,
            site_keys=site_keys,
        )
    raise ValueError(f"subset must be 'crispr' or 'all', got {subset!r}")


def _is_missing(exc: BaseException) -> bool:
    if not isinstance(exc, ClientError):
        return False
    error = exc.response.get("Error", {}) if getattr(exc, "response", None) else {}
    code = str(error.get("Code", ""))
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404


def _require_zarr():
    try:
        import zarr
        from zarr.core.buffer.cpu import Buffer
        from zarr.storage import MemoryStore
    except ImportError as exc:
        raise ImportError(
            "Reading JUMP-lite masks needs Zarr v3. Install with pip install -e '.[images]'."
        ) from exc
    return zarr, Buffer, MemoryStore


def decode_mask_array(metadata: bytes, chunk: bytes) -> np.ndarray:
    """Open a one-chunk Zarr v3 array from raw object bytes. Returns (H, W) uint16."""
    zarr, Buffer, MemoryStore = _require_zarr()
    store = MemoryStore(
        {
            META_KEY: Buffer.from_bytes(metadata),
            CHUNK_KEY: Buffer.from_bytes(chunk),
        }
    )
    array = np.asarray(zarr.open_array(store))
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D label image, got shape {array.shape}")
    return array.astype(np.uint16, copy=False)


def load_mask(
    site_key: str,
    object_type: str = "cells",
    codec: str = DEFAULT_MASK_CODEC,
    cfg: dict[str, Any] | None = None,
    client=None,
) -> np.ndarray | None:
    """Fetch one site mask from CPG. Returns (H, W) uint16 or None if the array is missing."""
    keys = mask_s3_keys(site_key, object_type=object_type, codec=codec, cfg=cfg)
    try:
        metadata = read_s3_bytes(GALLERY_BUCKET, keys["metadata"], client=client)
        chunk = read_s3_bytes(GALLERY_BUCKET, keys["chunk"], client=client)
    except ClientError as exc:
        if _is_missing(exc):
            return None
        raise
    return decode_mask_array(metadata, chunk)
