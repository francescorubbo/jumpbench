"""JUMP-lite Cellpose instance masks on public CPG S3 (one Zarr v3 array per site)."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from botocore.exceptions import ClientError
from tqdm import tqdm

from jumpbench.config import load_data_config
from jumpbench.data.metadata import (
    JOIN_WELL,
    attach_site_keys,
    filter_sites,
    filter_wells,
    load_perturbations,
    load_sites,
    parse_site_key,
)
from jumpbench.data.s3util import GALLERY_BUCKET, read_s3_bytes, write_bytes_atomic
from jumpbench.paths import resolve
from jumpbench.profiles.cellprofiler import filter_crispr_wells

MASK_OBJECTS = ("cells", "nuclei")
MASK_CODECS = ("jpegxl_lossy_mq", "zstd")
DEFAULT_MASK_CODEC = "jpegxl_lossy_mq"
CHUNK_KEY = "c/0/0/0"
META_KEY = "zarr.json"
MISSING_NAME = ".missing"


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


def default_masks_root(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_data_config()
    return resolve(cfg["layout"].get("masks", "data/masks"))


def mask_cache_dir(
    site_key: str,
    object_type: str = "cells",
    codec: str = DEFAULT_MASK_CODEC,
    cache_root: Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> Path:
    """Local directory for one site array: ``{root}/{object}/{codec}/source/batch/plate/well/site``."""
    if object_type not in MASK_OBJECTS:
        raise ValueError(f"object_type must be one of {MASK_OBJECTS}, got {object_type!r}")
    if codec not in MASK_CODECS:
        raise ValueError(f"codec must be one of {MASK_CODECS}, got {codec!r}")
    parsed = parse_site_key(site_key)
    root = Path(cache_root) if cache_root is not None else default_masks_root(cfg)
    return (
        root
        / object_type
        / codec
        / parsed["source"]
        / parsed["batch"]
        / parsed["plate"]
        / parsed["well"]
        / parsed["site"]
    )


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
    batches: list[str] | None = None,
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
            batches=batches,
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
            batches=batches,
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


def _cache_files(cache_dir: Path) -> tuple[Path, Path, Path]:
    return cache_dir / META_KEY, cache_dir / CHUNK_KEY, cache_dir / MISSING_NAME


def _cache_status(cache_dir: Path) -> str:
    """``complete`` (bytes on disk), ``missing`` (cached 404), or ``absent``."""
    meta_path, chunk_path, missing_path = _cache_files(cache_dir)
    if meta_path.exists() and chunk_path.exists():
        return "complete"
    if missing_path.exists():
        return "missing"
    return "absent"


def _read_cached_mask(cache_dir: Path) -> tuple[bool, np.ndarray | None]:
    """``(True, array|None)`` on a complete cache entry; ``(False, None)`` on a miss."""
    status = _cache_status(cache_dir)
    if status == "missing":
        return True, None
    if status == "complete":
        meta_path, chunk_path, _missing = _cache_files(cache_dir)
        return True, decode_mask_array(meta_path.read_bytes(), chunk_path.read_bytes())
    return False, None


def _write_cached_mask(cache_dir: Path, metadata: bytes, chunk: bytes) -> None:
    meta_path, chunk_path, missing_path = _cache_files(cache_dir)
    write_bytes_atomic(meta_path, metadata)
    write_bytes_atomic(chunk_path, chunk)
    if missing_path.exists():
        missing_path.unlink()


def _write_cached_missing(cache_dir: Path) -> None:
    _missing = _cache_files(cache_dir)[2]
    write_bytes_atomic(_missing, b"")


def load_mask(
    site_key: str,
    object_type: str = "cells",
    codec: str = DEFAULT_MASK_CODEC,
    cfg: dict[str, Any] | None = None,
    client=None,
    *,
    cache: bool = True,
    cache_root: Path | None = None,
) -> np.ndarray | None:
    """Load one site mask. Hits ``data/masks/`` first; on a miss fetches CPG and caches.

    Returns (H, W) uint16 or None if the array is missing on S3 (404s are cached too).
    """
    cache_dir: Path | None = None
    if cache:
        cache_dir = mask_cache_dir(
            site_key, object_type=object_type, codec=codec, cache_root=cache_root, cfg=cfg
        )
        hit, cached = _read_cached_mask(cache_dir)
        if hit:
            return cached

    keys = mask_s3_keys(site_key, object_type=object_type, codec=codec, cfg=cfg)
    try:
        metadata = read_s3_bytes(GALLERY_BUCKET, keys["metadata"], client=client)
        chunk = read_s3_bytes(GALLERY_BUCKET, keys["chunk"], client=client)
    except ClientError as exc:
        if _is_missing(exc):
            if cache_dir is not None:
                _write_cached_missing(cache_dir)
            return None
        raise
    if cache_dir is not None:
        _write_cached_mask(cache_dir, metadata, chunk)
    return decode_mask_array(metadata, chunk)


def cache_mask(
    site_key: str,
    object_type: str = "cells",
    codec: str = DEFAULT_MASK_CODEC,
    cfg: dict[str, Any] | None = None,
    client=None,
    *,
    cache_root: Path | None = None,
) -> str:
    """Ensure one site is in ``data/masks/``. Does not decode.

    Returns ``hit`` (already complete), ``missing`` (cached 404), or ``fetched``.
    """
    cache_dir = mask_cache_dir(
        site_key, object_type=object_type, codec=codec, cache_root=cache_root, cfg=cfg
    )
    status = _cache_status(cache_dir)
    if status == "complete":
        return "hit"
    if status == "missing":
        return "missing"
    keys = mask_s3_keys(site_key, object_type=object_type, codec=codec, cfg=cfg)
    try:
        metadata = read_s3_bytes(GALLERY_BUCKET, keys["metadata"], client=client)
        chunk = read_s3_bytes(GALLERY_BUCKET, keys["chunk"], client=client)
    except ClientError as exc:
        if _is_missing(exc):
            _write_cached_missing(cache_dir)
            return "missing"
        raise
    _write_cached_mask(cache_dir, metadata, chunk)
    return "fetched"


def cache_masks(
    site_keys: Iterable[str],
    object_type: str = "cells",
    codec: str = DEFAULT_MASK_CODEC,
    cfg: dict[str, Any] | None = None,
    *,
    cache_root: Path | None = None,
    jobs: int = 16,
    show_progress: bool = True,
) -> dict[str, int]:
    """Prefetch Cellpose masks in parallel so embed is not blocked on S3."""
    keys = list(dict.fromkeys(site_keys))
    todo: list[str] = []
    n_hit = 0
    n_missing = 0
    for key in keys:
        cache_dir = mask_cache_dir(
            key, object_type=object_type, codec=codec, cache_root=cache_root, cfg=cfg
        )
        status = _cache_status(cache_dir)
        if status == "complete":
            n_hit += 1
        elif status == "missing":
            n_missing += 1
        else:
            todo.append(key)

    n_fetched = 0
    workers = max(1, int(jobs))
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    cache_mask,
                    key,
                    object_type,
                    codec,
                    cfg,
                    cache_root=cache_root,
                )
                for key in todo
            ]
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="cache-masks", file=sys.stderr)
            for fut in iterator:
                result = fut.result()
                if result == "fetched":
                    n_fetched += 1
                elif result == "missing":
                    n_missing += 1
                else:
                    n_hit += 1
    return {
        "n": len(keys),
        "hit": n_hit,
        "fetched": n_fetched,
        "missing": n_missing,
    }
