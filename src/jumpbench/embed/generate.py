from __future__ import annotations

import sys
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from tqdm import tqdm

from jumpbench.config import channel_indices, resolve_model
from jumpbench.data.images import load_site_images
from jumpbench.data.masks import DEFAULT_MASK_CODEC, cache_masks, load_mask
from jumpbench.data.metadata import parse_site_key, well_id_from_site_key
from jumpbench.embed.backends import build_backend
from jumpbench.embed.crops import crop_cells_bbox, crop_cells_fixed, resize_tiles
from jumpbench.embed.preprocess import apply_preprocess, apply_preprocess_tiles
from jumpbench.embed.preview import montage_rgb, write_png
from jumpbench.embed.sites import CELL_CROPS, iter_embed_sites
from jumpbench.embed.tiling import crop_tiles, grid_coverage, reorder_channels, select_channels
from jumpbench.provenance import now_iso, write_json

CROP_MODES = ("grid", "cell_fixed", "cell_bbox")
POOL_MODES = ("crop", "site")
_SHARD_FLUSH = 64


def _status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def resolve_crop_size(card: dict[str, Any]) -> int:
    """Native crop window. Defaults to the model tile_size when --crop-size is omitted."""
    raw = card.get("crop_size")
    size = int(card["tile_size"] if raw is None else raw)
    if size < 1:
        raise ValueError(f"crop_size must be >= 1, got {size}")
    return size


def _embed_tiles(backend, tiles: np.ndarray, card: dict[str, Any]) -> np.ndarray:
    if tiles.shape[0] == 0:
        dim = getattr(backend, "embedding_dim", None) or 0
        return np.zeros((0, int(dim)), dtype=np.float32)
    if str(card.get("preprocess_scope", "site")) == "tile":
        tiles = apply_preprocess_tiles(tiles, card.get("preprocess"))
    target = getattr(backend, "resize_to", int(card["tile_size"]))
    if target is not None:
        tiles = resize_tiles(tiles, int(target))
    return backend.embed_tiles(tiles)


def _prepare_image(image: np.ndarray, card: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    zarr_channels = list(card["zarr_channels"])
    idx = channel_indices(card["channels"], zarr_channels)
    selected = select_channels(image, idx)
    names = list(card["channels"])
    order = card.get("model_channel_order")
    if order:
        selected = reorder_channels(selected, names, order)
        names = list(order)
    if str(card.get("preprocess_scope", "site")) != "tile":
        selected = apply_preprocess(selected, card.get("preprocess"))
    return selected, names


def crop_site(
    image: np.ndarray,
    card: dict[str, Any],
    *,
    site_key: str | None = None,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    """Cut native crops for one site without running a backend or resizing to the model."""
    crop = str(card.get("crop", "grid"))
    if crop not in CROP_MODES:
        raise ValueError(f"crop must be one of {CROP_MODES}, got {crop!r}")
    prepared, _names = _prepare_image(image, card)
    stats = {"n_skipped_edge": 0, "n_skipped_no_mask": 0}
    crop_size = resolve_crop_size(card)
    if crop == "grid":
        tiles, coords = crop_tiles(prepared, crop_size)
        return tiles, {"tile_y": coords[:, 0], "tile_x": coords[:, 1]}, stats

    object_type = str(card.get("mask_object", "cells"))
    codec = str(card.get("mask_codec", DEFAULT_MASK_CODEC))
    loaded = mask
    if loaded is None:
        if not site_key:
            raise ValueError("site_key is required to load a mask from S3")
        loaded = load_mask(site_key, object_type=object_type, codec=codec)
    empty_meta = {"object_label": np.zeros((0,), dtype=np.int32)}
    empty_tiles = np.zeros((0, prepared.shape[0], 1, 1), dtype=prepared.dtype)
    if loaded is None:
        stats["n_skipped_no_mask"] = 1
        return empty_tiles, empty_meta, stats
    if loaded.ndim == 3 and loaded.shape[0] == 1:
        loaded = loaded[0]
    if loaded.shape != prepared.shape[1:]:
        warnings.warn(
            f"Mask shape {loaded.shape} != image {(prepared.shape[1], prepared.shape[2])} "
            f"for {site_key or 'site'}; skipping",
            stacklevel=2,
        )
        stats["n_skipped_no_mask"] = 1
        return empty_tiles, empty_meta, stats

    if crop == "cell_fixed":
        cropped = crop_cells_fixed(prepared, loaded, crop_size)
    else:
        cropped = crop_cells_bbox(
            prepared, loaded, crop_size, margin=int(card.get("crop_margin", 16))
        )
    stats["n_skipped_edge"] = cropped.n_skipped_edge
    meta = {
        "object_label": cropped.object_label,
        "centroid_y": cropped.centroid_y,
        "centroid_x": cropped.centroid_x,
    }
    return cropped.tiles, meta, stats


def embed_site(
    image: np.ndarray,
    card: dict[str, Any],
    backend,
    *,
    site_key: str | None = None,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    """Return (features, per-crop columns, skip stats).

    Grid crops use ``tile_y`` / ``tile_x``. Cell crops use ``object_label`` and
    centroids. ``mask`` injects labels for tests; otherwise cell modes fetch from S3.
    """
    tiles, meta, stats = crop_site(image, card, site_key=site_key, mask=mask)
    feats = _embed_tiles(backend, tiles, card)
    return feats, meta, stats


def _wide_rows(site_key: str, feats: np.ndarray, extra: dict[str, np.ndarray]) -> pl.DataFrame:
    parsed = parse_site_key(site_key)
    n, dim = feats.shape
    data: dict[str, Any] = {f"feat_{i:04d}": feats[:, i] for i in range(dim)}
    data.update({name: extra[name] for name in extra})
    data.update(
        {
            "Metadata_Source": parsed["source"],
            "Metadata_Batch": parsed["batch"],
            "Metadata_Plate": parsed["plate"],
            "Metadata_Well": parsed["well"],
            "Metadata_Site": parsed["site"],
            "Metadata_id": well_id_from_site_key(site_key),
            "site_key": site_key,
        }
    )
    return pl.DataFrame(data)


def _long_rows(
    site_key: str,
    feats: np.ndarray,
    extra: dict[str, np.ndarray],
    model: str,
) -> pl.DataFrame:
    parsed = parse_site_key(site_key)
    n, dim = feats.shape
    if "tile_y" in extra and "tile_x" in extra:
        tile = [f"{int(y)}_{int(x)}" for y, x in zip(extra["tile_y"], extra["tile_x"], strict=True)]
        label = np.repeat(-1, n * dim)
    else:
        labels = extra.get("object_label", np.zeros((n,), dtype=np.int32))
        tile = [str(int(v)) for v in labels]
        label = np.repeat(labels.astype(np.int64), dim)
    records = {
        "tile": np.repeat(tile, dim),
        "label": label,
        "branch": np.repeat("", n * dim),
        "metric": np.tile([f"feat_{i:04d}" for i in range(dim)], n),
        "value": feats.reshape(-1),
        "object": np.repeat(model, n * dim),
        "tp": np.repeat(0, n * dim),
        "filename": np.repeat(site_key, n * dim),
        "Metadata_Source": np.repeat(parsed["source"], n * dim),
        "Metadata_Batch": np.repeat(parsed["batch"], n * dim),
        "Metadata_Plate": np.repeat(parsed["plate"], n * dim),
        "Metadata_Well": np.repeat(parsed["well"], n * dim),
        "Metadata_Site": np.repeat(parsed["site"], n * dim),
    }
    return pl.DataFrame(records)


def resolve_embed_sites(
    images_root: Path,
    crop: str,
    *,
    site_keys: Iterable[str] | None = None,
    subset: str | None = None,
    site_set: str = "all",
    max_wells: int | None = None,
    sources: Iterable[str] | None = None,
    plates: Iterable[str] | None = None,
    batches: Iterable[str] | None = None,
) -> list[str]:
    if crop in CELL_CROPS:
        site_set = "jump_lite"
    return list(
        iter_embed_sites(
            images_root,
            crop,
            site_keys=site_keys,
            subset=subset,
            site_set=site_set,
            max_wells=max_wells,
            sources=sources,
            plates=plates,
            batches=batches,
        )
    )


def _run_dir(
    output_dir: Path, card: dict[str, Any], codec: str, crop: str, mask_object: str
) -> Path:
    crop_size = resolve_crop_size(card)
    extra = "" if crop_size == int(card["tile_size"]) else f"_crop{crop_size}"
    if crop == "grid":
        return Path(output_dir) / card["name"] / f"{codec}{extra}"
    return Path(output_dir) / card["name"] / f"{codec}_{crop}_{mask_object}{extra}"


def _crop_index_rows(site_key: str, extra: dict[str, np.ndarray]) -> pl.DataFrame:
    parsed = parse_site_key(site_key)
    n = next(iter(extra.values())).shape[0] if extra else 0
    data: dict[str, Any] = {name: extra[name] for name in extra}
    data.update(
        {
            "Metadata_Source": np.repeat(parsed["source"], n),
            "Metadata_Batch": np.repeat(parsed["batch"], n),
            "Metadata_Plate": np.repeat(parsed["plate"], n),
            "Metadata_Well": np.repeat(parsed["well"], n),
            "Metadata_Site": np.repeat(parsed["site"], n),
            "site_key": np.repeat(site_key, n),
        }
    )
    return pl.DataFrame(data)


def preview_crops(
    card: dict[str, Any],
    images_root: Path,
    keys: Iterable[str],
    dest: Path,
    *,
    preview_n: int = 32,
    mask: np.ndarray | None = None,
) -> Path:
    """Write a PNG montage and crop index without running an embedding backend."""
    if preview_n < 1:
        raise ValueError(f"preview_n must be >= 1, got {preview_n}")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tiles_acc: list[np.ndarray] = []
    rows: list[pl.DataFrame] = []
    n_skipped_no_mask = 0
    n_skipped_edge = 0
    n_sites = 0
    n_tried = 0
    have = 0
    _status(f"Collecting {preview_n} preview crops ({card.get('crop', 'grid')}, no embeddings)")
    pbar = tqdm(
        total=preview_n,
        desc="preview crops",
        unit="crop",
        file=sys.stderr,
        miniters=1,
        dynamic_ncols=True,
    )
    try:
        for key in keys:
            n_tried += 1
            _status(f"[{have}/{preview_n} crops] site {n_tried}: {key}  (load images)")
            image = load_site_images(images_root, key)
            _status(f"[{have}/{preview_n} crops] site {n_tried}: crop / mask")
            tiles, extra, stats = crop_site(image, card, site_key=key, mask=mask)
            n_skipped_no_mask += stats["n_skipped_no_mask"]
            n_skipped_edge += stats["n_skipped_edge"]
            if tiles.shape[0] == 0:
                why = "no mask" if stats["n_skipped_no_mask"] else "no in-FOV crops"
                _status(f"[{have}/{preview_n} crops] site {n_tried}: skip ({why})")
                continue
            n_sites += 1
            remain = preview_n - have
            take = min(remain, tiles.shape[0])
            tiles_acc.append(tiles[:take])
            clipped = {k: v[:take] for k, v in extra.items()}
            rows.append(_crop_index_rows(key, clipped))
            have += take
            pbar.update(take)
            pbar.set_postfix(sites=n_tried, no_mask=n_skipped_no_mask, refresh=False)
            _status(
                f"[{have}/{preview_n} crops] site {n_tried}: kept {take} (site has {tiles.shape[0]})"
            )
            if have >= preview_n:
                break
    finally:
        pbar.close()
    if not tiles_acc:
        raise RuntimeError(
            f"No crops to preview ({n_tried} candidate sites, {n_skipped_no_mask} missing masks)"
        )
    _status(f"Writing montage of {have} crops from {n_sites} sites -> {dest}")
    stack = np.concatenate(tiles_acc, axis=0)
    montage_path = write_png(dest / "montage.png", montage_rgb(stack))
    index_path = dest / "crops.parquet"
    pl.concat(rows, how="diagonal").write_parquet(index_path)
    write_json(
        dest / "provenance.json",
        {
            "created_at": now_iso(),
            "dry_run": True,
            "model": card["name"],
            "crop": card.get("crop", "grid"),
            "crop_margin": card.get("crop_margin"),
            "mask_object": card.get("mask_object"),
            "crop_size": resolve_crop_size(card),
            "tile_size": card["tile_size"],
            "n_preview_crops": int(stack.shape[0]),
            "n_sites_sampled": n_sites,
            "n_sites_tried": n_tried,
            "n_sites_skipped_no_mask": n_skipped_no_mask,
            "n_objects_skipped_edge": n_skipped_edge,
            "montage": str(montage_path),
            "crops": str(index_path),
        },
    )
    _status(f"Dry-run done: {montage_path}")
    return montage_path


def _pool_site_features(
    feats: np.ndarray, extra: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = int(feats.shape[0])
    if n == 0:
        return feats, extra
    pooled = np.median(feats, axis=0, keepdims=True).astype(np.float32) if n > 1 else feats
    return pooled, {"n_crops": np.asarray([n], dtype=np.int32)}


def _completed_site_keys(run_dir: Path) -> set[str]:
    done = run_dir / "completed_sites.txt"
    if done.exists():
        return {line for line in done.read_text().splitlines() if line}
    out = run_dir / "site_embeddings.parquet"
    if out.exists():
        table = pl.read_parquet(out, columns=["site_key"])
        return set(table["site_key"].to_list())
    return set()


def _flush_shard(
    frames: list[pl.DataFrame],
    shard_dir: Path,
    shard_index: int,
    completed_path: Path,
    keys: list[str],
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    pl.concat(frames, how="diagonal").write_parquet(shard_dir / f"{shard_index:06d}.parquet")
    with completed_path.open("a", encoding="utf-8") as fh:
        for key in keys:
            fh.write(key + "\n")
        fh.flush()


def _concat_shards(run_dir: Path) -> Path:
    shard_dir = run_dir / "shards"
    parts = sorted(shard_dir.glob("*.parquet")) if shard_dir.is_dir() else []
    out = run_dir / "site_embeddings.parquet"
    tables: list[pl.DataFrame] = []
    if out.exists():
        tables.append(pl.read_parquet(out))
    tables.extend(pl.read_parquet(path) for path in parts)
    if not tables:
        raise RuntimeError(f"No embedding shards under {shard_dir}")
    table = pl.concat(tables, how="diagonal")
    if "site_key" in table.columns:
        table = table.unique(subset=["site_key"], keep="last")
    table.write_parquet(out)
    return out


def generate_embeddings(
    model: str,
    images_root: Path,
    output_dir: Path,
    site_keys: Iterable[str] | None = None,
    models_cfg: dict[str, Any] | None = None,
    codec: str = "jpegxl_mq",
    crop: str = "grid",
    mask_object: str = "cells",
    crop_margin: int = 16,
    crop_size: int | None = None,
    subset: str | None = None,
    site_set: str = "all",
    max_wells: int | None = None,
    sources: Iterable[str] | None = None,
    plates: Iterable[str] | None = None,
    batches: Iterable[str] | None = None,
    pool: str = "crop",
    run_dir: Path | None = None,
    mask_codec: str = DEFAULT_MASK_CODEC,
    dry_run: bool = False,
    preview_n: int = 32,
) -> Path:
    if crop not in CROP_MODES:
        raise ValueError(f"crop must be one of {CROP_MODES}, got {crop!r}")
    if crop in CELL_CROPS:
        site_set = "jump_lite"
    if pool not in POOL_MODES:
        raise ValueError(f"pool must be one of {POOL_MODES}, got {pool!r}")
    card = resolve_model(model, models_cfg)
    card["crop"] = crop
    card["mask_object"] = mask_object
    card["crop_margin"] = int(crop_margin)
    card["mask_codec"] = mask_codec
    card["crop_size"] = resolve_crop_size({**card, "crop_size": crop_size})
    images_root = Path(images_root)
    dest = (
        Path(run_dir)
        if run_dir is not None
        else _run_dir(output_dir, card, codec, crop, mask_object)
    )
    dest.mkdir(parents=True, exist_ok=True)

    mode = "dry-run preview" if dry_run else "embed"
    _status(
        f"{mode}: model={card['name']} crop={crop} crop_size={card['crop_size']} "
        f"tile_size={card['tile_size']} object={mask_object} sites={site_set} "
        f"subset={subset} pool={pool} images={images_root}"
    )
    site_iter = iter_embed_sites(
        images_root,
        crop,
        site_keys=site_keys,
        subset=subset,
        site_set=site_set,
        max_wells=max_wells,
        sources=sources,
        plates=plates,
        batches=batches,
    )
    if dry_run:
        return preview_crops(card, images_root, site_iter, dest / "preview", preview_n=preview_n)

    keys = list(site_iter)
    if not keys:
        raise FileNotFoundError(f"No sites found under {images_root}")
    completed = _completed_site_keys(dest)
    pending = [key for key in keys if key not in completed]
    _status(f"{len(keys)} candidate sites, {len(completed)} already done, {len(pending)} remaining")

    if crop in CELL_CROPS and pending:
        jobs = int(card.get("runtime", {}).get("num_workers") or 16)
        _status(
            f"Caching {len(pending)} {mask_object} masks ({mask_codec}) before embed (jobs={jobs})"
        )
        cached = cache_masks(
            pending,
            object_type=mask_object,
            codec=mask_codec,
            jobs=jobs,
        )
        _status(
            f"Mask cache: {cached['hit']} hit, {cached['fetched']} fetched, "
            f"{cached['missing']} missing"
        )

    backend = None
    n_skipped_no_mask = 0
    n_skipped_edge = 0
    n_sites_embedded = len(completed)
    n_crops_total = 0
    coverage: dict[str, float | int] | None = None
    embedding_dim: int | None = None
    shard_dir = dest / "shards"
    completed_path = dest / "completed_sites.txt"
    existing_shards = sorted(shard_dir.glob("*.parquet")) if shard_dir.is_dir() else []
    shard_index = len(existing_shards)
    frames: list[pl.DataFrame] = []
    flushed_keys: list[str] = []

    if pending:
        _status(f"Building backend for {card['name']} ({len(pending)} sites to embed)")
        backend = build_backend(card)
        device = getattr(backend, "device", None)
        if device is not None:
            _status(f"backend device={device}")
        embedding_dim = getattr(backend, "embedding_dim", None)

    fmt = card.get("aggregation", {}).get("output_format", "wide")
    for key in tqdm(pending, desc=f"embed:{card['name']}:{crop}"):
        image = load_site_images(images_root, key)
        if coverage is None and crop == "grid":
            coverage = grid_coverage(
                int(image.shape[1]), int(image.shape[2]), int(card["crop_size"])
            )
        feats, extra, stats = embed_site(image, card, backend, site_key=key)
        n_skipped_no_mask += stats["n_skipped_no_mask"]
        n_skipped_edge += stats["n_skipped_edge"]
        if feats.shape[0] == 0:
            continue
        n_crops_total += int(feats.shape[0])
        if embedding_dim is None:
            embedding_dim = int(feats.shape[1])
        if pool == "site":
            feats, extra = _pool_site_features(feats, extra)
        n_sites_embedded += 1
        if fmt == "jump_lite_long":
            frames.append(_long_rows(key, feats, extra, card["name"]))
        else:
            frames.append(_wide_rows(key, feats, extra))
        flushed_keys.append(key)
        if len(frames) >= _SHARD_FLUSH:
            _flush_shard(frames, shard_dir, shard_index, completed_path, flushed_keys)
            shard_index += 1
            frames = []
            flushed_keys = []

    if frames:
        _flush_shard(frames, shard_dir, shard_index, completed_path, flushed_keys)

    if n_sites_embedded == 0:
        raise RuntimeError(
            f"No embeddings produced ({crop}, {len(keys)} candidate sites, "
            f"{n_skipped_no_mask} missing masks)"
        )
    out = _concat_shards(dest)
    mean_crops = (n_crops_total / max(len(pending), 1)) if pending else None
    if mean_crops is None and pool == "site" and out.exists():
        table = pl.scan_parquet(out)
        if "n_crops" in table.collect_schema():
            mean_crops = float(table.select(pl.col("n_crops").mean()).collect().item() or 0)
    write_json(
        dest / "provenance.json",
        {
            "created_at": now_iso(),
            "model": card["name"],
            "channel_recipe": card["channel_recipe"],
            "channels": card["channels"],
            "model_channel_order": card.get("model_channel_order"),
            "crop_size": card["crop_size"],
            "tile_size": card["tile_size"],
            "preprocess": card["preprocess"],
            "preprocess_scope": card.get("preprocess_scope", "site"),
            "checkpoint": card.get("checkpoint"),
            "architecture": card.get("architecture"),
            "pretrained": card.get("pretrained"),
            "device": str(
                getattr(backend, "device", None)
                if backend is not None
                else card.get("runtime", {}).get("device")
            ),
            "crop": crop,
            "crop_margin": int(crop_margin) if crop == "cell_bbox" else None,
            "mask_object": mask_object if crop in CELL_CROPS else None,
            "mask_codec": mask_codec if crop in CELL_CROPS else None,
            "subset": subset if subset is not None else ("crispr" if crop in CELL_CROPS else None),
            "site_set": "jump_lite" if crop in CELL_CROPS else site_set,
            "batches": list(batches) if batches else None,
            "pool": pool,
            "embedding_dim": embedding_dim,
            "n_tiles_mean": mean_crops if crop == "grid" else None,
            "n_cells_mean": mean_crops if crop in CELL_CROPS else None,
            "fov_frac": None if coverage is None else coverage.get("fov_frac"),
            "n_tiles_y": None if coverage is None else coverage.get("n_tiles_y"),
            "n_tiles_x": None if coverage is None else coverage.get("n_tiles_x"),
            "image_height": None if coverage is None else coverage.get("image_height"),
            "image_width": None if coverage is None else coverage.get("image_width"),
            "n_sites": len(keys),
            "n_sites_embedded": n_sites_embedded,
            "n_sites_skipped_no_mask": n_skipped_no_mask,
            "n_objects_skipped_edge": n_skipped_edge,
            "images_root": str(images_root),
            "persist_codec": codec,
            "output": str(out),
            "comparison_note": card.get("notes"),
        },
    )
    return out
