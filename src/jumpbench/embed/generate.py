from __future__ import annotations

import sys
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from tqdm import tqdm

from jumpbench.config import channel_indices, resolve_model
from jumpbench.data.images import iter_local_sites, load_site_images, site_has_images
from jumpbench.data.masks import DEFAULT_MASK_CODEC, cache_masks, load_mask, mask_sites
from jumpbench.data.metadata import parse_site_key, well_id_from_site_key
from jumpbench.embed.backends import build_backend
from jumpbench.embed.crops import crop_cells_bbox, crop_cells_fixed, resize_tiles
from jumpbench.embed.preprocess import apply_preprocess, apply_preprocess_tiles
from jumpbench.embed.preview import montage_rgb, write_png
from jumpbench.embed.tiling import crop_tiles, reorder_channels, select_channels
from jumpbench.provenance import now_iso, write_json

CROP_MODES = ("grid", "cell_fixed", "cell_bbox")
CELL_CROPS = ("cell_fixed", "cell_bbox")


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


def iter_embed_sites(
    images_root: Path,
    crop: str,
    *,
    site_keys: Iterable[str] | None = None,
    subset: str | None = None,
    max_wells: int | None = None,
) -> Iterator[str]:
    """Yield sites that should be embedded. Cell crops probe paths; they do not walk the image tree."""
    images_root = Path(images_root)
    if crop == "grid":
        if site_keys is not None:
            yield from site_keys
            return
        _status(f"Scanning local sites under {images_root} (streaming, no pre-count)...")
        n = 0
        for key in iter_local_sites(images_root):
            n += 1
            if n == 1 or n % 1000 == 0:
                _status(f"  found {n} local sites...")
            yield key
        _status(f"Found {n} local sites")
        return

    subset = subset or "crispr"
    _status(f"Loading {subset} 4-site keys (no image-tree scan)...")
    masked = mask_sites(
        subset=subset,
        max_wells=max_wells,
        site_keys=list(site_keys) if site_keys else None,
    )
    candidates = masked["Metadata_Site_Key"].to_list()
    if site_keys is not None:
        wanted = set(site_keys)
        candidates = [key for key in candidates if key in wanted]
    _status(f"{len(candidates)} mask sites; probing which have local images...")
    n_checked = 0
    n_hit = 0
    for key in candidates:
        n_checked += 1
        if n_checked == 1 or n_checked % 500 == 0:
            _status(f"  probed {n_checked}/{len(candidates)} ({n_hit} local so far)")
        if site_has_images(images_root, key):
            n_hit += 1
            if n_hit == 1:
                _status(f"  first local site: {key}")
            yield key
    _status(f"{n_hit} of {len(candidates)} mask sites have local images")


def resolve_embed_sites(
    images_root: Path,
    crop: str,
    *,
    site_keys: Iterable[str] | None = None,
    subset: str | None = None,
    max_wells: int | None = None,
) -> list[str]:
    return list(
        iter_embed_sites(
            images_root,
            crop,
            site_keys=site_keys,
            subset=subset,
            max_wells=max_wells,
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
    max_wells: int | None = None,
    mask_codec: str = DEFAULT_MASK_CODEC,
    dry_run: bool = False,
    preview_n: int = 32,
) -> Path:
    if crop not in CROP_MODES:
        raise ValueError(f"crop must be one of {CROP_MODES}, got {crop!r}")
    card = resolve_model(model, models_cfg)
    card["crop"] = crop
    card["mask_object"] = mask_object
    card["crop_margin"] = int(crop_margin)
    card["mask_codec"] = mask_codec
    card["crop_size"] = resolve_crop_size({**card, "crop_size": crop_size})
    images_root = Path(images_root)
    run_dir = _run_dir(output_dir, card, codec, crop, mask_object)
    run_dir.mkdir(parents=True, exist_ok=True)

    mode = "dry-run preview" if dry_run else "embed"
    _status(
        f"{mode}: model={card['name']} crop={crop} crop_size={card['crop_size']} "
        f"tile_size={card['tile_size']} object={mask_object} images={images_root}"
    )
    site_iter = iter_embed_sites(
        images_root,
        crop,
        site_keys=site_keys,
        subset=subset,
        max_wells=max_wells,
    )
    if dry_run:
        return preview_crops(card, images_root, site_iter, run_dir / "preview", preview_n=preview_n)

    keys = list(site_iter)
    if not keys:
        raise FileNotFoundError(f"No sites found under {images_root}")

    if crop in CELL_CROPS:
        jobs = int(card.get("runtime", {}).get("num_workers") or 16)
        _status(
            f"Caching {len(keys)} {mask_object} masks ({mask_codec}) before embed (jobs={jobs})"
        )
        cached = cache_masks(
            keys,
            object_type=mask_object,
            codec=mask_codec,
            jobs=jobs,
        )
        _status(
            f"Mask cache: {cached['hit']} hit, {cached['fetched']} fetched, "
            f"{cached['missing']} missing"
        )

    _status(f"Building backend for {card['name']} ({len(keys)} sites)")
    backend = build_backend(card)
    device = getattr(backend, "device", None)
    if device is not None:
        _status(f"backend device={device}")
    frames = []
    n_skipped_no_mask = 0
    n_skipped_edge = 0
    n_sites_embedded = 0
    for key in tqdm(keys, desc=f"embed:{card['name']}:{crop}"):
        image = load_site_images(images_root, key)
        feats, extra, stats = embed_site(image, card, backend, site_key=key)
        n_skipped_no_mask += stats["n_skipped_no_mask"]
        n_skipped_edge += stats["n_skipped_edge"]
        if feats.shape[0] == 0:
            continue
        n_sites_embedded += 1
        fmt = card.get("aggregation", {}).get("output_format", "wide")
        if fmt == "jump_lite_long":
            frames.append(_long_rows(key, feats, extra, card["name"]))
        else:
            frames.append(_wide_rows(key, feats, extra))

    if not frames:
        raise RuntimeError(
            f"No embeddings produced ({crop}, {len(keys)} candidate sites, "
            f"{n_skipped_no_mask} missing masks)"
        )
    table = pl.concat(frames, how="diagonal")
    out = run_dir / "site_embeddings.parquet"
    table.write_parquet(out)
    write_json(
        run_dir / "provenance.json",
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
                getattr(backend, "device", None) or card.get("runtime", {}).get("device")
            ),
            "crop": crop,
            "crop_margin": int(crop_margin) if crop == "cell_bbox" else None,
            "mask_object": mask_object if crop in CELL_CROPS else None,
            "mask_codec": mask_codec if crop in CELL_CROPS else None,
            "subset": (subset or "crispr") if crop in CELL_CROPS else None,
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
