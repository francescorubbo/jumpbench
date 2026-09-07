from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from tqdm import tqdm

from jumpbench.config import channel_indices, resolve_model
from jumpbench.data.images import list_local_sites, load_site_images
from jumpbench.data.metadata import parse_site_key, well_id_from_site_key
from jumpbench.embed.backends import build_backend
from jumpbench.embed.preprocess import apply_preprocess
from jumpbench.embed.tiling import crop_tiles, reorder_channels, select_channels
from jumpbench.provenance import now_iso, write_json


def _prepare_image(image: np.ndarray, card: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    zarr_channels = list(card["zarr_channels"])
    idx = channel_indices(card["channels"], zarr_channels)
    selected = select_channels(image, idx)
    names = list(card["channels"])
    order = card.get("model_channel_order")
    if order:
        selected = reorder_channels(selected, names, order)
        names = list(order)
    return apply_preprocess(selected, card.get("preprocess")), names


def embed_site(image: np.ndarray, card: dict[str, Any], backend) -> tuple[np.ndarray, np.ndarray]:
    prepared, _names = _prepare_image(image, card)
    tiles, coords = crop_tiles(prepared, int(card["tile_size"]))
    # Standardize per tile if the last preprocess op is standard — already applied
    # on the full site. Paper applies standard on crops. Re-apply on tiles when
    # requested so the knob is explicit.
    feats = backend.embed_tiles(tiles)
    return feats, coords


def _wide_rows(site_key: str, feats: np.ndarray, coords: np.ndarray) -> pl.DataFrame:
    parsed = parse_site_key(site_key)
    n, dim = feats.shape
    data = {f"feat_{i:04d}": feats[:, i] for i in range(dim)}
    data.update(
        {
            "tile_y": coords[:, 0],
            "tile_x": coords[:, 1],
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


def _long_rows(site_key: str, feats: np.ndarray, coords: np.ndarray, model: str) -> pl.DataFrame:
    parsed = parse_site_key(site_key)
    n, dim = feats.shape
    tile = [f"{y}_{x}" for y, x in coords]
    records = {
        "tile": np.repeat(tile, dim),
        "label": np.repeat(-1, n * dim),
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


def generate_embeddings(
    model: str,
    images_root: Path,
    output_dir: Path,
    site_keys: Iterable[str] | None = None,
    models_cfg: dict[str, Any] | None = None,
    codec: str = "jpegxl_mq",
) -> Path:
    card = resolve_model(model, models_cfg)
    backend = build_backend(card)
    images_root = Path(images_root)
    output_dir = Path(output_dir) / card["name"] / codec
    output_dir.mkdir(parents=True, exist_ok=True)

    keys = list(site_keys) if site_keys is not None else list_local_sites(images_root)
    if not keys:
        raise FileNotFoundError(f"No sites found under {images_root}")

    frames = []
    for key in tqdm(keys, desc=f"embed:{card['name']}"):
        image = load_site_images(images_root, key)
        feats, coords = embed_site(image, card, backend)
        fmt = card.get("aggregation", {}).get("output_format", "wide")
        if fmt == "jump_lite_long":
            frames.append(_long_rows(key, feats, coords, card["name"]))
        else:
            frames.append(_wide_rows(key, feats, coords))

    table = pl.concat(frames, how="diagonal")
    out = output_dir / "site_embeddings.parquet"
    table.write_parquet(out)
    write_json(
        output_dir / "provenance.json",
        {
            "created_at": now_iso(),
            "model": card["name"],
            "channel_recipe": card["channel_recipe"],
            "channels": card["channels"],
            "model_channel_order": card.get("model_channel_order"),
            "tile_size": card["tile_size"],
            "preprocess": card["preprocess"],
            "checkpoint": card.get("checkpoint"),
            "architecture": card.get("architecture"),
            "pretrained": card.get("pretrained"),
            "n_sites": len(keys),
            "images_root": str(images_root),
            "persist_codec": codec,
            "output": str(out),
            "comparison_note": card.get("notes"),
        },
    )
    return out
