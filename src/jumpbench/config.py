from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from jumpbench.paths import repo_root, resolve

ZARR_CHANNELS = ("AGP", "DNA", "ER", "Mito", "RNA")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with resolve(path).open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_data_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or repo_root() / "configs" / "data.yaml")


def load_process_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or repo_root() / "configs" / "process.yaml")


def load_models_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or repo_root() / "configs" / "models.yaml")


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply dotted KEY=VALUE overrides. VALUE is YAML-parsed."""
    out = deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        cursor = out
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return out


def resolve_model(name: str, models_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a flat, runnable model card with channels already selected."""
    models_cfg = models_cfg or load_models_config()
    if name not in models_cfg["models"]:
        known = ", ".join(sorted(models_cfg["models"]))
        raise KeyError(f"Unknown model {name!r}. Known: {known}")
    card = deepcopy(models_cfg["models"][name])
    recipe = models_cfg.get("channel_recipe", "jump_lite_as_run")
    recipe_block = card.pop(recipe, None) or card.pop("jump_lite_as_run", {}) or {}
    # Drop the unused recipe so callers don't mix them.
    card.pop("paper_table_s3", None)
    card.pop("jump_lite_as_run", None)
    card["name"] = name
    card["channel_recipe"] = recipe
    card["channels"] = list(recipe_block.get("channels", card.get("channels", [])))
    if "model_channel_order" in recipe_block:
        card["model_channel_order"] = list(recipe_block["model_channel_order"])
    card["preprocess"] = list(card.get("preprocess") or [])
    card["tile_size"] = int(
        card.get("tile_size") or models_cfg.get("runtime", {}).get("tile_size") or 224
    )
    card["aggregation"] = deepcopy(models_cfg.get("aggregation", {}))
    card["runtime"] = deepcopy(models_cfg.get("runtime", {}))
    card["zarr_channels"] = list(models_cfg.get("zarr_channels", ZARR_CHANNELS))
    return card


def channel_indices(channel_names: list[str], zarr_channels: list[str] | None = None) -> list[int]:
    order = list(zarr_channels or ZARR_CHANNELS)
    missing = [c for c in channel_names if c not in order]
    if missing:
        raise ValueError(f"Unknown channels {missing}. Zarr order is {order}")
    return [order.index(c) for c in channel_names]
