from __future__ import annotations

from pathlib import Path

import polars as pl

from jumpbench.config import load_data_config
from jumpbench.paths import resolve

SITE_KEY_PARTS = ("source", "batch", "plate", "well", "site")


def _read(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(resolve(path))


def load_sites(path: str | Path | None = None) -> pl.DataFrame:
    cfg = load_data_config()
    return _read(path or cfg["local_metadata"]["sites"])


def load_wells(path: str | Path | None = None) -> pl.DataFrame:
    cfg = load_data_config()
    return _read(path or cfg["local_metadata"]["wells"])


def load_perturbations(path: str | Path | None = None) -> pl.DataFrame:
    cfg = load_data_config()
    return _read(path or cfg["local_metadata"]["perturbation"])


def load_refchem(path: str | Path | None = None) -> pl.DataFrame:
    cfg = load_data_config()
    return _read(path or cfg["local_metadata"]["refchem"])


def load_motive_allowlist(path: str | Path | None = None) -> pl.DataFrame:
    cfg = load_data_config()
    return _read(path or cfg["local_metadata"]["motive_allowlist"])


def site_key(source: str, batch: str, plate: str, well: str, site: str | int) -> str:
    return f"{source}__{batch}__{plate}__{well}__{site}"


def well_id(source: str, batch: str, plate: str, well: str) -> str:
    return f"{source}__{batch}__{plate}__{well}"


def parse_site_key(key: str) -> dict[str, str]:
    parts = key.split("__")
    if len(parts) < 5:
        raise ValueError(f"Expected source__batch__plate__well__site, got {key!r}")
    return dict(zip(SITE_KEY_PARTS, parts[:5], strict=True))


def well_id_from_site_key(key: str) -> str:
    parsed = parse_site_key(key)
    return well_id(parsed["source"], parsed["batch"], parsed["plate"], parsed["well"])
