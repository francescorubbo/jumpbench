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


JOIN_WELL = [
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_Well",
]

JOIN_SITE = JOIN_WELL + ["Metadata_Site"]


def cast_site_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize join columns so frozen manifests match JUMP load_data CSVs."""
    casts = []
    for col in JOIN_WELL:
        if col in df.columns:
            casts.append(pl.col(col).cast(pl.Utf8))
    if "Metadata_Site" in df.columns:
        casts.append(pl.col("Metadata_Site").cast(pl.Int64))
    return df.with_columns(casts) if casts else df


def attach_site_keys(df: pl.DataFrame) -> pl.DataFrame:
    df = cast_site_keys(df)
    if "Metadata_Site_Key" in df.columns or "Metadata_Site" not in df.columns:
        return df
    return df.with_columns(
        (
            pl.col("Metadata_Source")
            + "__"
            + pl.col("Metadata_Batch")
            + "__"
            + pl.col("Metadata_Plate")
            + "__"
            + pl.col("Metadata_Well")
            + "__"
            + pl.col("Metadata_Site").cast(pl.Utf8)
        ).alias("Metadata_Site_Key")
    )


def wells_from_site_keys(site_keys: list[str]) -> pl.DataFrame:
    parsed = [parse_site_key(k) for k in site_keys]
    return cast_site_keys(
        pl.DataFrame(
            {
                "Metadata_Source": [p["source"] for p in parsed],
                "Metadata_Batch": [p["batch"] for p in parsed],
                "Metadata_Plate": [p["plate"] for p in parsed],
                "Metadata_Well": [p["well"] for p in parsed],
            }
        )
    ).unique()


def filter_wells(
    wells: pl.DataFrame | None = None,
    *,
    max_wells: int | None = None,
    sources: list[str] | None = None,
    plates: list[str] | None = None,
    batches: list[str] | None = None,
    site_keys: list[str] | None = None,
) -> pl.DataFrame:
    wells = cast_site_keys(wells if wells is not None else load_wells())
    if site_keys:
        wells = wells.join(wells_from_site_keys(site_keys), on=JOIN_WELL, how="inner")
    if sources:
        wells = wells.filter(pl.col("Metadata_Source").is_in(list(sources)))
    if batches:
        wells = wells.filter(pl.col("Metadata_Batch").is_in(list(batches)))
    if plates:
        wells = wells.filter(pl.col("Metadata_Plate").is_in(list(plates)))
    if max_wells is not None:
        wells = wells.head(max_wells)
    return wells


def filter_sites(
    sites: pl.DataFrame | None = None,
    *,
    max_sites: int | None = None,
    max_wells: int | None = None,
    sources: list[str] | None = None,
    site_keys: list[str] | None = None,
    plates: list[str] | None = None,
    batches: list[str] | None = None,
) -> pl.DataFrame:
    sites = attach_site_keys(sites if sites is not None else load_sites())
    well_filters = sources or plates or batches
    if max_wells is not None or (well_filters and "Metadata_Well" in sites.columns):
        wells = filter_wells(
            sites.select(JOIN_WELL).unique(maintain_order=True),
            max_wells=max_wells,
            sources=sources,
            plates=plates,
            batches=batches,
            site_keys=None,
        )
        sites = sites.join(wells.select(JOIN_WELL), on=JOIN_WELL, how="inner")
        sources = None
        plates = None
        batches = None
    key_col = "Metadata_Site_Key" if "Metadata_Site_Key" in sites.columns else None
    if site_keys:
        if key_col is None:
            raise ValueError("Site table has no Metadata_Site_Key")
        sites = sites.filter(pl.col(key_col).is_in(list(site_keys)))
    if sources:
        sites = sites.filter(pl.col("Metadata_Source").is_in(list(sources)))
    if plates:
        sites = sites.filter(pl.col("Metadata_Plate").is_in(list(plates)))
    if batches:
        sites = sites.filter(pl.col("Metadata_Batch").is_in(list(batches)))
    if max_sites is not None:
        if key_col:
            keep = sites.select(key_col).unique(maintain_order=True).head(max_sites)
            sites = sites.join(keep, on=key_col, how="inner")
        else:
            sites = sites.head(max_sites)
    return sites
