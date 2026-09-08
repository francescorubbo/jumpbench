"""Resolve original JUMP TIFF URIs from Cell Painting Gallery load_data CSVs.

Default ``site_set=all`` keeps every Orig FOV on JUMP-lite wells (typically
6–9 sites, the same support assembled CellProfiler used). ``jump_lite``
reproduces the paper's 4-site embedding cohort.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import polars as pl
from tqdm import tqdm

from jumpbench.config import load_data_config
from jumpbench.data.metadata import (
    JOIN_SITE,
    JOIN_WELL,
    attach_site_keys,
    cast_site_keys,
    filter_sites,
    filter_wells,
    load_sites,
    parse_site_key,
)
from jumpbench.data.s3util import GALLERY_BUCKET, read_s3_bytes
from jumpbench.paths import resolve

SITE_SETS = ("all", "jump_lite")

ORIG_URL_COLUMNS = {
    "URL_OrigAGP": "AGP",
    "URL_OrigDNA": "DNA",
    "URL_OrigER": "ER",
    "URL_OrigMito": "Mito",
    "URL_OrigRNA": "RNA",
}

CHANNELS = tuple(ORIG_URL_COLUMNS.values())


def load_data_key(source: str, batch: str, plate: str) -> str:
    return f"cpg0016-jump/{source}/workspace/load_data_csv/{batch}/{plate}/load_data_with_illum.csv"


def load_data_cache_path(cache_dir: Path, source: str, batch: str, plate: str) -> Path:
    return Path(cache_dir) / f"{source}__{batch}__{plate}.csv"


def tidy_load_data(df: pl.DataFrame) -> pl.DataFrame:
    """Unpivot Orig channel URL columns into (channel, uri) rows."""
    df = cast_site_keys(df)
    present = [c for c in ORIG_URL_COLUMNS if c in df.columns]
    missing = [c for c in ORIG_URL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"load_data CSV missing {missing}. Columns: {df.columns}")
    keep = [c for c in JOIN_SITE if c in df.columns] + present
    long = df.select(keep).unpivot(
        on=present,
        index=[c for c in JOIN_SITE if c in df.columns],
        variable_name="url_column",
        value_name="uri",
    )
    mapping = pl.DataFrame(
        {"url_column": list(ORIG_URL_COLUMNS), "channel": list(ORIG_URL_COLUMNS.values())}
    )
    return long.join(mapping, on="url_column", how="left").drop("url_column")


def fetch_load_data(
    source: str,
    batch: str,
    plate: str,
    cache_dir: Path | None = None,
) -> pl.DataFrame:
    cache_path = None
    if cache_dir is not None:
        cache_path = load_data_cache_path(cache_dir, source, batch, plate)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return pl.read_csv(cache_path)
    key = load_data_key(source, batch, plate)
    raw = read_s3_bytes(GALLERY_BUCKET, key)
    df = pl.read_csv(BytesIO(raw))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)
    return df


def unique_plates(frame: pl.DataFrame) -> list[dict[str, str]]:
    return (
        cast_site_keys(frame)
        .select("Metadata_Source", "Metadata_Batch", "Metadata_Plate")
        .unique(maintain_order=True)
        .to_dicts()
    )


def _s3_key(index: pl.DataFrame) -> pl.DataFrame:
    return index.with_columns(
        pl.col("uri").str.replace(r"^s3://cellpainting-gallery/", "").alias("s3_key")
    )


def site_set_summary(index: pl.DataFrame) -> dict[str, object]:
    sites = index.select(JOIN_WELL + ["Metadata_Site"]).unique()
    per_well = sites.group_by(JOIN_WELL).len().rename({"len": "n_sites"})
    counts = per_well["n_sites"]
    site_set = None
    if "Metadata_site_set" in index.columns and index.height:
        site_set = index["Metadata_site_set"][0]
    min_sites = counts.min() if per_well.height else None
    median_sites = counts.median() if per_well.height else None
    max_sites = counts.max() if per_well.height else None
    return {
        "site_set": site_set,
        "wells": per_well.height,
        "sites": sites.height,
        "files": index.height,
        "sites_per_well_min": int(min_sites) if isinstance(min_sites, (int, float)) else 0,
        "sites_per_well_median": float(median_sites)
        if isinstance(median_sites, (int, float))
        else 0.0,
        "sites_per_well_max": int(max_sites) if isinstance(max_sites, (int, float)) else 0,
    }


def restrict_to_site_set(
    catalog: pl.DataFrame,
    *,
    site_set: str,
    wells: pl.DataFrame,
    frozen_sites: pl.DataFrame | None = None,
    site_keys: list[str] | None = None,
    max_sites: int | None = None,
) -> pl.DataFrame:
    """Keep JUMP-lite wells' Orig FOVs (all) or the frozen 4-site sample."""
    if site_set not in SITE_SETS:
        raise ValueError(f"site_set must be one of {SITE_SETS}, got {site_set!r}")
    catalog = attach_site_keys(_s3_key(cast_site_keys(catalog)))
    if site_set == "jump_lite":
        if frozen_sites is None:
            raise ValueError("frozen_sites is required for site_set=jump_lite")
        frozen = attach_site_keys(frozen_sites)
        indexed = frozen.join(
            catalog.drop("Metadata_Site_Key"),
            on=JOIN_SITE,
            how="inner",
        )
    else:
        indexed = catalog.join(
            cast_site_keys(wells).select(JOIN_WELL).unique(),
            on=JOIN_WELL,
            how="inner",
        )
        if site_keys:
            indexed = indexed.filter(pl.col("Metadata_Site_Key").is_in(list(site_keys)))
    if max_sites is not None:
        keep = indexed.select("Metadata_Site_Key").unique(maintain_order=True).head(max_sites)
        indexed = indexed.join(keep, on="Metadata_Site_Key", how="inner")
    return indexed.with_columns(pl.lit(site_set).alias("Metadata_site_set")).sort(
        ["Metadata_Site_Key", "channel"]
    )


def _plates_to_fetch(
    *,
    site_set: str,
    wells: pl.DataFrame,
    frozen_sites: pl.DataFrame,
    site_keys: list[str] | None,
) -> pl.DataFrame:
    if site_keys:
        parsed = [parse_site_key(k) for k in site_keys]
        return pl.DataFrame(
            {
                "Metadata_Source": [p["source"] for p in parsed],
                "Metadata_Batch": [p["batch"] for p in parsed],
                "Metadata_Plate": [p["plate"] for p in parsed],
            }
        )
    if site_set == "jump_lite":
        return frozen_sites
    return wells


def fetch_catalog(
    plates: pl.DataFrame,
    cache_dir: Path,
    show_progress: bool = True,
) -> pl.DataFrame:
    plates_meta = unique_plates(plates)
    if not plates_meta:
        raise ValueError("No plates to fetch load_data CSVs for.")
    chunks: list[pl.DataFrame] = []
    failures: list[str] = []
    iterator: object = plates_meta
    if show_progress:
        iterator = tqdm(plates_meta, desc="load_data CSVs")
    for plate in iterator:
        source, batch, plate_id = (
            plate["Metadata_Source"],
            plate["Metadata_Batch"],
            plate["Metadata_Plate"],
        )
        try:
            raw = fetch_load_data(source, batch, plate_id, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001 — surface S3/parse failures as a batch
            failures.append(f"{source}/{batch}/{plate_id}: {exc}")
            continue
        chunks.append(tidy_load_data(raw))
    if failures and not chunks:
        raise RuntimeError("Could not fetch any JUMP load_data CSVs:\n" + "\n".join(failures))
    if failures:
        raise RuntimeError(
            f"Failed {len(failures)}/{len(plates_meta)} load_data CSVs:\n" + "\n".join(failures)
        )
    return pl.concat(chunks, how="vertical_relaxed")


def build_tiff_index(
    sites: pl.DataFrame | None = None,
    *,
    site_set: str = "all",
    max_sites: int | None = None,
    max_wells: int | None = None,
    sources: list[str] | None = None,
    site_keys: list[str] | None = None,
    plates: list[str] | None = None,
    cache_dir: Path | None = None,
    show_progress: bool = True,
) -> pl.DataFrame:
    """Resolve Orig TIFF URIs for JUMP-lite wells.

    ``site_set=all`` uses every FOV in JUMP load_data for those wells.
    ``site_set=jump_lite`` keeps the frozen v1.0 4-site sample.
    """
    if site_set not in SITE_SETS:
        raise ValueError(f"site_set must be one of {SITE_SETS}, got {site_set!r}")
    cfg = load_data_config()
    if cache_dir is None:
        cache_dir = resolve(cfg["layout"].get("manifest", "data/manifest")) / "load_data"

    wells = filter_wells(
        max_wells=max_wells,
        sources=sources,
        plates=plates,
        site_keys=site_keys if site_set == "all" else None,
    )
    frozen = pl.DataFrame()
    if site_set == "jump_lite":
        frozen = filter_sites(
            sites if sites is not None else load_sites(),
            max_sites=max_sites,
            max_wells=max_wells,
            sources=sources,
            site_keys=site_keys,
            plates=plates,
        )
        if frozen.height == 0:
            raise ValueError("No sites selected. Check --site / --source / --max-sites.")
    elif wells.height == 0:
        raise ValueError("No wells selected. Check --plate / --source / --max-wells.")

    plate_frame = _plates_to_fetch(
        site_set=site_set, wells=wells, frozen_sites=frozen, site_keys=site_keys
    )
    catalog = fetch_catalog(plate_frame, cache_dir=cache_dir, show_progress=show_progress)
    indexed = restrict_to_site_set(
        catalog,
        site_set=site_set,
        wells=wells,
        frozen_sites=frozen if site_set == "jump_lite" else None,
        site_keys=None,
        max_sites=max_sites if site_set == "all" else None,
    )
    if indexed.height == 0:
        raise RuntimeError("URI index is empty after joining load_data to the JUMP-lite cohort.")
    n_sites = indexed.select("Metadata_Site_Key").n_unique()
    if indexed.height != n_sites * len(CHANNELS):
        raise RuntimeError(
            f"URI index incomplete: {indexed.height} channel rows for {n_sites} sites "
            f"(expected {n_sites * len(CHANNELS)})."
        )
    return indexed


def write_tiff_index(index: pl.DataFrame, dest: Path | None = None) -> Path:
    cfg = load_data_config()
    dest = Path(
        dest or resolve(cfg["layout"].get("manifest", "data/manifest")) / "tiff_uris.parquet"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    index.write_parquet(dest)
    return dest
