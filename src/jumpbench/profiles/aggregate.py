from __future__ import annotations

from pathlib import Path

import polars as pl

FEATURE_PREFIXES = ("feat_",)
META_PREFIX = "Metadata_"


def _feature_cols(df: pl.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c.startswith(META_PREFIX) or c in {
            "site_key",
            "tile_y",
            "tile_x",
            "tile",
            "label",
            "branch",
            "metric",
            "value",
            "object",
            "tp",
            "filename",
            "object_label",
            "centroid_y",
            "centroid_x",
        }:
            continue
        cols.append(c)
    # long form
    if "metric" in df.columns and "value" in df.columns:
        return ["value"]
    return cols


def aggregate_sites_to_wells(
    df: pl.DataFrame,
    how: str = "median",
    tile_how: str | None = None,
) -> pl.DataFrame:
    """Median-aggregate tiles to sites (if present) then sites to wells.

    Paper: crop-level features are median-aggregated to the well, combining
    all crops of an image, then all sites of the well.
    """
    tile_how = tile_how or how
    agg = {"median": pl.median, "mean": pl.mean}[how]
    tile_agg = {"median": pl.median, "mean": pl.mean}[tile_how]

    if {"metric", "value", "object"}.issubset(df.columns):
        well_cols = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"]
        grouped = df.group_by([*well_cols, "object", "metric"]).agg(
            tile_agg("value").alias("value")
        )
        wide = grouped.pivot(on="metric", index=well_cols, values="value")
        if "Metadata_id" not in wide.columns:
            wide = wide.with_columns(
                (
                    pl.col("Metadata_Source")
                    + "__"
                    + pl.col("Metadata_Batch")
                    + "__"
                    + pl.col("Metadata_Plate")
                    + "__"
                    + pl.col("Metadata_Well")
                ).alias("Metadata_id")
            )
        return wide

    feats = [c for c in df.columns if c.startswith("feat_") or c in _feature_cols(df)]
    well_cols = [
        c
        for c in ("Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well")
        if c in df.columns
    ]
    if not well_cols:
        raise ValueError("Need Metadata_Source/Batch/Plate/Well to aggregate")

    work = df
    per_object = "Metadata_Site" in df.columns and (
        {"tile_y", "tile_x"}.issubset(df.columns) or "object_label" in df.columns
    )
    if per_object:
        site_cols = [*well_cols, "Metadata_Site"]
        work = work.group_by(site_cols).agg([tile_agg(c).alias(c) for c in feats])
    out = work.group_by(well_cols).agg([agg(c).alias(c) for c in feats])
    out = out.with_columns(
        (
            pl.col("Metadata_Source")
            + "__"
            + pl.col("Metadata_Batch")
            + "__"
            + pl.col("Metadata_Plate")
            + "__"
            + pl.col("Metadata_Well")
        ).alias("Metadata_id")
    )
    return out


def aggregate_path(input_path: Path, output_path: Path, how: str = "median") -> Path:
    df = pl.read_parquet(input_path)
    wells = aggregate_sites_to_wells(df, how=how)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wells.write_parquet(output_path)
    return output_path
