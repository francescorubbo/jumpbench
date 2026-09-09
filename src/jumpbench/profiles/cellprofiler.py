"""CellProfiler profile loaders.

Assembled CPG profiles average 6–9 sites/well. That is only unfair against
embeddings if those embeddings were restricted to the JUMP-lite 4-site sample.
``jumpbench download-images --sites all`` (the default) matches the 6–9-site
support; ``--sites jump_lite`` reproduces the paper's 4-site embedding cohort.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from jumpbench.data.metadata import JOIN_WELL, load_perturbations
from jumpbench.paths import resolve

JOIN = list(JOIN_WELL)
JOIN_PLATE = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate"]
# Assembled CPG v1.0c profiles omit Metadata_Batch; Source+Plate+Well is unique.
JOIN_NO_BATCH = ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]
# Known CLI --subset values for assembled CellProfiler alignment.
SUBSETS = ("all", "crispr")

_CRISPR_COLUMNS = (
    "Metadata_Group",
    "Metadata_PlateType",
    "Metadata_plate_type",
    "Metadata_perturbation_type",
    "Metadata_Perturbation_Type",
    "Metadata_perturbation_modality",
    "Metadata_modality",
)


def _names(df: pl.DataFrame | pl.LazyFrame) -> list[str]:
    if isinstance(df, pl.LazyFrame):
        return list(df.collect_schema().names())
    return list(df.columns)


def _cast_join(df: pl.DataFrame | pl.LazyFrame, cols: list[str] | None = None):
    cols = list(JOIN if cols is None else cols)
    present = set(_names(df))
    return df.with_columns([pl.col(c).cast(pl.Utf8) for c in cols if c in present])


def _join_on(left_names: list[str] | set[str], right_names: list[str] | set[str]) -> list[str]:
    """Join keys present on both sides. Batch is optional (CPG assembled omits it)."""
    left, right = set(left_names), set(right_names)
    keys = [c for c in JOIN if c in left and c in right]
    missing = [c for c in JOIN_NO_BATCH if c not in keys]
    if missing:
        raise ValueError(
            "Need Metadata_Source, Metadata_Plate, and Metadata_Well to align "
            f"profiles (Metadata_Batch is filled from JUMP-lite wells). Missing: {missing}"
        )
    return keys


def _maybe_parse_site(df: pl.DataFrame | pl.LazyFrame):
    """Fill JOIN keys from a ``site`` / ``Metadata_id`` well key when needed."""
    names = _names(df)
    if all(c in names for c in JOIN_NO_BATCH):
        return df
    key_col = next((c for c in ("site", "Metadata_id", "Metadata_Site_Key") if c in names), None)
    if key_col is None:
        raise ValueError(
            "Profiles have neither Metadata_Source/Plate/Well nor a 'site' column "
            f"to parse them from. Columns: {names[:30]}"
        )
    parts = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"]
    return df.with_columns(
        [pl.col(key_col).str.split("__").list.get(i).alias(p) for i, p in enumerate(parts)]
    )


def _negcon_mask(df: pl.DataFrame) -> pl.Series:
    if "Metadata_negcon" in df.columns:
        col = df["Metadata_negcon"]
        if col.dtype == pl.Boolean:
            return col.fill_null(False)
        return col.cast(pl.Utf8).str.to_lowercase().is_in(["true", "1", "yes"])
    for col_name, key in (
        ("Metadata_control_type", "negcon"),
        ("Metadata_pert_type", "negcon"),
    ):
        if col_name in df.columns:
            return df[col_name].cast(pl.Utf8).str.to_lowercase() == key
    return pl.Series("negcon", [False] * df.height)


def crispr_mask(df: pl.DataFrame) -> pl.Series:
    """Boolean mask of CRISPR-labeled wells. Raises if no CRISPR column exists."""
    preferred = [c for c in _CRISPR_COLUMNS if c in df.columns]
    others = [c for c in df.columns if c.startswith("Metadata_") and c not in preferred]
    for col in [*preferred, *others]:
        try:
            labels = df[col].cast(pl.Utf8).fill_null("").str.to_lowercase()
        except Exception:
            continue
        mask = labels.str.contains("crispr")
        if int(mask.sum()) > 0:
            return mask.alias(col)
    if "Metadata_JCP2022" in df.columns:
        jcp = df["Metadata_JCP2022"].cast(pl.Utf8).fill_null("")
        mask = jcp.str.starts_with("JCP2022_8")
        if int(mask.sum()) > 0:
            return mask.alias("Metadata_JCP2022")
    present = [c for c in df.columns if c.startswith("Metadata_")]
    raise ValueError("No CRISPR labels found in metadata columns: " + ", ".join(present))


def filter_crispr_wells(wells: pl.DataFrame) -> pl.DataFrame:
    """CRISPR treatments plus plate-matched negative controls.

    JUMP_lite PA compares CRISPR replicates to DMSO/negcon on the same plate.
    Negcons are kept even when they are not themselves labeled CRISPR.
    """
    wells = _cast_join(wells)
    mask = crispr_mask(wells)
    plates = wells.filter(mask).select(JOIN_PLATE).unique()
    on_plates = wells.join(plates, on=JOIN_PLATE, how="inner")
    keep = crispr_mask(on_plates) | _negcon_mask(on_plates)
    out = on_plates.filter(keep)
    if out.height == 0:
        raise ValueError("CRISPR filter matched no wells")
    return out


def restrict_to_jump_lite_wells(
    profiles: pl.DataFrame,
    wells: pl.DataFrame | None = None,
    subset: str | None = None,
) -> pl.DataFrame:
    """Inner-join profiles onto the frozen JUMP-lite well set.

    This still does **not** make assembled CellProfiler features identical to
    ``cp_measure`` on the same pixels. It only aligns the well universe. Site
    support matches if embeddings were generated with ``--sites all``.
    """
    wells = wells if wells is not None else load_perturbations()
    subset = subset or "all"
    if subset == "crispr":
        wells = filter_crispr_wells(wells)
    elif subset not in {None, "all"}:
        raise ValueError(f"Unknown subset {subset!r}. Known: {', '.join(SUBSETS)}")
    left = _cast_join(_maybe_parse_site(profiles))
    right = _cast_join(wells)
    on = _join_on(left.columns, right.columns)
    keep_right = list(
        dict.fromkeys(
            [
                *on,
                *[c for c in right.columns if c.startswith("Metadata_") and c not in left.columns],
            ]
        )
    )
    return left.join(right.select(keep_right), on=on, how="inner")


def _prepare_source_lf(path: Path) -> pl.LazyFrame:
    lf = _maybe_parse_site(pl.scan_parquet(path))
    names = list(lf.collect_schema().names())
    join_cols = [c for c in JOIN if c in names]
    missing = [c for c in JOIN_NO_BATCH if c not in join_cols]
    if missing:
        raise ValueError(
            f"{path} has neither Metadata_Source/Plate/Well nor a 'site' column "
            f"to parse them from. Missing: {missing}"
        )
    lf = lf.with_columns([pl.col(c).cast(pl.Utf8) for c in join_cols])
    feature_cols = [
        c
        for c in names
        if not str(c).startswith("Metadata_") and c not in {*JOIN, "site", "Metadata_id"}
    ]
    return lf.select([*join_cols, *feature_cols])


def load_paper_cellprofiler(
    path: str | Path,
    align_wells: bool = True,
    subset: str | None = None,
) -> pl.DataFrame:
    df = pl.read_parquet(resolve(path))
    if align_wells:
        df = restrict_to_jump_lite_wells(df, subset=subset)
        df = df.with_columns(pl.lit("paper_assembled_6to9_sites").alias("Metadata_comparison_mode"))
    else:
        df = df.with_columns(pl.lit("paper_assembled_unaligned").alias("Metadata_comparison_mode"))
    df = df.with_columns(pl.lit("cellprofiler_paper").alias("Metadata_model"))
    if "Metadata_id" not in df.columns and all(c in df.columns for c in JOIN):
        df = df.with_columns(
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
    return df


def align_paper_cellprofiler(
    source: str | Path,
    output: str | Path,
    subset: str | None = None,
) -> Path:
    """Stream-join assembled CPG profiles onto JUMP-lite wells and write parquet."""
    source = resolve(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    wells = load_perturbations()
    subset = subset or "all"
    if subset == "crispr":
        wells = filter_crispr_wells(wells)
    elif subset not in {None, "all"}:
        raise ValueError(f"Unknown subset {subset!r}. Known: {', '.join(SUBSETS)}")
    extra = [c for c in wells.columns if c.startswith("Metadata_") and c not in JOIN]
    print(f"Aligning {wells.height} JUMP-lite wells (subset={subset}) from {source}")
    right = _cast_join(wells).select(JOIN + extra).lazy()
    left = _prepare_source_lf(source)
    on = _join_on(left.collect_schema().names(), right.collect_schema().names())
    joined = left.join(right, on=on, how="inner").with_columns(
        [
            (
                pl.col("Metadata_Source")
                + "__"
                + pl.col("Metadata_Batch")
                + "__"
                + pl.col("Metadata_Plate")
                + "__"
                + pl.col("Metadata_Well")
            ).alias("Metadata_id"),
            pl.lit("paper_assembled_6to9_sites").alias("Metadata_comparison_mode"),
            pl.lit("cellprofiler_paper").alias("Metadata_model"),
        ]
    )
    joined.sink_parquet(output)
    n = pl.scan_parquet(output).select(pl.len()).collect().item()
    if n == 0:
        output.unlink(missing_ok=True)
        raise ValueError(
            f"No matching wells after joining {source} onto JUMP-lite"
            + (" CRISPR" if subset == "crispr" else "")
            + " wells"
        )
    return output


def comparison_is_fair(mode: str) -> bool:
    return mode in {"fair_same_sites", "fair_all_sites", "cp_measure_jump_lite"}
