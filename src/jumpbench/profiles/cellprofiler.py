"""CellProfiler profile loaders.

Assembled CPG profiles average 6–9 sites/well. That is only unfair against
embeddings if those embeddings were restricted to the JUMP-lite 4-site sample.
``jumpbench download-images --sites all`` (the default) matches the 6–9-site
support; ``--sites jump_lite`` reproduces the paper's 4-site embedding cohort.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from jumpbench.data.metadata import load_perturbations
from jumpbench.paths import resolve

JOIN = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"]


def restrict_to_jump_lite_wells(
    profiles: pl.DataFrame,
    wells: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Inner-join profiles onto the frozen JUMP-lite well set.

    This still does **not** make assembled CellProfiler features identical to
    ``cp_measure`` on the same pixels. It only aligns the well universe. Site
    support matches if embeddings were generated with ``--sites all``.
    """
    wells = wells if wells is not None else load_perturbations()
    left = profiles
    right = wells
    for col in JOIN:
        if col not in left.columns or col not in right.columns:
            raise ValueError(f"Missing join column {col}")
        left = left.with_columns(pl.col(col).cast(pl.Utf8))
        right = right.with_columns(pl.col(col).cast(pl.Utf8))
    keep_right = JOIN + [
        c for c in right.columns if c.startswith("Metadata_") and c not in left.columns
    ]
    return left.join(right.select(keep_right), on=JOIN, how="inner")


def load_paper_cellprofiler(path: str | Path, align_wells: bool = True) -> pl.DataFrame:
    df = pl.read_parquet(resolve(path))
    if align_wells:
        df = restrict_to_jump_lite_wells(df)
        df = df.with_columns(pl.lit("paper_assembled_6to9_sites").alias("Metadata_comparison_mode"))
    else:
        df = df.with_columns(pl.lit("paper_assembled_unaligned").alias("Metadata_comparison_mode"))
    df = df.with_columns(pl.lit("cellprofiler_paper").alias("Metadata_model"))
    return df


def comparison_is_fair(mode: str) -> bool:
    return mode in {"fair_same_sites", "fair_all_sites", "cp_measure_jump_lite"}
