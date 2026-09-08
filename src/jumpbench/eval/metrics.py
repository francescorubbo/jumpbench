from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from jumpbench.profiles.normalize import feature_columns

PAPER_PA_CRISPR = 0.815


def _to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas()


def phenotypic_activity(
    df: pl.DataFrame,
    compound_col: str = "Metadata_JCP2022",
    negcon_col: str = "Metadata_negcon",
    batch_col: str = "Metadata_Plate",
    group_col: str | None = "Metadata_Group",
    distance: str = "cosine",
    null_size: int = 10_000,
    p_threshold: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Phenotypic activity (replicate vs plate-matched negative controls).

    Matches JUMP_lite/src/norm_3/metrics.py: copairs average precision, then
    Normalized Average Precision so random retrieval is 0.
    """
    from copairs import map as copairs_map
    from copairs.matching import assign_reference_index

    features = feature_columns(df)
    pdf = _to_pandas(df)
    has_groups = group_col is not None and group_col in pdf.columns
    if has_groups:
        pos_sameby = [compound_col, group_col, "Metadata_reference_index"]
        neg_sameby = [batch_col, group_col]
    else:
        pos_sameby = [compound_col, "Metadata_reference_index"]
        neg_sameby = [batch_col]
    pos_diffby: list[str] = []
    neg_diffby = [compound_col, negcon_col, "Metadata_reference_index"]

    parts = []
    groups = sorted(pdf[group_col].unique()) if has_groups else [None]
    for grp in groups:
        grp_df = pdf[pdf[group_col] == grp].copy() if has_groups else pdf.copy()
        grp_df = grp_df.reset_index(drop=True)
        if negcon_col in grp_df.columns:
            grp_df = assign_reference_index(
                grp_df,
                f"{negcon_col} == True",
                reference_col="Metadata_reference_index",
                default_value=-1,
            )
        else:
            raise KeyError(f"Need {negcon_col} boolean column for PA")
        meta = grp_df.filter(regex="^Metadata")
        profiles = grp_df[features].to_numpy()
        parts.append(
            copairs_map.average_precision(
                meta, profiles, pos_sameby, pos_diffby, neg_sameby, neg_diffby, distance=distance
            )
        )
    activity_ap = pd.concat(parts, ignore_index=True)
    if negcon_col in activity_ap.columns:
        activity_ap = activity_ap.query(f"{negcon_col} == False").copy()
    activity_map = copairs_map.mean_average_precision(
        activity_ap, pos_sameby, null_size=null_size, threshold=p_threshold, seed=seed
    ).copy()
    mean_nap = float(activity_map["mean_normalized_average_precision"].mean())
    return {
        "mean_nap": mean_nap,
        "median_nap": float(activity_map["mean_normalized_average_precision"].median()),
        "n_perturbations": int(len(activity_map)),
        "activity_map": activity_map,
    }


def phenotypic_consistency(
    df: pl.DataFrame,
    compound_col: str = "Metadata_JCP2022",
    target_col: str = "Metadata_target_list",
    negcon_col: str = "Metadata_negcon",
    min_compounds_per_target: int = 3,
    distance: str = "cosine",
    null_size: int = 10_000,
    p_threshold: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Same-target vs different-target retrieval (RefChem PC)."""
    from copairs import map as copairs_map

    features = feature_columns(df)
    pdf = _to_pandas(df)
    if negcon_col in pdf.columns:
        pdf = pdf.loc[~pdf[negcon_col].astype(bool)].copy()
    if target_col not in pdf.columns:
        raise KeyError(f"Need {target_col} for phenotypic consistency")

    exploded = pdf.copy()
    exploded[target_col] = exploded[target_col].apply(
        lambda x: x if isinstance(x, list) else str(x).split("|") if pd.notna(x) else []
    )
    exploded = exploded.explode(target_col)
    exploded = exploded[exploded[target_col].notna() & (exploded[target_col] != "")]
    counts = exploded.groupby(target_col)[compound_col].nunique()
    keep = counts[counts >= min_compounds_per_target].index
    exploded = exploded[exploded[target_col].isin(keep)]
    if exploded.empty:
        return {"mean_nap": float("nan"), "n_targets": 0, "activity_map": pd.DataFrame()}

    # Consensus per compound then score target retrieval.
    feat_means = exploded.groupby(compound_col)[features].mean()
    target_map = exploded.groupby(compound_col)[target_col].first()
    consensus = feat_means.join(target_map).reset_index()
    meta = consensus[[compound_col, target_col]].copy()
    meta.columns = [compound_col, "Metadata_target"]
    profiles = consensus[features].to_numpy()
    pos_sameby = ["Metadata_target"]
    pos_diffby = [compound_col]
    neg_sameby: list[str] = []
    neg_diffby = ["Metadata_target"]
    ap = copairs_map.average_precision(
        meta, profiles, pos_sameby, pos_diffby, neg_sameby, neg_diffby, distance=distance
    )
    mmap = copairs_map.mean_average_precision(
        ap, pos_sameby, null_size=null_size, threshold=p_threshold, seed=seed
    )
    return {
        "mean_nap": float(mmap["mean_normalized_average_precision"].mean()),
        "n_targets": int(len(mmap)),
        "activity_map": mmap,
    }


def balanced_pa_pc(pa: float, pc: float) -> float:
    """Paper's ranking scalar (unscaled product). Config selection uses min-max rescaling."""
    if not np.isfinite(pa) or not np.isfinite(pc):
        return float("nan")
    return float(pa * pc)


def evaluate_profiles(
    df: pl.DataFrame,
    tasks: tuple[str, ...] = ("pa", "pc"),
    **kwargs: Any,
) -> dict[str, Any]:
    paper_ref = kwargs.pop("paper_ref", None)
    out: dict[str, Any] = {}
    if "pa" in tasks:
        pa_kwargs = {
            k: v for k, v in kwargs.items() if k in phenotypic_activity.__code__.co_varnames
        }
        pa = phenotypic_activity(df, **pa_kwargs)
        out["pa"] = {k: v for k, v in pa.items() if k != "activity_map"}
        out["_pa_map"] = pa.get("activity_map")
        if paper_ref in {"crispr", "crispr_pa"}:
            mean_nap = float(out["pa"]["mean_nap"])
            out["pa"]["subset"] = "crispr"
            out["pa"]["paper_nap"] = PAPER_PA_CRISPR
            out["pa"]["delta_vs_paper"] = mean_nap - PAPER_PA_CRISPR
    if "pc" in tasks:
        try:
            pc = phenotypic_consistency(df)
            out["pc"] = {k: v for k, v in pc.items() if k != "activity_map"}
        except KeyError as exc:
            out["pc"] = {"error": str(exc)}
    if "pa" in out and "pc" in out and "mean_nap" in out.get("pc", {}):
        out["balanced_pa_pc"] = balanced_pa_pc(
            float(out["pa"]["mean_nap"]),
            float(out["pc"].get("mean_nap", float("nan"))),
        )
    return out


def evaluate_path(input_path: Path, **kwargs: Any) -> dict[str, Any]:
    return evaluate_profiles(pl.read_parquet(input_path), **kwargs)
