"""CPU profile processing matching the JUMP_lite variance-first recipe.

This is a readable reimplementation of src/norm_3 CPU RobustMAD + sklearn PCA
+ CORAL TVN-EFAAR. It does not require RAPIDS. Numeric results will be close
but not bit-identical to the paper's GPU sweep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.linalg import fractional_matrix_power, pinvh
from scipy.stats import median_abs_deviation, norm, rankdata
from sklearn.decomposition import PCA

from jumpbench.config import load_process_config
from jumpbench.data.metadata import load_perturbations
from jumpbench.paths import resolve

META_PREFIX = "Metadata_"


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if not c.startswith(META_PREFIX)
        and df[c].dtype.is_numeric()
    ]


def _as_numpy(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select(cols).to_numpy().astype(np.float64, copy=False)


def drop_high_na(X: np.ndarray, names: list[str], cutoff: float) -> tuple[np.ndarray, list[str]]:
    frac = np.isnan(X).mean(axis=0)
    keep = frac <= cutoff
    return X[:, keep], [n for n, k in zip(names, keep, strict=True) if k]


def filter_low_variance(
    X: np.ndarray,
    names: list[str],
    freq_cut: float,
    unique_cut: float,
) -> tuple[np.ndarray, list[str]]:
    keep = []
    for i in range(X.shape[1]):
        col = X[:, i]
        col = col[np.isfinite(col)]
        if col.size == 0:
            keep.append(False)
            continue
        unique = np.unique(col)
        unique_frac = unique.size / col.size
        # frequency of the mode vs the rest, matching cytominer-style freqCut
        counts = np.unique(col, return_counts=True)[1]
        counts.sort()
        freq_ratio = counts[-1] / max(counts[-2], 1) if counts.size > 1 else np.inf
        keep.append(unique_frac >= unique_cut and freq_ratio <= (1.0 / max(freq_cut, 1e-12)))
    mask = np.asarray(keep)
    if not mask.any():
        return X, names
    return X[:, mask], [n for n, k in zip(names, mask, strict=True) if k]


def robustmad(X: np.ndarray, ref: np.ndarray, epsilon: float = 1e-18) -> np.ndarray:
    median = np.median(ref, axis=0)
    mad = median_abs_deviation(ref, axis=0, scale="normal")
    mad = np.where(mad == 0, epsilon, mad)
    return (X - median) / (mad + epsilon)


def standardize(X: np.ndarray, ref: np.ndarray, epsilon: float = 1e-18) -> np.ndarray:
    mean = np.mean(ref, axis=0)
    std = np.std(ref, axis=0)
    std = np.where(std == 0, epsilon, std)
    return (X - mean) / (std + epsilon)


def inverse_normal_transform(X: np.ndarray) -> np.ndarray:
    out = np.empty_like(X, dtype=np.float64)
    for i in range(X.shape[1]):
        col = X[:, i]
        ranks = rankdata(col, method="average")
        out[:, i] = norm.ppf((ranks - 0.5) / col.size)
    return out


def prune_correlated(X: np.ndarray, names: list[str], threshold: float) -> tuple[np.ndarray, list[str]]:
    if X.shape[1] <= 1:
        return X, names
    corr = np.corrcoef(X, rowvar=False)
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0)
    keep = np.ones(X.shape[1], dtype=bool)
    order = np.argsort(-np.nanvar(X, axis=0))
    for i in order:
        if not keep[i]:
            continue
        keep &= ~((abs_corr[i] > threshold) & (np.arange(X.shape[1]) > i))
        keep[i] = True
    return X[:, keep], [n for n, k in zip(names, keep, strict=True) if k]


def tvn_efaar(
    X: np.ndarray,
    control_mask: np.ndarray,
    batch_labels: np.ndarray,
    epsilon: float = 0.5,
) -> np.ndarray:
    """CORAL: whiten each plate on controls, recolor to pooled-control covariance."""
    controls = X[control_mask]
    n_features = X.shape[1]
    target = np.cov(controls, rowvar=False) + epsilon * np.eye(n_features)
    target_sqrt = fractional_matrix_power(target, 0.5).real
    out = X.copy()
    for batch in np.unique(batch_labels):
        batch_mask = batch_labels == batch
        batch_controls = out[batch_mask & control_mask]
        if len(batch_controls) < 2:
            continue
        source = np.cov(batch_controls, rowvar=False) + epsilon * np.eye(n_features)
        source_inv_sqrt = fractional_matrix_power(source, -0.5).real
        if np.any(~np.isfinite(source_inv_sqrt)):
            source_inv_sqrt = pinvh(fractional_matrix_power(source, 0.5).real)
        out[batch_mask] = out[batch_mask] @ source_inv_sqrt @ target_sqrt
    return out.real


def _control_mask(df: pl.DataFrame, col: str, key: str) -> np.ndarray:
    if col not in df.columns:
        if "Metadata_negcon" in df.columns:
            return df["Metadata_negcon"].to_numpy().astype(bool)
        raise KeyError(f"No control column {col}")
    series = df[col]
    if series.dtype == pl.Boolean:
        return series.to_numpy()
    return (series.cast(pl.Utf8) == str(key)).to_numpy()


def attach_perturbation_metadata(df: pl.DataFrame) -> pl.DataFrame:
    meta = load_perturbations()
    join = [c for c in ("Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well") if c in df.columns and c in meta.columns]
    if len(join) < 4:
        return df
    extra = [c for c in meta.columns if c.startswith("Metadata_") and c not in df.columns]
    left = df.with_columns([pl.col(c).cast(pl.Utf8) for c in join])
    right = meta.select(join + extra).with_columns([pl.col(c).cast(pl.Utf8) for c in join])
    out = left.join(right, on=join, how="left")
    if "Metadata_control_type" not in out.columns and "Metadata_pert_type" in out.columns:
        out = out.with_columns(pl.col("Metadata_pert_type").alias("Metadata_control_type"))
    if "Metadata_negcon" not in out.columns and "Metadata_pert_type" in out.columns:
        out = out.with_columns((pl.col("Metadata_pert_type") == "negcon").alias("Metadata_negcon"))
    elif "Metadata_negcon" not in out.columns and "Metadata_control_type" in out.columns:
        out = out.with_columns((pl.col("Metadata_control_type") == "negcon").alias("Metadata_negcon"))
    return out


def process_profiles(
    df: pl.DataFrame,
    preset: str = "paper_dl_default",
    process_cfg: dict[str, Any] | None = None,
) -> pl.DataFrame:
    cfg = (process_cfg or load_process_config())["presets"][preset]
    work = attach_perturbation_metadata(df)
    names = feature_columns(work)
    X = _as_numpy(work, names)
    X, names = drop_high_na(X, names, float(cfg.get("drop_na_frac", 0.3)))
    X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
    X, names = filter_low_variance(
        X,
        names,
        float(cfg.get("variance_freq_cut", 0.05)),
        float(cfg.get("variance_unique_cut", 0.01)),
    )
    controls = _control_mask(
        work,
        cfg.get("control_col", "Metadata_control_type"),
        cfg.get("control_key", "negcon"),
    )
    ref = X[controls] if cfg.get("fit_on_controls", True) and controls.any() else X
    method = cfg.get("normalize", "robustmad")
    if method == "robustmad":
        X = robustmad(X, ref, float(cfg.get("robustmad_epsilon", 1e-18)))
    elif method == "standardize":
        X = standardize(X, ref)
    elif method in (None, "none"):
        pass
    else:
        raise ValueError(method)
    if cfg.get("inverse_normal_transform"):
        X = inverse_normal_transform(X)
    if cfg.get("prune_correlated"):
        X, names = prune_correlated(X, names, float(cfg.get("corr_threshold", 0.9)))
    n_comp = cfg.get("pca_components")
    if n_comp:
        n_comp = min(int(n_comp), X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_comp, svd_solver="full")
        X = pca.fit_transform(X)
        names = [f"PC_{i + 1:03d}" for i in range(X.shape[1])]
    if cfg.get("tvn_efaar") and cfg.get("batch_col") in work.columns:
        batches = work[cfg["batch_col"]].cast(pl.Utf8).to_numpy()
        X = tvn_efaar(X, controls, batches, float(cfg.get("tvn_epsilon", 0.5)))
    meta_cols = [c for c in work.columns if c.startswith(META_PREFIX)]
    feat_df = pl.DataFrame({n: X[:, i] for i, n in enumerate(names)})
    return pl.concat([work.select(meta_cols), feat_df], how="horizontal")


def process_path(input_path: Path, output_path: Path, preset: str = "paper_dl_default") -> Path:
    df = pl.read_parquet(resolve(input_path))
    out = process_profiles(df, preset=preset)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_path)
    return output_path
