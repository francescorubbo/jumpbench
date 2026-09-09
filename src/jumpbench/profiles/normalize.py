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

from jumpbench.config import apply_overrides, load_process_config
from jumpbench.data.metadata import load_perturbations
from jumpbench.paths import resolve

META_PREFIX = "Metadata_"
DEVICES = ("cpu", "mps", "auto")


def resolve_device(device: str | None = "cpu") -> str:
    """Map cpu|mps|auto to a concrete device. auto prefers MPS when torch has it."""
    requested = (device or "cpu").lower()
    if requested == "cpu":
        return "cpu"
    if requested not in {"mps", "auto"}:
        raise ValueError(f"Unknown device {device!r}. Known: {', '.join(DEVICES)}")
    try:
        import torch
    except ImportError:
        if requested == "mps":
            raise RuntimeError(
                "torch is required for --device mps (pip install -e '.[embed]')"
            ) from None
        return "cpu"
    available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if requested == "mps" and not available:
        raise RuntimeError("MPS is not available on this machine")
    return "mps" if available else "cpu"


def _corrcoef(X: np.ndarray, device: str = "cpu") -> np.ndarray:
    if X.shape[1] <= 1:
        return np.corrcoef(X, rowvar=False)
    if device == "cpu":
        return np.corrcoef(X, rowvar=False)
    import torch

    tensor = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
    corr = torch.corrcoef(tensor.T)
    return corr.detach().cpu().numpy().astype(np.float64, copy=False)


def _pca_fit_transform(
    X: np.ndarray,
    fit_rows: np.ndarray,
    n_components: int,
    device: str = "cpu",
) -> np.ndarray:
    n_comp = min(int(n_components), fit_rows.shape[0] - 1, fit_rows.shape[1], X.shape[0] - 1)
    n_comp = max(1, n_comp)
    if device == "cpu":
        pca = PCA(n_components=n_comp, svd_solver="full")
        pca.fit(fit_rows)
        return pca.transform(X)
    import torch

    fit = torch.as_tensor(np.ascontiguousarray(fit_rows), dtype=torch.float32, device=device)
    data = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
    mean = fit.mean(dim=0)
    centered = fit - mean
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:n_comp]
    out = (data - mean) @ components.T
    return out.detach().cpu().numpy().astype(np.float64, copy=False)


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith(META_PREFIX) and df[c].dtype.is_numeric()]


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
    """JUMP_lite CPU RobustMAD: raw MAD (scipy default scale=1), not Gaussian-scaled."""
    median = np.median(ref, axis=0)
    mad = median_abs_deviation(ref, axis=0, scale=1.0)
    mad = np.where(mad == 0, epsilon, mad)
    return (X - median) / (mad + epsilon)


def standardize(X: np.ndarray, ref: np.ndarray, epsilon: float = 1e-18) -> np.ndarray:
    mean = np.mean(ref, axis=0)
    std = np.std(ref, axis=0)
    std = np.where(std == 0, epsilon, std)
    return (X - mean) / (std + epsilon)


def _apply_by_batch(
    X: np.ndarray,
    batch_labels: np.ndarray,
    control_mask: np.ndarray,
    fit_on_controls: bool,
    transform,
    epsilon: float,
) -> np.ndarray:
    out = np.empty_like(X, dtype=np.float64)
    for batch in np.unique(batch_labels):
        mask = batch_labels == batch
        if fit_on_controls:
            ref = X[mask & control_mask]
            if len(ref) == 0:
                ref = X[mask]
        else:
            ref = X[mask]
        out[mask] = transform(X[mask], ref, epsilon)
    return out


def inverse_normal_transform(X: np.ndarray) -> np.ndarray:
    out = np.empty_like(X, dtype=np.float64)
    for i in range(X.shape[1]):
        col = X[:, i]
        finite = np.isfinite(col)
        if not finite.any():
            out[:, i] = col
            continue
        ranks = np.empty(col.size, dtype=np.float64)
        ranks[finite] = rankdata(col[finite], method="average")
        ranks[~finite] = np.nan
        n = int(finite.sum())
        transformed = np.full(col.size, np.nan, dtype=np.float64)
        transformed[finite] = norm.ppf((ranks[finite] - 0.5) / n)
        out[:, i] = transformed
    return out


def drop_outliers(
    X: np.ndarray,
    names: list[str],
    cutoff: float = 100.0,
) -> tuple[np.ndarray, list[str]]:
    """Drop features whose |z-score| exceeds cutoff (JUMP_lite drop_outliers)."""
    if X.size == 0 or cutoff is None:
        return X, names
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std = np.where(std == 0, 1e-8, std)
    z = (X - mean) / (std + 1e-8)
    max_abs = np.nanmax(np.abs(z), axis=0)
    max_abs = np.where(np.isfinite(max_abs), max_abs, np.inf)
    keep = max_abs <= cutoff
    if not keep.any():
        return X, names
    return X[:, keep], [n for n, k in zip(names, keep, strict=True) if k]


def _greedy_independent_set(adj: np.ndarray) -> list[int]:
    """Return indices of redundant nodes (not in the min-degree independent set)."""
    graph = adj.copy()
    np.fill_diagonal(graph, 0)
    remaining = set(range(graph.shape[0]))
    independent: list[int] = []
    while remaining:
        degrees = graph.sum(axis=1)
        min_degree_node = min(remaining, key=lambda node: degrees[node])
        independent.append(min_degree_node)
        neighbors = set(np.where(graph[min_degree_node] == 1)[0])
        remaining -= neighbors | {min_degree_node}
        drop = neighbors | {min_degree_node}
        for node in drop:
            graph[node, :] = 0
            graph[:, node] = 0
    redundant = set(range(adj.shape[0])) - set(independent)
    return list(redundant)


def prune_correlated(
    X: np.ndarray,
    names: list[str],
    threshold: float,
    method: str = "greedy",
    device: str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    if X.shape[1] <= 1:
        return X, names
    corr = _corrcoef(X, device=device)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0)
    if method == "independent_set":
        adj = abs_corr > threshold
        redundant = set(_greedy_independent_set(adj.astype(np.int8)))
        keep = [i not in redundant for i in range(X.shape[1])]
    else:
        keep_mask = np.ones(X.shape[1], dtype=bool)
        order = np.argsort(-np.nanvar(X, axis=0))
        for i in order:
            if not keep_mask[i]:
                continue
            keep_mask &= ~((abs_corr[i] > threshold) & (np.arange(X.shape[1]) > i))
            keep_mask[i] = True
        keep = keep_mask.tolist()
    if not any(keep):
        return X, names
    return X[:, keep], [n for n, k in zip(names, keep, strict=True) if k]


def tvn_efaar(
    X: np.ndarray,
    control_mask: np.ndarray,
    batch_labels: np.ndarray,
    epsilon: float = 0.5,
) -> np.ndarray:
    """CORAL: whiten each batch on controls, recolor to pooled-control covariance."""
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


def tvn_efaar_full(
    X: np.ndarray,
    control_mask: np.ndarray,
    batch_labels: np.ndarray,
    n_components: int = 128,
    epsilon: float = 0.5,
    dim_ratio_threshold: float = 2.5,
    device: str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    """JUMP_lite TVN-EFAAR: scale → PCA on controls → per-batch scale → CORAL."""
    X = standardize(X, X[control_mask])
    n_controls = int(control_mask.sum())
    n_comp = min(int(n_components), X.shape[1], max(1, n_controls - 1), X.shape[0] - 1)
    min_controls = n_controls
    counts = []
    for batch in np.unique(batch_labels):
        n_c = int((control_mask & (batch_labels == batch)).sum())
        if n_c >= 2:
            counts.append(n_c)
    if counts:
        min_controls = min(counts)
    if min_controls >= 2 and n_comp / min_controls > dim_ratio_threshold:
        n_comp = max(1, int(min_controls * dim_ratio_threshold))
    X = _pca_fit_transform(X, X[control_mask], n_comp, device=device)
    scaled = np.empty_like(X, dtype=np.float64)
    for batch in np.unique(batch_labels):
        mask = batch_labels == batch
        ctrl = mask & control_mask
        if int(ctrl.sum()) >= 2:
            scaled[mask] = standardize(X[mask], X[ctrl])
        else:
            scaled[mask] = X[mask]
    X = tvn_efaar(scaled, control_mask, batch_labels, epsilon)
    names = [f"PC_{i + 1:03d}" for i in range(X.shape[1])]
    return X, names


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
    join = [
        c
        for c in ("Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well")
        if c in df.columns and c in meta.columns
    ]
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
        out = out.with_columns(
            (pl.col("Metadata_control_type") == "negcon").alias("Metadata_negcon")
        )
    return out


def process_profiles(
    df: pl.DataFrame,
    preset: str = "paper_dl_default",
    process_cfg: dict[str, Any] | None = None,
    overrides: list[str] | None = None,
    preset_overrides: dict[str, Any] | None = None,
    device: str = "cpu",
) -> pl.DataFrame:
    full = process_cfg or load_process_config()
    cfg = dict(full["presets"][preset])
    if preset_overrides:
        cfg.update(preset_overrides)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    device = resolve_device(device)
    work = attach_perturbation_metadata(df)
    names = feature_columns(work)
    X = _as_numpy(work, names)
    X, names = drop_high_na(X, names, float(cfg.get("drop_na_frac", 0.3)))
    col_medians = np.nanmedian(X, axis=0)
    X = np.where(np.isfinite(X), X, col_medians)
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
    batch_col = cfg.get("batch_col")
    batches = (
        work[batch_col].cast(pl.Utf8).to_numpy()
        if batch_col and batch_col in work.columns
        else None
    )
    method = cfg.get("normalize", "robustmad")
    fit_on_controls = bool(cfg.get("fit_on_controls", True))
    epsilon = float(cfg.get("robustmad_epsilon", 1e-18))
    if method == "robustmad":
        transform = robustmad
    elif method == "standardize":
        transform = standardize
    elif method in (None, "none"):
        transform = None
    else:
        raise ValueError(method)
    if transform is not None:
        if batches is not None:
            X = _apply_by_batch(X, batches, controls, fit_on_controls, transform, epsilon)
        else:
            ref = X[controls] if fit_on_controls and controls.any() else X
            X = transform(X, ref, epsilon)
    outlier_cutoff = cfg.get("outlier_cutoff")
    if outlier_cutoff is not None:
        X, names = drop_outliers(X, names, float(outlier_cutoff))
    if cfg.get("inverse_normal_transform"):
        X = inverse_normal_transform(X)
        col_medians = np.nanmedian(X, axis=0)
        X = np.where(np.isfinite(X), X, col_medians)
    if cfg.get("prune_correlated"):
        X, names = prune_correlated(
            X,
            names,
            float(cfg.get("corr_threshold", 0.9)),
            method=str(cfg.get("corr_prune_method", "greedy")),
            device=device,
        )
    n_comp = cfg.get("pca_components")
    if n_comp:
        n_comp = min(int(n_comp), X.shape[0] - 1, X.shape[1])
        X = _pca_fit_transform(X, X, n_comp, device=device)
        names = [f"PC_{i + 1:03d}" for i in range(X.shape[1])]
    if cfg.get("tvn_efaar"):
        tvn_batch_col = cfg.get("tvn_batch_col") or batch_col
        if tvn_batch_col and tvn_batch_col in work.columns:
            tvn_batches = work[tvn_batch_col].cast(pl.Utf8).to_numpy()
            tvn_n = cfg.get("tvn_n_components")
            tvn_eps = float(cfg.get("tvn_epsilon", 0.5))
            if tvn_n and not n_comp:
                X, names = tvn_efaar_full(
                    X,
                    controls,
                    tvn_batches,
                    n_components=int(tvn_n),
                    epsilon=tvn_eps,
                    dim_ratio_threshold=float(cfg.get("tvn_dim_ratio_threshold", 2.5)),
                    device=device,
                )
            else:
                X = tvn_efaar(X, controls, tvn_batches, tvn_eps)
    meta_cols = [c for c in work.columns if c.startswith(META_PREFIX)]
    feat_df = pl.DataFrame({n: X[:, i] for i, n in enumerate(names)})
    return pl.concat([work.select(meta_cols), feat_df], how="horizontal")


def process_path(
    input_path: Path,
    output_path: Path,
    preset: str = "paper_dl_default",
    overrides: list[str] | None = None,
    preset_overrides: dict[str, Any] | None = None,
    device: str = "cpu",
) -> Path:
    df = pl.read_parquet(resolve(input_path))
    out = process_profiles(
        df,
        preset=preset,
        overrides=overrides,
        preset_overrides=preset_overrides,
        device=device,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_path)
    return output_path
