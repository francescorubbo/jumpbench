"""JUMP_lite-style profile processing helpers."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from jumpbench.profiles.normalize import (
    _apply_by_batch,
    _corrcoef,
    _pca_fit_transform,
    drop_outliers,
    process_profiles,
    prune_correlated,
    resolve_device,
    robustmad,
    tvn_efaar_full,
)


def test_robustmad_is_per_plate_when_batched():
    # Plate A centered at 0, plate B at 100. Per-plate MAD should zero each plate's controls.
    rng = np.random.default_rng(0)
    a_ctrl = rng.normal(0, 1, size=(20, 2))
    a_trt = rng.normal(3, 1, size=(10, 2))
    b_ctrl = rng.normal(100, 1, size=(20, 2))
    b_trt = rng.normal(103, 1, size=(10, 2))
    X = np.vstack([a_ctrl, a_trt, b_ctrl, b_trt])
    batches = np.array(["A"] * 30 + ["B"] * 30)
    controls = np.array([True] * 20 + [False] * 10 + [True] * 20 + [False] * 10)
    out = _apply_by_batch(X, batches, controls, True, robustmad, 1e-18)
    assert abs(np.median(out[controls & (batches == "A"), 0])) < 1e-9
    assert abs(np.median(out[controls & (batches == "B"), 0])) < 1e-9
    # Global MAD of equal-sized 0 vs 100 control clusters leaves plate B near 1, not 0.
    global_out = robustmad(X, X[controls], 1e-18)
    assert abs(np.median(global_out[controls & (batches == "B"), 0])) > 0.5


def test_drop_outliers_removes_extreme_feature():
    # A single outlier's |z| is at most sqrt(n-1), so n must exceed ~10k for cutoff=100.
    n = 20_000
    X = np.zeros((n, 3), dtype=np.float64)
    X[:, 0] = 1.0
    X[0, 1] = 10_000.0
    X[:, 2] = np.linspace(0, 1, n)
    kept_x, names = drop_outliers(X, ["ok", "spike", "ramp"], cutoff=100)
    assert "spike" not in names
    assert names == ["ok", "ramp"]
    assert kept_x.shape[1] == 2


def test_independent_set_prune_drops_duplicate():
    rng = np.random.default_rng(1)
    a = rng.normal(size=80)
    X = np.column_stack([a, a, rng.normal(size=80)])
    kept, names = prune_correlated(X, ["a", "a_dup", "b"], threshold=0.9, method="independent_set")
    assert kept.shape[1] == 2
    assert "b" in names
    assert names.count("a") + names.count("a_dup") == 1


def test_tvn_efaar_full_pca_on_controls():
    rng = np.random.default_rng(2)
    n, d = 60, 8
    X = rng.normal(size=(n, d))
    controls = np.zeros(n, dtype=bool)
    controls[:20] = True
    batches = np.array(["p"] * 30 + ["q"] * 30)
    Y, names = tvn_efaar_full(X, controls, batches, n_components=4, epsilon=0.5)
    assert Y.shape == (n, 4)
    assert names == ["PC_001", "PC_002", "PC_003", "PC_004"]
    assert np.isfinite(Y).all()


def test_paper_cp_preset_runs_on_tiny_table():
    rng = np.random.default_rng(3)
    n = 40
    plates = ["P1"] * 20 + ["P2"] * 20
    wells = [f"A{i:02d}" for i in range(20)] * 2
    is_ctrl = np.array([True] * 8 + [False] * 12 + [True] * 8 + [False] * 12)
    df = pl.DataFrame(
        {
            "Metadata_Source": ["test_source"] * n,
            "Metadata_Batch": plates,
            "Metadata_Plate": plates,
            "Metadata_Well": wells,
            "Metadata_control_type": ["negcon" if c else "trt" for c in is_ctrl],
            "Metadata_negcon": is_ctrl.tolist(),
            "feat_0000": rng.normal(size=n),
            "feat_0001": rng.normal(size=n),
            "feat_0002": rng.normal(size=n),
            "feat_0003": rng.normal(size=n),
        }
    )
    out = process_profiles(df, preset="paper_cp_default")
    feat_cols = [c for c in out.columns if c.startswith("PC_")]
    assert feat_cols
    assert out.height == n
    assert np.isfinite(out.select(feat_cols).to_numpy()).all()


def test_preset_overrides_change_tvn_width():
    rng = np.random.default_rng(4)
    n = 40
    plates = ["P1"] * 20 + ["P2"] * 20
    wells = [f"A{i:02d}" for i in range(20)] * 2
    is_ctrl = np.array([True] * 8 + [False] * 12 + [True] * 8 + [False] * 12)
    df = pl.DataFrame(
        {
            "Metadata_Source": ["test_source"] * n,
            "Metadata_Batch": plates,
            "Metadata_Plate": plates,
            "Metadata_Well": wells,
            "Metadata_control_type": ["negcon" if c else "trt" for c in is_ctrl],
            "Metadata_negcon": is_ctrl.tolist(),
            "feat_0000": rng.normal(size=n),
            "feat_0001": rng.normal(size=n),
            "feat_0002": rng.normal(size=n),
            "feat_0003": rng.normal(size=n),
            "feat_0004": rng.normal(size=n),
            "feat_0005": rng.normal(size=n),
        }
    )
    wide = process_profiles(
        df,
        preset="paper_cp_default",
        overrides=["tvn_n_components=6", "prune_correlated=false"],
    )
    narrow = process_profiles(
        df,
        preset="paper_cp_default",
        preset_overrides={"tvn_n_components": 3, "prune_correlated": False},
    )
    assert len([c for c in wide.columns if c.startswith("PC_")]) == 6
    assert len([c for c in narrow.columns if c.startswith("PC_")]) == 3


def test_corrcoef_cpu_finite():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 5))
    corr = _corrcoef(X, device="cpu")
    assert corr.shape == (5, 5)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-6)


def test_mps_corrcoef_and_pca_match_cpu_loosely():
    torch = pytest.importorskip("torch")
    if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")
    rng = np.random.default_rng(6)
    X = rng.normal(size=(80, 8))
    cpu = _corrcoef(X, device="cpu")
    mps = _corrcoef(X, device=resolve_device("mps"))
    assert np.allclose(cpu, mps, atol=1e-3, rtol=1e-3)
    Y_cpu = _pca_fit_transform(X, X, 4, device="cpu")
    Y_mps = _pca_fit_transform(X, X, 4, device="mps")
    from scipy.spatial.distance import pdist

    assert np.allclose(pdist(Y_cpu), pdist(Y_mps), atol=1e-2, rtol=1e-2)
