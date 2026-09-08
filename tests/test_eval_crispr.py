from __future__ import annotations

import numpy as np
import polars as pl

from jumpbench.eval.metrics import PAPER_PA_CRISPR, evaluate_profiles


def test_evaluate_crispr_paper_ref():
    rng = np.random.default_rng(0)
    compounds = ["g0"] * 4 + ["g1"] * 4 + ["g2"] * 4 + ["g3"] * 4 + ["DMSO"] * 8
    n = len(compounds)
    df = pl.DataFrame(
        {
            "Metadata_JCP2022": compounds,
            "Metadata_negcon": [c == "DMSO" for c in compounds],
            "Metadata_Plate": ["P1"] * n,
            "feat_0000": rng.normal(size=n),
            "feat_0001": rng.normal(size=n),
        }
    )
    out = evaluate_profiles(df, tasks=("pa",), group_col=None, paper_ref="crispr")
    assert out["pa"]["paper_nap"] == PAPER_PA_CRISPR
    assert "delta_vs_paper" in out["pa"]
    assert out["pa"]["subset"] == "crispr"
    assert np.isfinite(out["pa"]["mean_nap"])
