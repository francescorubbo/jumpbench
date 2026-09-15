from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from jumpbench.eval.metrics import evaluate_profiles


def test_evaluate_pa_omits_paper_nap():
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
    out = evaluate_profiles(df, tasks=("pa",), group_col=None)
    assert "paper_nap" not in out["pa"]
    assert "delta_vs_paper" not in out["pa"]
    assert np.isfinite(out["pa"]["mean_nap"])


def _mixed_group_profiles() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    well = 0
    for plate, group, ptype, perts in (
        ("P1", "group_crispr", "crispr", ["g0", "g1"]),
        ("P2", "group_low", "compound", ["c0", "c1"]),
    ):
        for pert in perts:
            for _ in range(4):
                well += 1
                rows.append(
                    {
                        "Metadata_Source": "s",
                        "Metadata_Batch": "b",
                        "Metadata_Plate": plate,
                        "Metadata_Well": f"W{well:02d}",
                        "Metadata_JCP2022": pert,
                        "Metadata_negcon": False,
                        "Metadata_Group": group,
                        "Metadata_Perturbation_Type": ptype,
                    }
                )
        for _ in range(8):
            well += 1
            rows.append(
                {
                    "Metadata_Source": "s",
                    "Metadata_Batch": "b",
                    "Metadata_Plate": plate,
                    "Metadata_Well": f"W{well:02d}",
                    "Metadata_JCP2022": "DMSO",
                    "Metadata_negcon": True,
                    "Metadata_Group": group,
                    "Metadata_Perturbation_Type": ptype,
                }
            )
    df = pl.DataFrame(rows)
    return df.with_columns(
        pl.Series("feat_0000", rng.normal(size=df.height)),
        pl.Series("feat_0001", rng.normal(size=df.height)),
    )


def test_evaluate_subset_crispr_drops_other_groups():
    df = _mixed_group_profiles()
    mixed = evaluate_profiles(df, tasks=("pa",), group_col=None)
    crispr = evaluate_profiles(df, tasks=("pa",), subset="crispr", group_col=None)
    assert mixed["pa"]["n_perturbations"] == 4
    assert crispr["pa"]["n_perturbations"] == 2
    assert crispr["pa"]["subset"] == "crispr"
    assert "paper_nap" not in crispr["pa"]
    assert np.isfinite(crispr["pa"]["mean_nap"])


def test_evaluate_unknown_subset_raises():
    df = _mixed_group_profiles()
    with pytest.raises(ValueError, match="Unknown subset"):
        evaluate_profiles(df, tasks=("pa",), subset="orf")
