from jumpbench.data.metadata import load_perturbations, load_sites, load_wells
from jumpbench.profiles.normalize import attach_perturbation_metadata
import polars as pl


def test_frozen_cohort_counts():
    assert load_sites().height == 655_101
    assert load_wells().height == 163_776
    assert load_perturbations().height == 163_776


def test_negcon_flag_from_pert_type():
    wells = load_perturbations().head(50).select(
        "Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"
    )
    dummy = wells.with_columns(pl.lit(1.0).alias("feat_0000"))
    out = attach_perturbation_metadata(dummy)
    assert "Metadata_negcon" in out.columns
    assert "Metadata_JCP2022" in out.columns
    assert out["Metadata_negcon"].dtype == pl.Boolean
