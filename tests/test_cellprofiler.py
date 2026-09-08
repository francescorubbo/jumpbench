"""CRISPR well filtering and assembled-CP alignment."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from jumpbench.data.metadata import load_perturbations
from jumpbench.profiles.cellprofiler import (
    align_paper_cellprofiler,
    filter_crispr_wells,
    load_paper_cellprofiler,
    restrict_to_jump_lite_wells,
)


def _wells() -> pl.DataFrame:
    rows = []
    for well, pert, ptype, neg in (
        ("A01", "g1", "CRISPR", False),
        ("A02", "g2", "CRISPR", False),
        ("A03", "DMSO", "CRISPR", True),
        ("B01", "c1", "COMPOUND", False),
        ("B02", "DMSO", "COMPOUND", True),
    ):
        rows.append(
            {
                "Metadata_Source": "source_13",
                "Metadata_Batch": "batchA",
                "Metadata_Plate": "P1" if ptype == "CRISPR" else "P2",
                "Metadata_Well": well,
                "Metadata_PlateType": ptype,
                "Metadata_JCP2022": pert,
                "Metadata_negcon": neg,
            }
        )
    return pl.DataFrame(rows)


def test_crispr_filter_keeps_plate_matched_negcons():
    out = filter_crispr_wells(_wells())
    keys = set(out["Metadata_Well"].to_list())
    assert keys == {"A01", "A02", "A03"}
    assert out.filter(pl.col("Metadata_negcon")).height == 1


def test_restrict_subset_crispr():
    profiles = (
        _wells()
        .select("Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well")
        .with_columns(pl.lit(1.0).alias("Cells_AreaShape_Area"))
    )
    aligned = restrict_to_jump_lite_wells(profiles, wells=_wells(), subset="crispr")
    assert aligned.height == 3


def test_align_paper_cellprofiler_crispr(tmp_path: Path, monkeypatch):
    wells = _wells()
    source = tmp_path / "assembled.parquet"
    wells.with_columns(pl.lit(1.5).alias("Nuclei_Intensity_Mean")).write_parquet(source)
    monkeypatch.setattr("jumpbench.profiles.cellprofiler.load_perturbations", lambda: wells)
    out = tmp_path / "aligned.parquet"
    align_paper_cellprofiler(source, out, subset="crispr")
    got = pl.read_parquet(out)
    assert got.height == 3
    assert "Nuclei_Intensity_Mean" in got.columns
    assert got["Metadata_model"][0] == "cellprofiler_paper"


def test_align_recovers_batch_from_wells(tmp_path: Path, monkeypatch):
    """CPG assembled v1.0c profiles omit Metadata_Batch."""
    wells = _wells()
    source = tmp_path / "assembled.parquet"
    wells.select(
        "Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"
    ).with_columns(pl.lit(2.0).alias("Cells_AreaShape_Area")).write_parquet(source)
    monkeypatch.setattr("jumpbench.profiles.cellprofiler.load_perturbations", lambda: wells)
    out = tmp_path / "aligned.parquet"
    align_paper_cellprofiler(source, out, subset="crispr")
    got = pl.read_parquet(out)
    assert got.height == 3
    assert got["Metadata_Batch"].to_list() == ["batchA"] * 3
    assert got["Metadata_id"][0] == "source_13__batchA__P1__A01"


def test_align_parses_site_column(tmp_path: Path, monkeypatch):
    wells = _wells()
    source = tmp_path / "assembled.parquet"
    pl.DataFrame(
        {
            "site": [
                "source_13__batchA__P1__A01__1",
                "source_13__batchA__P1__A02__1",
                "source_13__batchA__P2__B01__1",
            ],
            "Cells_AreaShape_Area": [1.0, 2.0, 3.0],
        }
    ).write_parquet(source)
    monkeypatch.setattr("jumpbench.profiles.cellprofiler.load_perturbations", lambda: wells)
    out = tmp_path / "aligned.parquet"
    align_paper_cellprofiler(source, out, subset="crispr")
    got = pl.read_parquet(out)
    assert set(got["Metadata_Well"].to_list()) == {"A01", "A02"}


def test_restrict_without_batch():
    wells = _wells()
    profiles = wells.select("Metadata_Source", "Metadata_Plate", "Metadata_Well").with_columns(
        pl.lit(1.0).alias("Cells_AreaShape_Area")
    )
    aligned = restrict_to_jump_lite_wells(profiles, wells=wells, subset="crispr")
    assert aligned.height == 3
    assert "Metadata_Batch" in aligned.columns


def test_load_paper_cellprofiler_unaligned(tmp_path: Path):
    path = tmp_path / "p.parquet"
    pl.DataFrame({"Metadata_Source": ["s"], "feat": [1.0]}).write_parquet(path)
    df = load_paper_cellprofiler(path, align_wells=False)
    assert df["Metadata_comparison_mode"][0] == "paper_assembled_unaligned"


def test_crispr_jcp_prefix_fallback():
    wells = pl.DataFrame(
        {
            "Metadata_Source": ["s"] * 4,
            "Metadata_Batch": ["b"] * 4,
            "Metadata_Plate": ["P1", "P1", "P1", "P2"],
            "Metadata_Well": ["A01", "A02", "A03", "B01"],
            "Metadata_JCP2022": [
                "JCP2022_807842",
                "JCP2022_800002",
                "JCP2022_033954",
                "JCP2022_033954",
            ],
            "Metadata_pert_type": ["trt", "negcon", "trt", "trt"],
            "Metadata_negcon": [False, True, False, False],
        }
    )
    out = filter_crispr_wells(wells)
    assert set(out["Metadata_Well"].to_list()) == {"A01", "A02"}


def test_frozen_crispr_subset_is_nonempty():
    wells = load_perturbations()
    try:
        crispr = filter_crispr_wells(wells)
    except ValueError as exc:
        pytest.skip(str(exc))
    assert crispr.height > 0
    assert crispr.height < wells.height
    n_neg = (
        crispr["Metadata_negcon"].sum()
        if "Metadata_negcon" in crispr.columns
        else crispr.filter(pl.col("Metadata_pert_type") == "negcon").height
        if "Metadata_pert_type" in crispr.columns
        else 0
    )
    assert n_neg > 0
