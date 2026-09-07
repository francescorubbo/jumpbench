from __future__ import annotations

import polars as pl

from jumpbench.data.index import CHANNELS, load_data_key, restrict_to_site_set, tidy_load_data
from jumpbench.data.metadata import JOIN_SITE, JOIN_WELL, attach_site_keys, filter_wells, load_sites
from jumpbench.data.s3util import parse_s3_uri


def test_load_data_key_matches_jump_lite_sql():
    assert load_data_key("source_13", "20220914_Run1", "CP-CC9-R1-01") == (
        "cpg0016-jump/source_13/workspace/load_data_csv/20220914_Run1/"
        "CP-CC9-R1-01/load_data_with_illum.csv"
    )


def test_tidy_load_data_unpivots_orig_channels_and_drops_illum():
    df = pl.DataFrame(
        {
            "URL_OrigRNA": ["s3://cellpainting-gallery/cpg0016-jump/source_13/images/a/RNA.tif"],
            "URL_OrigMito": ["s3://cellpainting-gallery/cpg0016-jump/source_13/images/a/Mito.tif"],
            "URL_OrigER": ["s3://cellpainting-gallery/cpg0016-jump/source_13/images/a/ER.tif"],
            "URL_OrigDNA": ["s3://cellpainting-gallery/cpg0016-jump/source_13/images/a/DNA.tif"],
            "URL_OrigAGP": ["s3://cellpainting-gallery/cpg0016-jump/source_13/images/a/AGP.tif"],
            "URL_IllumDNA": ["s3://ignore/illum.npy"],
            "Metadata_Source": ["source_13"],
            "Metadata_Batch": ["20220914_Run1"],
            "Metadata_Plate": ["CP-CC9-R1-01"],
            "Metadata_Well": ["A02"],
            "Metadata_Site": [0],
        }
    )
    tidy = tidy_load_data(df)
    assert sorted(tidy["channel"].unique().to_list()) == sorted(CHANNELS)
    assert tidy.height == 5
    assert "illum" not in " ".join(tidy["uri"].to_list()).lower()
    agp = tidy.filter(pl.col("channel") == "AGP")["uri"][0]
    assert agp.endswith("AGP.tif")


def test_filter_wells_caps_without_touching_sites():
    wells = filter_wells(max_wells=3, sources=["source_13"])
    assert wells.height == 3
    assert wells.select(JOIN_WELL).n_unique() == 3


def _nine_site_catalog() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    well = {
        "Metadata_Source": "source_13",
        "Metadata_Batch": "20220914_Run1",
        "Metadata_Plate": "CP-CC9-R1-01",
        "Metadata_Well": "A02",
    }
    rows = []
    for site in range(9):
        for channel in CHANNELS:
            rows.append(
                {
                    **well,
                    "Metadata_Site": site,
                    "channel": channel,
                    "uri": (
                        "s3://cellpainting-gallery/cpg0016-jump/source_13/"
                        f"images/a/s{site}_{channel}.tif"
                    ),
                }
            )
    catalog = pl.DataFrame(rows)
    wells = pl.DataFrame([well])
    frozen = attach_site_keys(
        pl.DataFrame([{**well, "Metadata_Site": s} for s in range(4)])
    )
    return catalog, wells, frozen


def test_all_sites_keeps_every_fov_jump_lite_keeps_four():
    catalog, wells, frozen = _nine_site_catalog()
    all_sites = restrict_to_site_set(catalog, site_set="all", wells=wells)
    lite = restrict_to_site_set(
        catalog, site_set="jump_lite", wells=wells, frozen_sites=frozen
    )
    assert all_sites.select("Metadata_Site").n_unique() == 9
    assert lite.select("Metadata_Site").n_unique() == 4
    assert all_sites["Metadata_site_set"][0] == "all"
    assert lite["Metadata_site_set"][0] == "jump_lite"


def test_frozen_sites_have_join_keys():
    sites = load_sites()
    assert set(JOIN_SITE) <= set(sites.columns)
    assert sites.height == 655_101


def test_parse_gallery_s3_uri():
    bucket, key = parse_s3_uri(
        "s3://cellpainting-gallery/cpg0016-jump/source_13/images/20220914_Run1/images/p/x.tif"
    )
    assert bucket == "cellpainting-gallery"
    assert key.startswith("cpg0016-jump/")
