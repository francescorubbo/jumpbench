from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from jumpbench.config import apply_overrides, load_models_config, resolve_model
from jumpbench.embed.preprocess import (
    apply_preprocess,
    apply_preprocess_tiles,
    clip_percentile,
    percentile_minmax,
    standard,
)
from jumpbench.embed.tiling import crop_tiles, reorder_channels, select_channels
from jumpbench.profiles.aggregate import aggregate_sites_to_wells
from jumpbench.profiles.cellprofiler import comparison_is_fair


def test_dinov2_channel_recipes_disagree():
    as_run = resolve_model(
        "dinov2", apply_overrides(load_models_config(), ["channel_recipe=jump_lite_as_run"])
    )
    paper = resolve_model(
        "dinov2", apply_overrides(load_models_config(), ["channel_recipe=paper_table_s3"])
    )
    assert as_run["channels"] == ["AGP", "DNA", "ER"]
    assert paper["channels"] == ["DNA", "AGP", "Mito"]
    assert as_run["channels"] != paper["channels"]


def test_subcell_reorders_to_rybg_semantic_order():
    card = resolve_model("subcell")
    assert card["channels"] == ["AGP", "DNA", "ER", "Mito"]
    assert card["model_channel_order"] == ["Mito", "ER", "DNA", "AGP"]


def test_clip_percentile_and_standard():
    image = np.zeros((2, 8, 8), dtype=np.float32)
    image[0, 0, 0] = 1000
    image[0, 1, 1] = -1000
    clipped = clip_percentile(image, 0.5, 99.5)
    assert clipped.max() < 1000
    normed = standard(np.ones((1, 4, 4), dtype=np.float32) * 3)
    assert abs(float(normed.mean())) < 1e-5


def test_percentile_minmax_maps_percentiles_to_unit_interval():
    rng = np.random.default_rng(0)
    image = rng.normal(50.0, 10.0, size=(2, 32, 32)).astype(np.float32)
    image[0, 0, 0] = 10_000
    out = percentile_minmax(image, 1, 99)
    assert out.shape == image.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_percentile_minmax_tiles_are_independent():
    dim = np.linspace(0.0, 100.0, 64, dtype=np.float32).reshape(1, 8, 8)
    bright = np.linspace(500.0, 600.0, 64, dtype=np.float32).reshape(1, 8, 8)
    tiles = np.stack([dim, bright], axis=0)
    out = apply_preprocess_tiles(tiles, [{"op": "percentile_minmax", "low": 1, "high": 99}])
    assert out.shape == tiles.shape
    assert out[0].min() == pytest.approx(0.0, abs=0.05)
    assert out[0].max() == pytest.approx(1.0, abs=0.05)
    assert out[1].min() == pytest.approx(0.0, abs=0.05)
    assert out[1].max() == pytest.approx(1.0, abs=0.05)
    # Site-wide min-max would pin the dim crop near 0 and the bright crop near 1.
    assert float(out[0].mean()) == pytest.approx(float(out[1].mean()), abs=0.05)


def test_timm_card_is_bag_of_channels_per_crop():
    as_run = resolve_model("timm")
    paper = resolve_model(
        "timm", apply_overrides(load_models_config(), ["channel_recipe=paper_table_s3"])
    )
    assert as_run["channels"] == ["AGP", "DNA", "ER", "Mito", "RNA"]
    assert paper["channels"] == ["DNA", "AGP", "Mito", "RNA", "ER"]
    assert as_run["preprocess_scope"] == "tile"
    assert as_run["architecture"] == "resnet50"
    assert as_run["preprocess"] == [{"op": "percentile_minmax", "low": 1, "high": 99}]
    assert resolve_model("dinov2")["preprocess_scope"] == "site"


def test_apply_preprocess_openphenom_order():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 4096, size=(5, 32, 32)).astype(np.float32)
    out = apply_preprocess(
        image,
        [
            {"op": "clip_percentile", "low": 0.5, "high": 99.5},
            {"op": "rescale_minmax"},
            {"op": "to_8bit"},
            {"op": "standard"},
        ],
    )
    assert out.shape == image.shape
    assert np.isfinite(out).all()


def test_nonoverlapping_crops_drop_remainder():
    image = np.zeros((3, 500, 500), dtype=np.float32)
    tiles, coords = crop_tiles(image, 224)
    assert tiles.shape == (4, 3, 224, 224)
    assert len(coords) == 4


def test_channel_select_and_reorder():
    image = np.stack([np.full((2, 2), i, dtype=np.float32) for i in range(5)])
    selected = select_channels(image, [0, 1, 2, 3])  # AGP DNA ER Mito
    names = ["AGP", "DNA", "ER", "Mito"]
    reordered = reorder_channels(selected, names, ["Mito", "ER", "DNA", "AGP"])
    assert reordered[0, 0, 0] == 3  # Mito
    assert reordered[3, 0, 0] == 0  # AGP


def test_median_tile_then_site_aggregation():
    rows = []
    for site, values in (("1", [1.0, 3.0]), ("2", [10.0, 30.0])):
        for i, v in enumerate(values):
            rows.append(
                {
                    "Metadata_Source": "s",
                    "Metadata_Batch": "b",
                    "Metadata_Plate": "p",
                    "Metadata_Well": "A01",
                    "Metadata_Site": site,
                    "tile_y": i,
                    "tile_x": 0,
                    "feat_0000": v,
                }
            )
    df = pl.DataFrame(rows)
    wells = aggregate_sites_to_wells(df, how="median")
    assert wells.height == 1
    # site1 median=2, site2 median=20, well median=11
    assert wells["feat_0000"][0] == pytest.approx(11.0)


def test_fairness_flags():
    assert comparison_is_fair("fair_same_sites")
    assert comparison_is_fair("fair_all_sites")
    assert not comparison_is_fair("paper_as_published")
    assert not comparison_is_fair("wells_aligned_only")


def test_tile_size_override():
    cfg = apply_overrides(load_models_config(), ["models.dinov2.tile_size=256"])
    card = resolve_model("dinov2", cfg)
    assert card["tile_size"] == 256
