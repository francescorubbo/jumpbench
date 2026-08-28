from __future__ import annotations

import numpy as np
import pytest

from jumpbench.config import apply_overrides, load_models_config, resolve_model
from jumpbench.embed.preprocess import apply_preprocess, clip_percentile, standard
from jumpbench.embed.tiling import crop_tiles, reorder_channels, select_channels
from jumpbench.profiles.aggregate import aggregate_sites_to_wells
from jumpbench.profiles.cellprofiler import comparison_is_fair
import polars as pl


def test_dinov2_channel_recipes_disagree():
    as_run = resolve_model("dinov2", apply_overrides(load_models_config(), ["channel_recipe=jump_lite_as_run"]))
    paper = resolve_model("dinov2", apply_overrides(load_models_config(), ["channel_recipe=paper_table_s3"]))
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
    assert not comparison_is_fair("paper_as_published")
    assert not comparison_is_fair("wells_aligned_only")


def test_tile_size_override():
    cfg = apply_overrides(load_models_config(), ["models.dinov2.tile_size=256"])
    card = resolve_model("dinov2", cfg)
    assert card["tile_size"] == 256
