from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from jumpbench.cli import build_parser
from jumpbench.config import apply_overrides, load_models_config, resolve_model
from jumpbench.embed.backends import DummyBackend
from jumpbench.embed.crops import crop_cells_bbox, crop_cells_fixed, resize_tiles
from jumpbench.embed.generate import crop_site, embed_site
from jumpbench.profiles.aggregate import aggregate_sites_to_wells


def _blob(mask: np.ndarray, label: int, y0: int, y1: int, x0: int, x1: int) -> None:
    mask[y0:y1, x0:x1] = label


def test_fixed_crop_keeps_interior_and_skips_edge():
    image = np.zeros((2, 64, 64), dtype=np.float32)
    image[:, 20:28, 20:28] = 5.0
    image[:, 0:6, 0:6] = 9.0
    mask = np.zeros((64, 64), dtype=np.uint16)
    _blob(mask, 1, 20, 28, 20, 28)
    _blob(mask, 2, 0, 6, 0, 6)
    out = crop_cells_fixed(image, mask, crop_size=16)
    assert out.n_kept == 1
    assert out.n_skipped_edge == 1
    assert out.object_label.tolist() == [1]
    assert out.tiles.shape == (1, 2, 16, 16)
    cy, cx = float(out.centroid_y[0]), float(out.centroid_x[0])
    assert 23 <= cy <= 25
    assert 23 <= cx <= 25
    assert out.tiles[0, 0].max() == pytest.approx(5.0)


def test_bbox_crop_skips_margin_that_crosses_edge():
    image = np.zeros((1, 32, 32), dtype=np.float32)
    image[:, 8:12, 8:12] = 1.0
    image[:, 0:4, 0:4] = 2.0
    mask = np.zeros((32, 32), dtype=np.uint16)
    _blob(mask, 1, 8, 12, 8, 12)
    _blob(mask, 2, 0, 4, 0, 4)
    kept = crop_cells_bbox(image, mask, crop_size=16, margin=2)
    assert kept.object_label.tolist() == [1]
    assert kept.n_skipped_edge == 1
    assert kept.tiles.shape == (1, 1, 16, 16)
    skipped = crop_cells_bbox(image, mask, crop_size=16, margin=16)
    assert skipped.n_kept == 0
    assert skipped.n_skipped_edge == 2


def test_label_zero_is_ignored():
    image = np.ones((1, 32, 32), dtype=np.float32)
    mask = np.zeros((32, 32), dtype=np.uint16)
    out = crop_cells_fixed(image, mask, crop_size=8)
    assert out.n_kept == 0
    assert out.n_skipped_edge == 0


def test_dummy_embed_cell_fixed_with_injected_mask():
    card = resolve_model(
        "dummy",
        apply_overrides(load_models_config(), ["models.dummy.tile_size=16"]),
    )
    card["crop"] = "cell_fixed"
    backend = DummyBackend(card)
    image = np.arange(5 * 64 * 64, dtype=np.uint16).reshape(5, 64, 64)
    mask = np.zeros((64, 64), dtype=np.uint16)
    _blob(mask, 4, 24, 40, 24, 40)
    feats, extra, stats = embed_site(image, card, backend, site_key="s__b__p__A01__0", mask=mask)
    assert feats.shape[0] == 1
    assert extra["object_label"].tolist() == [4]
    assert stats["n_skipped_edge"] == 0
    assert stats["n_skipped_no_mask"] == 0


def test_dummy_embed_cell_bbox_with_injected_mask():
    card = resolve_model(
        "dummy",
        apply_overrides(load_models_config(), ["models.dummy.tile_size=16"]),
    )
    card["crop"] = "cell_bbox"
    card["crop_margin"] = 2
    backend = DummyBackend(card)
    image = np.ones((5, 64, 64), dtype=np.float32)
    mask = np.zeros((64, 64), dtype=np.uint16)
    _blob(mask, 1, 20, 28, 20, 28)
    feats, extra, stats = embed_site(image, card, backend, mask=mask)
    assert feats.shape[0] == 1
    assert extra["object_label"].tolist() == [1]
    assert stats["n_skipped_no_mask"] == 0


def test_resize_tiles_is_noop_when_already_sized():
    tiles = np.arange(2 * 1 * 8 * 8, dtype=np.float32).reshape(2, 1, 8, 8)
    assert resize_tiles(tiles, 8) is tiles


def test_crop_size_is_independent_of_model_tile_size():
    card = resolve_model(
        "dummy",
        apply_overrides(load_models_config(), ["models.dummy.tile_size=16"]),
    )
    card["crop"] = "cell_fixed"
    card["crop_size"] = 32
    image = np.zeros((5, 64, 64), dtype=np.uint16)
    image[:, 16:48, 16:48] = 1000
    mask = np.zeros((64, 64), dtype=np.uint16)
    _blob(mask, 1, 24, 40, 24, 40)
    tiles, extra, stats = crop_site(image, card, mask=mask)
    assert tiles.shape == (1, 3, 32, 32)
    assert extra["object_label"].tolist() == [1]
    assert stats["n_skipped_edge"] == 0

    seen = []

    class _ShapeBackend:
        embedding_dim = 4

        def embed_tiles(self, batch):
            seen.append(batch.shape)
            return np.zeros((batch.shape[0], 4), dtype=np.float32)

    feats, extra_b, _ = embed_site(image, card, _ShapeBackend(), mask=mask)
    assert feats.shape == (1, 4)
    assert seen == [(1, 3, 16, 16)]
    assert extra_b["object_label"].tolist() == [1]


def test_embed_site_missing_mask_skips_site(monkeypatch):
    card = resolve_model("dummy")
    card["crop"] = "cell_fixed"
    backend = DummyBackend(card)
    image = np.zeros((5, 32, 32), dtype=np.float32)
    monkeypatch.setattr("jumpbench.embed.generate.load_mask", lambda *_a, **_k: None)
    feats, extra, stats = embed_site(image, card, backend, site_key="s__b__p__A01__0")
    assert feats.shape[0] == 0
    assert stats["n_skipped_no_mask"] == 1
    assert extra["object_label"].shape[0] == 0


def test_embed_cli_crop_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "embed",
            "--model",
            "dummy",
            "--crop",
            "cell_bbox",
            "--object",
            "nuclei",
            "--crop-margin",
            "8",
        ]
    )
    assert args.crop == "cell_bbox"
    assert args.mask_object == "nuclei"
    assert args.crop_margin == 8
    grid = parser.parse_args(["embed", "--model", "dummy"])
    assert grid.crop == "grid"
    assert grid.mask_object == "cells"
    dry = parser.parse_args(["embed", "--model", "dummy", "--dry-run", "--preview-n", "8"])
    assert dry.dry_run is True
    assert dry.preview_n == 8
    sized = parser.parse_args(
        ["embed", "--model", "dummy", "--crop", "cell_fixed", "--crop-size", "96"]
    )
    assert sized.crop_size == 96
    assert parser.parse_args(["embed", "--model", "dummy"]).crop_size is None


def test_aggregate_object_label_median_then_site():
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
                    "object_label": i + 1,
                    "centroid_y": 0.0,
                    "centroid_x": 0.0,
                    "feat_0000": v,
                }
            )
    wells = aggregate_sites_to_wells(pl.DataFrame(rows), how="median")
    assert wells.height == 1
    assert wells["feat_0000"][0] == pytest.approx(11.0)
