from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import tifffile

from jumpbench.cli import build_parser
from jumpbench.config import apply_overrides, load_models_config
from jumpbench.data.metadata import filter_wells
from jumpbench.embed.generate import generate_embeddings
from jumpbench.embed.sites import iter_embed_sites, iter_local_sites_for_wells
from jumpbench.embed.tiling import grid_coverage


def _write_site(root: Path, site_key: str, size: int = 64) -> None:
    source, batch, plate, well, site = site_key.split("__")
    dest = root / source / batch / plate / well
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for channel in ("AGP", "DNA", "ER", "Mito", "RNA"):
        arr = rng.integers(0, 4096, size=(size, size), dtype=np.uint16)
        tifffile.imwrite(dest / f"{site}__{channel}.tif", arr)


def test_embed_cli_campaign_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "embed",
            "--model",
            "timm",
            "--subset",
            "crispr",
            "--sites",
            "jump_lite",
            "--batch",
            "20220914_Run1",
            "--pool",
            "site",
            "--run-dir",
            "data/embeddings/timm/run1/b0",
        ]
    )
    assert args.subset == "crispr"
    assert args.sites == "jump_lite"
    assert args.batch == ["20220914_Run1"]
    assert args.pool == "site"
    assert args.run_dir == Path("data/embeddings/timm/run1/b0")
    masks = parser.parse_args(["download-masks", "--batch", "20220914_Run1", "--subset", "crispr"])
    assert masks.batch == ["20220914_Run1"]


def test_filter_wells_batch_before_cap():
    wells = filter_wells(batches=["20220914_Run1"], sources=["source_13"], max_wells=10)
    assert wells.height == 10
    assert wells["Metadata_Batch"].n_unique() == 1
    assert wells["Metadata_Batch"][0] == "20220914_Run1"


def test_grid_coverage_448_keeps_less_than_224():
    small = grid_coverage(1080, 1280, 224)
    large = grid_coverage(1080, 1280, 448)
    assert small["n_tiles"] == 20
    assert large["n_tiles"] == 4
    assert large["fov_frac"] < small["fov_frac"]


def test_iter_local_sites_for_wells(tmp_path: Path):
    keep = "s__b__p__A01__0"
    other = "s__b__p__A02__3"
    _write_site(tmp_path, keep)
    _write_site(tmp_path, other)
    wells = pl.DataFrame(
        {
            "Metadata_Source": ["s"],
            "Metadata_Batch": ["b"],
            "Metadata_Plate": ["p"],
            "Metadata_Well": ["A01"],
        }
    )
    keys = list(iter_local_sites_for_wells(tmp_path, wells))
    assert keys == [keep]


def test_cell_crop_rejects_all_fov_site_set():
    with pytest.raises(ValueError, match="4-site"):
        list(iter_embed_sites(Path("."), "cell_fixed", site_set="all"))


def test_pool_site_and_run_dir(tmp_path: Path):
    site = "s__b__p__A01__1"
    _write_site(tmp_path / "images", site, size=64)
    cfg = apply_overrides(load_models_config(), ["models.dummy.tile_size=32"])
    run_dir = tmp_path / "run"
    out = generate_embeddings(
        "dummy",
        tmp_path / "images",
        tmp_path / "embeddings",
        models_cfg=cfg,
        codec="synth",
        pool="site",
        run_dir=run_dir,
        site_keys=[site],
    )
    table = pl.read_parquet(out)
    assert table.height == 1
    assert "n_crops" in table.columns
    assert int(table["n_crops"][0]) == 4
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert provenance["pool"] == "site"
    assert provenance["embedding_dim"] == 32
    assert provenance["fov_frac"] == pytest.approx(1.0)
    assert provenance["n_tiles_mean"] == pytest.approx(4.0)


def test_embed_resume_skips_completed(tmp_path: Path, monkeypatch):
    a = "s__b__p__A01__1"
    b = "s__b__p__A01__2"
    images = tmp_path / "images"
    _write_site(images, a, size=32)
    _write_site(images, b, size=32)
    cfg = apply_overrides(load_models_config(), ["models.dummy.tile_size=32"])
    run_dir = tmp_path / "run"
    generate_embeddings(
        "dummy",
        images,
        tmp_path / "embeddings",
        models_cfg=cfg,
        codec="synth",
        pool="site",
        run_dir=run_dir,
        site_keys=[a],
    )
    seen: list[str] = []
    real_load = __import__(
        "jumpbench.embed.generate", fromlist=["load_site_images"]
    ).load_site_images

    def wrapped(root, key, *args, **kwargs):
        seen.append(key)
        return real_load(root, key, *args, **kwargs)

    monkeypatch.setattr("jumpbench.embed.generate.load_site_images", wrapped)
    generate_embeddings(
        "dummy",
        images,
        tmp_path / "embeddings",
        models_cfg=cfg,
        codec="synth",
        pool="site",
        run_dir=run_dir,
        site_keys=[a, b],
    )
    assert seen == [b]
    table = pl.read_parquet(run_dir / "site_embeddings.parquet")
    assert sorted(table["site_key"].to_list()) == [a, b]
