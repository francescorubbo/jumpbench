from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from jumpbench.config import apply_overrides, load_models_config, resolve_model
from jumpbench.embed.generate import generate_embeddings, preview_crops
from jumpbench.embed.preview import montage_rgb, write_png


def _dummy_card(**kwargs):
    card = resolve_model(
        "dummy",
        apply_overrides(load_models_config(), ["models.dummy.tile_size=16"]),
    )
    card.update(kwargs)
    return card


def test_montage_rgb_layout():
    tiles = np.zeros((3, 2, 8, 8), dtype=np.float32)
    tiles[0, 0] = 1.0
    rgb = montage_rgb(tiles, ncol=2, pad=2)
    assert rgb.shape == (22, 22, 3)
    assert rgb.dtype == np.uint8
    assert rgb.max() > 0


def test_write_png_magic(tmp_path):
    path = write_png(tmp_path / "m.png", np.zeros((4, 4, 3), dtype=np.uint8))
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_preview_crops_cell_fixed_writes_montage(tmp_path, monkeypatch):
    image = np.zeros((5, 64, 64), dtype=np.uint16)
    image[:, 24:40, 24:40] = 1000
    mask = np.zeros((64, 64), dtype=np.uint16)
    mask[24:40, 24:40] = 7
    monkeypatch.setattr("jumpbench.embed.generate.load_site_images", lambda *_a, **_k: image)
    card = _dummy_card(crop="cell_fixed", mask_object="cells")
    dest = tmp_path / "preview"
    out = preview_crops(
        card,
        tmp_path,
        ["s__b__p__A01__0"],
        dest,
        preview_n=4,
        mask=mask,
    )
    assert out == dest / "montage.png"
    assert out.exists()
    crops = pl.read_parquet(dest / "crops.parquet")
    assert crops.height == 1
    assert crops["object_label"].to_list() == [7]
    prov = (dest / "provenance.json").read_text()
    assert '"dry_run": true' in prov


def test_generate_embeddings_dry_run_skips_backend(tmp_path, monkeypatch):
    image = np.arange(5 * 64 * 64, dtype=np.uint16).reshape(5, 64, 64)
    monkeypatch.setattr("jumpbench.embed.generate.load_site_images", lambda *_a, **_k: image)
    monkeypatch.setattr(
        "jumpbench.embed.generate.iter_embed_sites",
        lambda *_a, **_k: iter(["s__b__p__A01__0"]),
    )

    def _fail_backend(_card):
        raise AssertionError("embedding backend should not be built in dry-run")

    monkeypatch.setattr("jumpbench.embed.generate.build_backend", _fail_backend)
    cfg = apply_overrides(load_models_config(), ["models.dummy.tile_size=32"])
    out = generate_embeddings(
        "dummy",
        tmp_path,
        tmp_path / "embeddings",
        models_cfg=cfg,
        dry_run=True,
        preview_n=2,
        crop="grid",
        codec="jpegxl_mq",
    )
    preview_dir = tmp_path / "embeddings" / "dummy" / "jpegxl_mq" / "preview"
    assert out == preview_dir / "montage.png"
    assert out.exists()
    crops = pl.read_parquet(preview_dir / "crops.parquet")
    assert crops.height == 2
    assert not (
        tmp_path / "embeddings" / "dummy" / "jpegxl_mq" / "site_embeddings.parquet"
    ).exists()


def test_preview_n_must_be_positive():
    card = _dummy_card(crop="grid")
    with pytest.raises(ValueError, match="preview_n"):
        preview_crops(card, Path("."), ["s__b__p__A01__0"], Path("."), preview_n=0)


def test_preview_prints_progress(tmp_path, monkeypatch, capsys):
    image = np.zeros((5, 64, 64), dtype=np.uint16)
    image[:, 24:40, 24:40] = 1000
    mask = np.zeros((64, 64), dtype=np.uint16)
    mask[24:40, 24:40] = 1
    monkeypatch.setattr("jumpbench.embed.generate.load_site_images", lambda *_a, **_k: image)
    preview_crops(
        _dummy_card(crop="cell_fixed"),
        tmp_path,
        ["s__b__p__A01__0"],
        tmp_path / "preview",
        preview_n=1,
        mask=mask,
    )
    err = capsys.readouterr().err
    assert "Collecting 1 preview crops" in err
    assert "load images" in err
    assert "Dry-run done" in err


def test_resolve_cell_sites_probes_paths_not_tree(tmp_path, monkeypatch):
    import tifffile

    from jumpbench.data.images import nested_relpath
    from jumpbench.embed.generate import resolve_embed_sites

    site = "source_2__batchA__plate1__A01__1"
    missing = "source_2__batchA__plate1__Z99__1"
    path = tmp_path / nested_relpath(site, "DNA", ".tif")
    path.parent.mkdir(parents=True)
    tifffile.imwrite(path, np.zeros((8, 8), dtype=np.uint16))

    monkeypatch.setattr(
        "jumpbench.embed.sites.mask_sites",
        lambda **_k: pl.DataFrame({"Metadata_Site_Key": [site, missing]}),
    )

    def _boom(_root):
        raise AssertionError("must not walk the image tree")

    monkeypatch.setattr("jumpbench.embed.sites.iter_local_sites", _boom)
    assert resolve_embed_sites(tmp_path, "cell_fixed") == [site]
