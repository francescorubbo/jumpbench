from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import tifffile

from jumpbench.cli import _site_filters, build_parser
from jumpbench.config import apply_overrides, load_models_config
from jumpbench.data.images import CHANNEL_FILES
from jumpbench.embed.generate import generate_embeddings
from jumpbench.embed.loader import S3TiffLoader, iter_loaded_sites
from jumpbench.embed.sites import iter_embed_sites


def _tiff_bytes(value: int, size: int = 64) -> bytes:
    buf = BytesIO()
    tifffile.imwrite(buf, np.full((size, size), value, dtype=np.uint16))
    return buf.getvalue()


def _site_index(site_key: str, prefix: str = "fake") -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "Metadata_Site_Key": site_key,
                "channel": channel,
                "s3_key": f"{prefix}/{site_key}/{channel}.tif",
            }
            for channel in CHANNEL_FILES
        ]
    )


def _write_jxl_site(root: Path, site_key: str, value: int = 1) -> None:
    source, batch, plate, well, site = site_key.split("__")
    dest = root / source / batch / plate / well
    dest.mkdir(parents=True, exist_ok=True)
    for channel in CHANNEL_FILES:
        (dest / f"{site}__{channel}.jxl").write_bytes(f"not-a-real-jxl-{value}".encode())


def test_embed_cli_image_source_s3():
    parser = build_parser()
    args = parser.parse_args(
        [
            "embed",
            "--model",
            "timm",
            "--image-source",
            "s3",
            "--crop",
            "cell_fixed",
            "--crop-size",
            "96",
            "--prefetch-jobs",
            "8",
        ]
    )
    assert args.image_source == "s3"
    assert args.prefetch_jobs == 8
    assert args.crop_size == 96


def test_index_cli_subset_batch_uncaps():
    parser = build_parser()
    args = parser.parse_args(
        [
            "index-images",
            "--subset",
            "crispr",
            "--sites",
            "jump_lite",
            "--batch",
            "20220914_Run1",
        ]
    )
    filters = _site_filters(args)
    assert filters["max_sites"] is None
    assert filters["max_wells"] is None
    assert filters["subset"] == "crispr"
    assert filters["batches"] == ["20220914_Run1"]
    download = parser.parse_args(
        ["download-images", "--codec", "raw", "--subset", "crispr", "--sites", "jump_lite"]
    )
    assert download.subset == "crispr"
    assert download.codec == "raw"


def test_s3_grid_skips_all_fov():
    with pytest.raises(ValueError, match="S3 embed supports --sites jump_lite"):
        list(
            iter_embed_sites(
                Path("."),
                "grid",
                site_set="all",
                require_local=False,
                subset="crispr",
            )
        )


def test_s3_cell_sites_skip_local_probe(monkeypatch):
    site = "source_13__20220914_Run1__CP-CC9-R1-01__A02__0"
    monkeypatch.setattr(
        "jumpbench.embed.sites.mask_sites",
        lambda **_k: pl.DataFrame({"Metadata_Site_Key": [site]}),
    )

    def _boom(*_a, **_k):
        raise AssertionError("must not probe local images")

    monkeypatch.setattr("jumpbench.embed.sites.site_has_images", _boom)
    keys = list(
        iter_embed_sites(Path("/nope"), "cell_fixed", site_set="jump_lite", require_local=False)
    )
    assert keys == [site]


def test_s3_embed_ignores_local_jxl(tmp_path: Path, monkeypatch):
    site = "s__b__p__A01__1"
    images = tmp_path / "images"
    _write_jxl_site(images, site, value=1)
    payloads = {f"fake/{site}/{ch}.tif": _tiff_bytes(500) for ch in CHANNEL_FILES}

    def _read(_bucket: str, key: str, **_k) -> bytes:
        return payloads[key]

    def _local_boom(*_a, **_k):
        raise AssertionError("local JPEG XL must not be loaded")

    monkeypatch.setattr("jumpbench.embed.loader.read_s3_bytes", _read)
    monkeypatch.setattr("jumpbench.embed.generate.load_site_images", _local_boom)
    monkeypatch.setattr("jumpbench.data.images.load_site_images", _local_boom)
    monkeypatch.setattr(
        "jumpbench.embed.generate.build_tiff_index",
        lambda **_k: (_ for _ in ()).throw(AssertionError("must use injected index")),
    )
    cfg = apply_overrides(load_models_config(), ["models.dummy.tile_size=32"])
    run_dir = tmp_path / "run"
    out = generate_embeddings(
        "dummy",
        images,
        tmp_path / "embeddings",
        models_cfg=cfg,
        codec="jpegxl_mq",
        pool="site",
        run_dir=run_dir,
        site_keys=[site],
        image_source="s3",
        site_index=_site_index(site),
        prefetch_jobs=1,
    )
    table = pl.read_parquet(out)
    assert table.height == 1
    provenance = (run_dir / "provenance.json").read_text()
    assert '"image_source": "s3"' in provenance
    assert '"persist_codec": "raw"' in provenance
    assert '"input_hw": 32' in provenance


def test_s3_cell_crop_uses_streamed_tiff(tmp_path: Path, monkeypatch):
    site = "s__b__p__A01__0"
    image = np.zeros((5, 32, 32), dtype=np.uint16)
    image[:, 8:16, 8:16] = 400
    mask = np.zeros((32, 32), dtype=np.uint16)
    mask[8:16, 8:16] = 1
    buf = BytesIO()
    tifffile.imwrite(buf, np.full((32, 32), 400, dtype=np.uint16))
    payload = buf.getvalue()

    monkeypatch.setattr("jumpbench.embed.loader.read_s3_bytes", lambda *_a, **_k: payload)
    monkeypatch.setattr(
        "jumpbench.embed.generate.iter_embed_sites",
        lambda *_a, **_k: iter([site]),
    )
    monkeypatch.setattr(
        "jumpbench.embed.generate.cache_masks",
        lambda *_a, **_k: {"n": 1, "hit": 1, "fetched": 0, "missing": 0},
    )
    monkeypatch.setattr("jumpbench.embed.generate.load_mask", lambda *_a, **_k: mask)
    monkeypatch.setattr(
        "jumpbench.embed.generate.load_site_images",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("local load")),
    )
    cfg = apply_overrides(load_models_config(), ["models.dummy.tile_size=16"])
    out = generate_embeddings(
        "dummy",
        tmp_path / "images",
        tmp_path / "embeddings",
        models_cfg=cfg,
        crop="cell_fixed",
        crop_size=16,
        image_source="s3",
        site_index=_site_index(site),
        pool="site",
        prefetch_jobs=2,
    )
    table = pl.read_parquet(out)
    assert table.height == 1
    assert int(table["n_crops"][0]) == 1


def test_s3_embed_resume_skips_completed(tmp_path: Path, monkeypatch):
    a = "s__b__p__A01__1"
    b = "s__b__p__A01__2"
    fetched: list[str] = []
    payloads = {}
    for site in (a, b):
        for ch in CHANNEL_FILES:
            payloads[f"fake/{site}/{ch}.tif"] = _tiff_bytes(10)

    def _read(_bucket: str, key: str, **_k) -> bytes:
        fetched.append(key)
        return payloads[key]

    monkeypatch.setattr("jumpbench.embed.loader.read_s3_bytes", _read)
    cfg = apply_overrides(load_models_config(), ["models.dummy.tile_size=32"])
    run_dir = tmp_path / "run"
    index = pl.concat([_site_index(a), _site_index(b)])
    generate_embeddings(
        "dummy",
        tmp_path / "images",
        tmp_path / "embeddings",
        models_cfg=cfg,
        pool="site",
        run_dir=run_dir,
        site_keys=[a],
        image_source="s3",
        site_index=index,
        prefetch_jobs=1,
    )
    fetched.clear()
    generate_embeddings(
        "dummy",
        tmp_path / "images",
        tmp_path / "embeddings",
        models_cfg=cfg,
        pool="site",
        run_dir=run_dir,
        site_keys=[a, b],
        image_source="s3",
        site_index=index,
        prefetch_jobs=1,
    )
    assert all(a not in key for key in fetched)
    assert any(b in key for key in fetched)
    table = pl.read_parquet(run_dir / "site_embeddings.parquet")
    assert sorted(table["site_key"].to_list()) == [a, b]


def test_s3_loader_missing_site():
    loader = S3TiffLoader(_site_index("s__b__p__A01__1"))
    with pytest.raises(FileNotFoundError, match="No Orig TIFF URI"):
        loader.load("missing__key")


def test_prefetch_yields_all_keys():
    seen: list[str] = []

    def _load(key: str) -> np.ndarray:
        seen.append(key)
        return np.zeros((1, 2, 2), dtype=np.uint16)

    keys = ["a", "b", "c", "d"]
    got = [key for key, _img in iter_loaded_sites(keys, _load, prefetch=3)]
    assert sorted(got) == keys
    assert sorted(seen) == keys
