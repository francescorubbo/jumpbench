from __future__ import annotations

import json
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
from jumpbench.embed.loader import (
    S3TiffLoader,
    decode_jump_lite_mq_array,
    iter_loaded_sites,
    mq_store_prefix,
    site_load_fn,
)
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


def _mq_zarr_objects(stack: np.ndarray) -> dict[str, bytes]:
    import imagecodecs

    encoded = bytes(imagecodecs.jpegxl_encode(stack, lossless=True))
    meta = {
        "shape": list(stack.shape),
        "chunks": list(stack.shape),
        "dtype": "<u2",
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "dimension_separator": ".",
        "compressor": {"id": "imagecodecs_jpegxl", "lossless": True},
        "zarr_format": 2,
    }
    return {".zarray": json.dumps(meta).encode(), ".zattrs": b"{}", "0.0.0": encoded}


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


def test_embed_cli_image_source_s3_mq():
    parser = build_parser()
    args = parser.parse_args(["embed", "--model", "timm", "--image-source", "s3_mq"])
    assert args.image_source == "s3_mq"


def test_decode_jump_lite_mq_array_roundtrip():
    stack = np.arange(5 * 8 * 8, dtype=np.uint16).reshape(5, 8, 8)
    objs = _mq_zarr_objects(stack)
    out = decode_jump_lite_mq_array(objs[".zarray"], objs["0.0.0"])
    np.testing.assert_array_equal(out, stack)


def test_s3_mq_loader_reads_zarr_objects(monkeypatch):
    site = "s__b__p__A01__0"
    stack = np.arange(5 * 8 * 8, dtype=np.uint16).reshape(5, 8, 8)
    objs = _mq_zarr_objects(stack)
    fetched: list[str] = []

    def _read(_bucket: str, key: str, **_k) -> bytes:
        fetched.append(key)
        return objs[key.rsplit("/", 1)[-1]]

    monkeypatch.setattr("jumpbench.embed.loader.read_s3_bytes", _read)
    load, close = site_load_fn("s3_mq", Path("."), None)
    try:
        out = load(site)
    finally:
        close()
    np.testing.assert_array_equal(out, stack)
    prefix = mq_store_prefix()
    assert f"{prefix}/{site}/.zarray" in fetched
    assert f"{prefix}/{site}/0.0.0" in fetched


def test_s3_mq_grid_all_sites_rejected(tmp_path: Path):
    cfg = apply_overrides(load_models_config(), ["models.dummy.tile_size=16"])
    with pytest.raises(ValueError, match="s3_mq is JUMP-lite 4-site only"):
        generate_embeddings(
            "dummy",
            tmp_path,
            tmp_path / "embeddings",
            models_cfg=cfg,
            image_source="s3_mq",
            crop="grid",
            site_set="all",
        )


def test_s3_mq_embed_skips_tiff_index(tmp_path: Path, monkeypatch):
    site = "s__b__p__A01__0"
    stack = np.zeros((5, 32, 32), dtype=np.uint16)
    stack[:, 4:12, 4:12] = 200
    objs = _mq_zarr_objects(stack)

    def _read(_bucket: str, key: str, **_k) -> bytes:
        return objs[key.rsplit("/", 1)[-1]]

    monkeypatch.setattr("jumpbench.embed.loader.read_s3_bytes", _read)
    monkeypatch.setattr(
        "jumpbench.embed.generate.iter_embed_sites",
        lambda *_a, **_k: iter([site]),
    )
    monkeypatch.setattr(
        "jumpbench.embed.generate.build_tiff_index",
        lambda **_k: (_ for _ in ()).throw(AssertionError("s3_mq must not build a TIFF index")),
    )
    monkeypatch.setattr(
        "jumpbench.embed.generate.load_site_images",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("local load")),
    )
    cfg = apply_overrides(load_models_config(), ["models.dummy.tile_size=32"])
    run_dir = tmp_path / "run"
    out = generate_embeddings(
        "dummy",
        tmp_path / "images",
        tmp_path / "embeddings",
        models_cfg=cfg,
        crop="grid",
        site_set="jump_lite",
        image_source="s3_mq",
        pool="site",
        run_dir=run_dir,
        prefetch_jobs=1,
    )
    table = pl.read_parquet(out)
    assert table.height == 1
    provenance = (run_dir / "provenance.json").read_text()
    assert '"image_source": "s3_mq"' in provenance
    assert '"persist_codec": "jpegxl_mq"' in provenance
    assert "jpegxl_lossy_mq.zarr" in provenance


def test_s3_mq_embed_skips_missing_site(tmp_path: Path, monkeypatch):
    present = "s__b__p__A01__1"
    missing = "s__b__p__A01__2"
    store: dict[str, bytes] = {}
    for name, payload in _mq_zarr_objects(np.full((5, 32, 32), 40, dtype=np.uint16)).items():
        store[f"{mq_store_prefix()}/{present}/{name}"] = payload

    def _read(_bucket: str, key: str, **_k) -> bytes:
        from botocore.exceptions import ClientError

        if key not in store:
            raise ClientError(
                {
                    "Error": {"Code": "404", "Message": "Not Found"},
                    "ResponseMetadata": {"HTTPStatusCode": 404},
                },
                "GetObject",
            )
        return store[key]

    monkeypatch.setattr("jumpbench.embed.loader.read_s3_bytes", _read)
    monkeypatch.setattr(
        "jumpbench.embed.generate.iter_embed_sites",
        lambda *_a, **_k: iter([missing, present]),
    )
    monkeypatch.setattr(
        "jumpbench.embed.generate.load_site_images",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("local load")),
    )
    cfg = apply_overrides(load_models_config(), ["models.dummy.tile_size=32"])
    run_dir = tmp_path / "run"
    out = generate_embeddings(
        "dummy",
        tmp_path / "images",
        tmp_path / "embeddings",
        models_cfg=cfg,
        crop="grid",
        site_set="jump_lite",
        image_source="s3_mq",
        pool="site",
        run_dir=run_dir,
        prefetch_jobs=2,
    )
    table = pl.read_parquet(out)
    assert table["site_key"].to_list() == [present]
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert provenance["n_sites_skipped_no_image"] == 1
    assert provenance["n_sites_embedded"] == 1


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


def test_prefetch_skips_missing_with_callback():
    missing: list[str] = []

    def _load(key: str) -> np.ndarray:
        if key == "b":
            raise FileNotFoundError(key)
        return np.zeros((1, 2, 2), dtype=np.uint16)

    got = [
        key
        for key, _img in iter_loaded_sites(
            ["a", "b", "c"], _load, prefetch=3, on_missing=missing.append
        )
    ]
    assert sorted(got) == ["a", "c"]
    assert missing == ["b"]
