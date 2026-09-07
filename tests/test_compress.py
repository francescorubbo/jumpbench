from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from jumpbench.data.compress import decode_image, encode_image
from jumpbench.data.images import list_local_sites, load_site_images


def test_jpegxl_mq_roundtrip_uint16():
    imagecodecs = pytest.importorskip("imagecodecs")
    if not getattr(imagecodecs, "JPEGXL", False) and not hasattr(imagecodecs, "jpegxl_encode"):
        pytest.skip("imagecodecs built without JPEG XL")
    y, x = np.ogrid[:64, :80]
    original = (800 + 400 * np.sin(x / 6.0) + 3 * y).astype(np.uint16)
    payload = encode_image(original, "jpegxl_mq")
    restored = decode_image(payload, "jpegxl_mq")
    assert restored.shape == original.shape
    assert restored.dtype == np.uint16
    assert payload
    mae = np.mean(np.abs(original.astype(np.int32) - restored.astype(np.int32)))
    assert mae < 40


def test_load_site_prefers_jxl_over_tif(tmp_path: Path):
    imagecodecs = pytest.importorskip("imagecodecs")
    if not getattr(imagecodecs, "JPEGXL", False) and not hasattr(imagecodecs, "jpegxl_encode"):
        pytest.skip("imagecodecs built without JPEG XL")
    site = "source_2__batchA__plate1__A01__1"
    channels = ("AGP", "DNA", "ER", "Mito", "RNA")
    y, x = np.ogrid[:16, :16]
    for i, ch in enumerate(channels):
        arr = (200 + 50 * i + x + y).astype(np.uint16)
        tifffile.imwrite(tmp_path / f"{site}__{ch}.tif", np.zeros_like(arr))
        (tmp_path / f"{site}__{ch}.jxl").write_bytes(encode_image(arr, "jpegxl_mq"))
    stacked = load_site_images(tmp_path, site)
    assert stacked.shape == (5, 16, 16)
    assert list_local_sites(tmp_path) == [site]
    assert stacked.mean() > 100


def test_nested_layout_and_flat_fallback(tmp_path: Path):
    from jumpbench.data.images import image_path, migrate_flat_images

    site = "source_13__batchA__plate1__A02__0"
    nested = image_path(tmp_path, site, "DNA", "jpegxl_mq")
    assert nested == tmp_path / "source_13" / "batchA" / "plate1" / "A02" / "0__DNA.jxl"
    flat = tmp_path / f"{site}__DNA.jxl"
    flat.write_bytes(b"x" * 50)
    assert list_local_sites(tmp_path) == [site]
    moved = migrate_flat_images(tmp_path, show_progress=False)
    assert moved == 1
    assert nested.exists()
    assert not flat.exists()
    assert list_local_sites(tmp_path) == [site]


def test_persist_encoded_writes_jxl(tmp_path: Path):
    pytest.importorskip("imagecodecs")
    from jumpbench.data.compress import get_codec
    from jumpbench.data.download import _persist_encoded

    y, x = np.ogrid[:32, :32]
    array = (400 + x + y).astype(np.uint16)
    dest = tmp_path / "site__DNA.jxl"
    _persist_encoded(array, dest, get_codec("jpegxl_mq"))
    assert dest.exists()
    assert dest.stat().st_size > 0
    assert not dest.with_name(dest.name + ".part").exists()
