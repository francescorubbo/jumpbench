from __future__ import annotations

import numpy as np
import pytest
from botocore.exceptions import ClientError

from jumpbench.data.masks import (
    decode_mask_array,
    load_mask,
    mask_s3_keys,
    mask_sites,
    mask_store_prefix,
)
from jumpbench.data.metadata import load_perturbations
from jumpbench.profiles.cellprofiler import filter_crispr_wells


def _client_error(code: str, status: int = 404) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": "missing"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "GetObject",
    )


def test_mask_s3_keys_cells_vs_nuclei():
    site = "source_13__20220914_Run1__CP-CC9-R1-01__A02__0"
    cells = mask_s3_keys(site, object_type="cells")
    nuclei = mask_s3_keys(site, object_type="nuclei")
    root = mask_store_prefix("cells")
    assert cells["metadata"].endswith(f"{site}/zarr.json")
    assert cells["chunk"].endswith(f"{site}/c/0/0/0")
    assert cells["metadata"].startswith("cpg0016-jump/source_all/")
    assert "/cell_masks/jpegxl_lossy_mq.zarr/" in cells["metadata"]
    assert "/nuclei_masks/jpegxl_lossy_mq.zarr/" in nuclei["metadata"]
    assert cells["chunk"] != nuclei["chunk"]
    assert root.endswith("cell_masks/jpegxl_lossy_mq.zarr")


def test_crispr_mask_sites_are_jump_lite_four_fovs():
    wells = filter_crispr_wells(load_perturbations())
    sites = mask_sites(subset="crispr")
    assert wells.height == 48_881
    assert sites.height == 195_524
    per_well = sites.group_by(
        ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"]
    ).len()
    assert per_well["len"].min() == 4
    assert per_well["len"].max() == 4


def test_load_mask_404_returns_none(monkeypatch):
    def boom(*_args, **_kwargs):
        raise _client_error("NoSuchKey")

    monkeypatch.setattr("jumpbench.data.masks.read_s3_bytes", boom)
    assert load_mask("source_13__20220914_Run1__CP-CC9-R1-01__A02__0") is None


def test_decode_mask_array_roundtrip():
    zarr = pytest.importorskip("zarr")
    from zarr.storage import MemoryStore

    src = MemoryStore()
    data = np.zeros((1, 8, 10), dtype=np.uint16)
    data[0, 2:5, 3:7] = 3
    arr = zarr.create_array(src, name="/", shape=data.shape, dtype="uint16", chunks=data.shape)
    arr[:] = data
    meta = src._store_dict["zarr.json"].to_bytes()
    chunk = src._store_dict["c/0/0/0"].to_bytes()
    got = decode_mask_array(meta, chunk)
    assert got.shape == (8, 10)
    assert got.dtype == np.uint16
    assert int((got == 3).sum()) == 12
