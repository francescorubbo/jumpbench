from __future__ import annotations

from pathlib import Path

import pytest

from jumpbench.data.download import (
    DownloadPlan,
    classify_task,
    confirm_download,
    estimate_remaining_bytes,
    format_download_plan,
)
from jumpbench.data.s3util import (
    download_s3_file,
    format_bytes,
    part_path,
    read_s3_bytes,
    unsigned_s3,
)


def test_format_bytes():
    assert format_bytes(0) == "0 B"
    assert format_bytes(1024) == "1.0 KiB"
    assert format_bytes(2_768_896) == "2.6 MiB"


def test_gallery_client_pins_us_east_1():
    client = unsigned_s3()
    assert client.meta.region_name == "us-east-1"


def test_classify_complete_partial_missing(tmp_path: Path):
    dest = tmp_path / "images"
    dest.mkdir()
    site = "source_13__batch__plate__A02__0"
    complete = dest / f"{site}__DNA.tif"
    complete.write_bytes(b"x" * 200_000)
    assert classify_task(dest, site, "DNA", "k", codec="raw").status == "complete"

    missing_site = "source_13__batch__plate__A02__1"
    assert classify_task(dest, missing_site, "DNA", "k", codec="raw").status == "missing"

    partial_site = "source_13__batch__plate__A02__2"
    part = part_path(dest / f"{partial_site}__DNA.tif")
    part.write_bytes(b"partial")
    task = classify_task(dest, partial_site, "DNA", "k", codec="raw")
    assert task.status == "partial"
    assert task.local_bytes == 7

    tiny = dest / f"{missing_site}__AGP.tif"
    tiny.write_bytes(b"tiny")
    assert classify_task(dest, missing_site, "AGP", "k", codec="raw").status == "missing"


def test_classify_jpegxl_complete_and_reclaim_tiff(tmp_path: Path):
    dest = tmp_path / "images"
    dest.mkdir()
    site = "source_13__batch__plate__A02__0"
    jxl = dest / f"{site}__DNA.jxl"
    jxl.write_bytes(b"x" * 200)
    assert classify_task(dest, site, "DNA", "k", codec="jpegxl_mq").status == "complete"

    leftover = dest / f"{site}__AGP.tif"
    leftover.write_bytes(b"x" * 200_000)
    task = classify_task(dest, site, "AGP", "k", codec="jpegxl_mq")
    assert task.status == "reclaim"
    assert task.local_tiff == leftover
    assert task.dest.suffix == ".jxl"


def test_estimate_remaining_from_sample():
    bytes_, mean, src = estimate_remaining_bytes(10, [1000, 3000])
    assert mean == 2000
    assert bytes_ == 20_000
    assert "HEAD" in src
    zero, _, none = estimate_remaining_bytes(0, [1])
    assert zero == 0
    assert none == "none"


class _FakeBody:
    def __init__(self, data: bytes):
        self.data = data

    def iter_chunks(self, chunk_size: int = 65536):
        for i in range(0, len(self.data), chunk_size):
            yield self.data[i : i + chunk_size]

    def close(self):
        return None


class _FakeS3:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.gets: list[dict] = []

    def get_object(self, Bucket, Key, **extra):
        self.gets.append(extra)
        start = 0
        if "Range" in extra:
            start = int(extra["Range"].removeprefix("bytes=").split("-")[0])
        body = self.payload[start:]
        return {"Body": _FakeBody(body), "ContentLength": len(body)}


def test_download_writes_through_part_and_resumes(tmp_path: Path):
    dest = tmp_path / "a.tif"
    payload = b"hello world JUMP"
    client = _FakeS3(payload)
    part = part_path(dest)
    part.write_bytes(payload[:5])
    download_s3_file("cellpainting-gallery", "key", dest, client=client)
    assert dest.read_bytes() == payload
    assert not part.exists()
    assert client.gets[0]["Range"] == "bytes=5-"


def test_download_skips_complete_file(tmp_path: Path):
    dest = tmp_path / "a.tif"
    dest.write_bytes(b"x" * 200_000)
    client = _FakeS3(b"new")
    download_s3_file("cellpainting-gallery", "key", dest, client=client)
    assert dest.read_bytes() == b"x" * 200_000
    assert client.gets == []


class _TimeoutThenOk:
    def __init__(self, payload: bytes, fail_times: int = 2):
        self.payload = payload
        self.fail_times = fail_times
        self.gets = 0

    def get_object(self, Bucket, Key, **extra):
        from botocore.exceptions import ReadTimeoutError

        self.gets += 1
        fail = self.gets <= self.fail_times

        class _Body:
            def __init__(self, data: bytes, boom: bool):
                self.data = data
                self.boom = boom

            def iter_chunks(self, chunk_size: int = 65536):
                if self.boom:
                    raise ReadTimeoutError(endpoint_url=None)
                for i in range(0, len(self.data), chunk_size):
                    yield self.data[i : i + chunk_size]

            def close(self):
                return None

        return {"Body": _Body(self.payload, fail), "ContentLength": len(self.payload)}


def test_read_s3_bytes_retries_read_timeout(monkeypatch):
    monkeypatch.setattr("jumpbench.data.s3util._backoff", lambda _attempt: None)
    payload = b"tiff-bytes-here"
    client = _TimeoutThenOk(payload, fail_times=2)
    reported: list[int] = []
    got = read_s3_bytes("cellpainting-gallery", "key", client=client, progress=reported.append)
    assert got == payload
    assert client.gets == 3
    assert reported == [len(payload)]


def test_read_s3_bytes_gives_up_after_attempts(monkeypatch):
    from botocore.exceptions import ReadTimeoutError

    monkeypatch.setattr("jumpbench.data.s3util._backoff", lambda _attempt: None)
    monkeypatch.setattr("jumpbench.data.s3util.GET_ATTEMPTS", 3)
    client = _TimeoutThenOk(b"x", fail_times=99)
    with pytest.raises(ReadTimeoutError):
        read_s3_bytes("cellpainting-gallery", "key", client=client)
    assert client.gets == 3


def test_confirm_yes_skips_prompt_and_no_refuses(monkeypatch):
    plan = DownloadPlan(
        dest=Path("/tmp"),
        index_path=Path("/tmp/i.parquet"),
        stats={
            "site_set": "all",
            "sites": 9,
            "wells": 1,
            "sites_per_well_min": 9,
            "sites_per_well_median": 9,
            "sites_per_well_max": 9,
        },
        tasks=[],
        remaining_n=3,
        estimated_remaining_bytes=10_000,
        estimated_s3_bytes=100_000,
        free_bytes=10**12,
        codec="jpegxl_mq",
    )
    assert confirm_download(plan, yes=True)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert confirm_download(plan, yes=False, interactive=True) is False
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert confirm_download(plan, yes=False, interactive=True) is True
    with pytest.raises(SystemExit):
        confirm_download(plan, yes=False, interactive=False)
    text = format_download_plan(plan)
    assert "Disk remaining: 3" in text
    assert "S3 remaining" in text
    assert "jpegxl_mq" in text
