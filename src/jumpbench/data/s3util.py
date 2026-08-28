from __future__ import annotations

from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from boto3.s3.transfer import TransferConfig


def unsigned_s3():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def download_s3_file(
    bucket: str,
    key: str,
    dest: Path,
    expected_size: int | None = None,
    max_concurrency: int = 16,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
        return dest
    if dest.exists() and expected_size is None and dest.stat().st_size > 0:
        return dest
    s3 = unsigned_s3()
    cfg = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=max_concurrency,
        use_threads=True,
    )
    s3.download_file(bucket, key, str(dest), Config=cfg)
    if expected_size is not None and dest.stat().st_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for s3://{bucket}/{key}: "
            f"expected {expected_size}, got {dest.stat().st_size}"
        )
    return dest
