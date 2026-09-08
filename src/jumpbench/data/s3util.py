from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    IncompleteReadError,
    ReadTimeoutError,
    ResponseStreamingError,
)

GALLERY_BUCKET = "cellpainting-gallery"
# Public CPG objects live in us-east-1. boto3 otherwise inherits ~/.aws/config
# (here: us-west-2) and every GET 301-redirects before the body starts.
GALLERY_REGION = "us-east-1"
PART_SUFFIX = ".part"
MIN_COMPLETE_BYTES = 64 * 1024
CHUNK_SIZE = 256 * 1024
GET_ATTEMPTS = 8
STREAM_ERRORS = (
    ReadTimeoutError,
    ConnectTimeoutError,
    ConnectionClosedError,
    EndpointConnectionError,
    IncompleteReadError,
    ResponseStreamingError,
)

_thread_local = threading.local()


def unsigned_s3(max_pool_connections: int = 32):
    existing = getattr(_thread_local, "s3", None)
    pool = getattr(_thread_local, "pool", None)
    if existing is not None and pool == max_pool_connections:
        return existing
    client = boto3.client(
        "s3",
        region_name=GALLERY_REGION,
        config=Config(
            signature_version=UNSIGNED,
            region_name=GALLERY_REGION,
            connect_timeout=10,
            read_timeout=120,
            max_pool_connections=max(8, max_pool_connections),
            retries={"max_attempts": 8, "mode": "standard"},
        ),
    )
    _thread_local.s3 = client
    _thread_local.pool = max_pool_connections
    return client


def _clear_thread_client() -> None:
    _thread_local.s3 = None
    _thread_local.pool = None


def _backoff(attempt: int) -> None:
    time.sleep(min(8.0, 0.4 * (2 ** (attempt - 1))))


def parse_s3_uri(url: str) -> tuple[str, str]:
    """Return (bucket, key) from s3:// or HTTPS Cell Painting Gallery URLs."""
    url = url.strip()
    if url.startswith("s3://"):
        parsed = urlparse(url)
        return parsed.netloc, parsed.path.lstrip("/")
    if "cellpainting-gallery.s3" in url or "s3.amazonaws.com/cellpainting-gallery" in url:
        parsed = urlparse(url)
        path = parsed.path.lstrip("/")
        if path.startswith("cellpainting-gallery/"):
            path = path.split("/", 1)[1]
        return GALLERY_BUCKET, path
    raise ValueError(f"Not an S3 URI: {url!r}")


def format_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(n)} B"
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def read_s3_bytes(
    bucket: str,
    key: str,
    client=None,
    progress: Callable[[int], None] | None = None,
) -> bytes:
    """GET one object into memory. Retries mid-stream read timeouts.

    Progress is reported once after a successful full read so a retry cannot
    double-count bytes on the download bar.
    """
    given = client
    last_exc: BaseException | None = None
    for attempt in range(1, GET_ATTEMPTS + 1):
        use = given or unsigned_s3()
        body = None
        try:
            obj = use.get_object(Bucket=bucket, Key=key)
            body = obj["Body"]
            chunks: list[bytes] = []
            if hasattr(body, "iter_chunks"):
                iterator = body.iter_chunks(CHUNK_SIZE)
            else:
                iterator = [body.read()]
            for chunk in iterator:
                if not chunk:
                    continue
                chunks.append(chunk)
            payload = b"".join(chunks)
            if progress:
                progress(len(payload))
            return payload
        except STREAM_ERRORS as exc:
            last_exc = exc
            if given is None:
                _clear_thread_client()
            if attempt >= GET_ATTEMPTS:
                break
            _backoff(attempt)
        finally:
            close = getattr(body, "close", None) if body is not None else None
            if close:
                close()
    assert last_exc is not None
    raise last_exc


def write_bytes_atomic(dest: Path, payload: bytes) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = part_path(dest)
    with part.open("wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    part.replace(dest)
    return dest


def head_s3(bucket: str, key: str, client=None) -> dict:
    return (client or unsigned_s3()).head_object(Bucket=bucket, Key=key)


def part_path(dest: Path) -> Path:
    return Path(dest).with_name(Path(dest).name + PART_SUFFIX)


def _range_start(range_header: str) -> int:
    spec = range_header.removeprefix("bytes=")
    start = spec.split("-", 1)[0]
    return int(start or 0)


def download_s3_file(
    bucket: str,
    key: str,
    dest: Path,
    expected_size: int | None = None,
    max_concurrency: int = 1,
    client=None,
    progress: Callable[[int], None] | None = None,
) -> Path:
    """Download one object, writing through ``dest.part`` so crashes can resume.

    Completed files at ``dest`` are skipped. A leftover ``.part`` is resumed
    with a Range GET. ``max_concurrency`` is unused for these small TIFFs;
    kept so older callers still type-check.
    """
    del max_concurrency
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = part_path(dest)
    given = client

    if dest.exists():
        size = dest.stat().st_size
        complete = (expected_size is not None and size == expected_size) or (
            expected_size is None and size >= MIN_COMPLETE_BYTES
        )
        if complete:
            if part.exists():
                part.unlink()
            return dest
        dest.unlink()

    last_exc: BaseException | None = None
    for attempt in range(1, GET_ATTEMPTS + 1):
        use = given or unsigned_s3()
        start = part.stat().st_size if part.exists() else 0
        extra = {}
        if start:
            extra["Range"] = f"bytes={start}-"
        body = None
        try:
            try:
                obj = use.get_object(Bucket=bucket, Key=key, **extra)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if start and code in {"InvalidRange", "416"}:
                    part.unlink(missing_ok=True)
                    obj = use.get_object(Bucket=bucket, Key=key)
                    start = 0
                else:
                    raise
            mode = "ab" if start else "wb"
            body = obj["Body"]
            with part.open(mode) as fh:
                if hasattr(body, "iter_chunks"):
                    iterator = body.iter_chunks(CHUNK_SIZE)
                else:
                    iterator = [body.read()]
                for chunk in iterator:
                    if not chunk:
                        continue
                    fh.write(chunk)
                    if progress:
                        progress(len(chunk))
                fh.flush()
                os.fsync(fh.fileno())
            if expected_size is not None and part.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Size mismatch for s3://{bucket}/{key}: "
                    f"expected {expected_size}, got {part.stat().st_size}"
                )
            part.replace(dest)
            return dest
        except STREAM_ERRORS as exc:
            last_exc = exc
            if given is None:
                _clear_thread_client()
            if attempt >= GET_ATTEMPTS:
                break
            _backoff(attempt)
        finally:
            close = getattr(body, "close", None) if body is not None else None
            if close:
                close()
    assert last_exc is not None
    raise last_exc
