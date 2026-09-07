"""Stream original JUMP TIFFs from S3 and persist compressed files.

By default we never write TIFFs to disk. Each Orig object is fetched into
memory, encoded as JPEG XL (paper MQ, Butteraugli distance 3.0), and written
through ``*.jxl.part``. Leftover local TIFFs from earlier runs are transcoded
and deleted. Transfers skip completed objects and estimate S3 vs disk before
asking to proceed.
"""

from __future__ import annotations

import os
import shutil
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import tifffile
from tqdm import tqdm

from jumpbench.config import load_data_config
from jumpbench.data.compress import DEFAULT_CODEC, ImageCodec, encode_image, get_codec
from jumpbench.data.images import (
    candidate_paths,
    find_tiff,
    image_path,
    migrate_flat_images,
    tiff_candidates,
)
from jumpbench.data.index import build_tiff_index, site_set_summary, write_tiff_index
from jumpbench.data.s3util import (
    GALLERY_BUCKET,
    MIN_COMPLETE_BYTES,
    download_s3_file,
    format_bytes,
    head_s3,
    parse_s3_uri,
    part_path,
    read_s3_bytes,
    write_bytes_atomic,
)
from jumpbench.paths import resolve

FALLBACK_TIFF_BYTES = 2_768_896  # observed JUMP Orig TIFF on source_13
HEAD_SAMPLE = 24
ENCODE_SAMPLE = 4
Status = Literal["complete", "partial", "missing", "reclaim"]


@dataclass
class FileTask:
    site_key: str
    channel: str
    s3_key: str
    dest: Path
    status: Status
    local_bytes: int = 0
    local_tiff: Path | None = None


@dataclass
class DownloadPlan:
    dest: Path
    index_path: Path
    stats: dict[str, object]
    tasks: list[FileTask]
    sample_sizes: list[int] = field(default_factory=list)
    mean_bytes: float = float(FALLBACK_TIFF_BYTES)
    mean_disk_bytes: float = float(FALLBACK_TIFF_BYTES)
    complete_n: int = 0
    remaining_n: int = 0
    s3_n: int = 0
    reclaim_n: int = 0
    complete_bytes: int = 0
    estimated_remaining_bytes: int = 0
    estimated_s3_bytes: int = 0
    free_bytes: int = 0
    estimate_source: str = "fallback"
    codec: str = DEFAULT_CODEC


def classify_task(
    dest: Path,
    site_key: str,
    channel: str,
    s3_key: str,
    codec: str | ImageCodec = DEFAULT_CODEC,
) -> FileTask:
    spec = get_codec(codec) if isinstance(codec, str) else codec
    path = image_path(dest, site_key, channel, spec)
    leftover_tiff = find_tiff(dest, site_key, channel)
    existing = None
    for candidate in candidate_paths(dest, site_key, channel, spec.suffix):
        if candidate.exists() and candidate.stat().st_size >= spec.min_complete_bytes:
            existing = candidate
            break
    if existing is not None:
        return FileTask(site_key, channel, s3_key, path, "complete", existing.stat().st_size)
    if (
        spec.kind != "raw"
        and leftover_tiff is not None
        and leftover_tiff.stat().st_size >= MIN_COMPLETE_BYTES
    ):
        return FileTask(
            site_key,
            channel,
            s3_key,
            path,
            "reclaim",
            leftover_tiff.stat().st_size,
            local_tiff=leftover_tiff,
        )
    for candidate in candidate_paths(dest, site_key, channel, spec.suffix):
        part = part_path(candidate)
        if part.exists() and part.stat().st_size > 0:
            return FileTask(site_key, channel, s3_key, path, "partial", part.stat().st_size)
    return FileTask(site_key, channel, s3_key, path, "missing", 0)


def classify_index(
    index: pl.DataFrame,
    dest: Path,
    codec: str | ImageCodec = DEFAULT_CODEC,
) -> list[FileTask]:
    spec = get_codec(codec) if isinstance(codec, str) else codec
    rows = index.select(["Metadata_Site_Key", "channel", "s3_key"]).to_dicts()
    return [
        classify_task(dest, row["Metadata_Site_Key"], row["channel"], row["s3_key"], spec)
        for row in rows
    ]


def _sample_remaining(
    index: pl.DataFrame,
    remaining_keys: set[tuple[str, str]],
    n: int,
) -> pl.DataFrame:
    work = index.with_columns(
        (pl.col("Metadata_Site_Key") + "\0" + pl.col("channel")).alias("_k")
    )
    want = {f"{site}\0{ch}" for site, ch in remaining_keys}
    work = work.filter(pl.col("_k").is_in(list(want)))
    if work.height == 0:
        return work
    if "Metadata_Source" in work.columns:
        parts = []
        sources = work["Metadata_Source"].unique().to_list()
        per = max(2, n // max(len(sources), 1))
        for src in sources:
            sub = work.filter(pl.col("Metadata_Source") == src)
            take = min(per, sub.height)
            parts.append(sub.sample(n=take, seed=0) if take < sub.height else sub)
        sampled = pl.concat(parts)
        if sampled.height > n:
            sampled = sampled.sample(n=n, seed=0)
        return sampled
    if work.height <= n:
        return work
    return work.sample(n=n, seed=0)


def sample_object_sizes(
    keys: list[str],
    jobs: int = 16,
    client=None,
    show_progress: bool = True,
) -> list[int]:
    if not keys:
        return []
    jobs = max(1, min(jobs, len(keys)))
    sizes: list[int] = []
    pbar = tqdm(
        total=len(keys),
        desc="HEAD sample",
        unit="obj",
        disable=not show_progress,
    )

    def _one(key: str) -> int:
        return int(head_s3(GALLERY_BUCKET, key, client=client)["ContentLength"])

    try:
        if jobs == 1:
            for key in keys:
                sizes.append(_one(key))
                pbar.update(1)
            return sizes
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_one, key) for key in keys]
            for fut in as_completed(futures):
                sizes.append(fut.result())
                pbar.update(1)
        return sizes
    finally:
        pbar.close()


def estimate_remaining_bytes(remaining_n: int, sample_sizes: list[int]) -> tuple[int, float, str]:
    if remaining_n <= 0:
        return 0, 0.0, "none"
    if sample_sizes:
        mean = float(sum(sample_sizes) / len(sample_sizes))
        return int(mean * remaining_n), mean, f"{len(sample_sizes)} HEADs"
    return int(FALLBACK_TIFF_BYTES * remaining_n), float(FALLBACK_TIFF_BYTES), "fallback 2.6 MiB"


def _sample_encoded_sizes(
    tasks: list[FileTask], spec: ImageCodec, n: int = ENCODE_SAMPLE
) -> list[int]:
    sizes: list[int] = []
    for task in tasks:
        if len(sizes) >= n:
            break
        if task.local_tiff is None or not task.local_tiff.exists():
            continue
        try:
            array = np.asarray(tifffile.imread(task.local_tiff))
            sizes.append(len(encode_image(array, spec)))
        except Exception:  # noqa: BLE001 — estimate can fall back
            continue
    return sizes


def _disk_estimate(
    remaining_n: int,
    mean_s3: float,
    spec: ImageCodec,
    encoded_sizes: list[int],
    s3_source: str,
) -> tuple[int, float, str]:
    if remaining_n <= 0:
        return 0, 0.0, "none"
    if spec.kind == "raw":
        return int(mean_s3 * remaining_n), mean_s3, s3_source
    if encoded_sizes:
        mean = float(sum(encoded_sizes) / len(encoded_sizes))
        return int(mean * remaining_n), mean, f"{len(encoded_sizes)} local encodes"
    mean = mean_s3 * spec.paper_ratio
    return (
        int(mean * remaining_n),
        mean,
        f"paper {spec.name} {spec.paper_ratio:.1%} × TIFF ({s3_source})",
    )


def plan_tiff_download(
    index: pl.DataFrame,
    dest: Path,
    index_path: Path,
    *,
    codec: str = DEFAULT_CODEC,
    sample_n: int = HEAD_SAMPLE,
    jobs: int = 16,
    head_objects: bool = True,
    show_progress: bool = True,
) -> DownloadPlan:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    spec = get_codec(codec)
    tasks = classify_index(index, dest, spec)
    complete = [t for t in tasks if t.status == "complete"]
    remaining = [t for t in tasks if t.status != "complete"]
    s3_tasks = [t for t in remaining if t.status != "reclaim"]
    reclaim = [t for t in remaining if t.status == "reclaim"]
    complete_bytes = sum(t.local_bytes for t in complete)
    sample_sizes: list[int] = []
    if head_objects and s3_tasks:
        remaining_keys = {(t.site_key, t.channel) for t in s3_tasks}
        sampled = _sample_remaining(index, remaining_keys, sample_n)
        keys = (
            sampled["s3_key"].to_list()
            if sampled.height
            else [t.s3_key for t in s3_tasks[:sample_n]]
        )
        try:
            sample_sizes = sample_object_sizes(keys, jobs=jobs, show_progress=show_progress)
        except Exception:  # noqa: BLE001 — estimate can fall back
            sample_sizes = []
    s3_bytes, mean_s3, s3_source = estimate_remaining_bytes(len(s3_tasks), sample_sizes)
    if spec.kind == "raw":
        s3_bytes = max(0, s3_bytes - sum(t.local_bytes for t in remaining))
    encoded_sizes = _sample_encoded_sizes(reclaim, spec) if spec.kind != "raw" else []
    disk_bytes, mean_disk, disk_source = _disk_estimate(
        len(remaining), mean_s3, spec, encoded_sizes, s3_source
    )
    try:
        free = shutil.disk_usage(dest).free
    except OSError:
        free = 0
    return DownloadPlan(
        dest=dest,
        index_path=index_path,
        stats=site_set_summary(index),
        tasks=tasks,
        sample_sizes=sample_sizes,
        mean_bytes=mean_s3,
        mean_disk_bytes=mean_disk,
        complete_n=len(complete),
        remaining_n=len(remaining),
        s3_n=len(s3_tasks),
        reclaim_n=len(reclaim),
        complete_bytes=complete_bytes,
        estimated_remaining_bytes=disk_bytes,
        estimated_s3_bytes=s3_bytes,
        free_bytes=free,
        estimate_source=disk_source,
        codec=spec.name,
    )


def format_download_plan(plan: DownloadPlan) -> str:
    stats = plan.stats
    spec = get_codec(plan.codec)
    noun = "TIFFs" if spec.kind == "raw" else spec.suffix.lstrip(".").upper()
    lines = [
        f"Codec: {spec.name} — {spec.description}",
        f"Site set: {stats.get('site_set')}  "
        f"{stats.get('sites')} sites / {stats.get('wells')} wells  "
        f"(FOVs per well "
        f"{stats.get('sites_per_well_min')}/"
        f"{stats.get('sites_per_well_median'):.0f}/"
        f"{stats.get('sites_per_well_max')})",
        f"Files: {len(plan.tasks)} {noun} → {plan.dest}",
        f"Already complete: {plan.complete_n} ({format_bytes(plan.complete_bytes)})",
    ]
    if spec.kind == "raw":
        lines.append(
            f"Remaining: {plan.remaining_n} "
            f"(~{format_bytes(plan.estimated_remaining_bytes)}, {plan.estimate_source})"
        )
    else:
        reclaim = f", reclaim {plan.reclaim_n} local TIFF" if plan.reclaim_n else ""
        lines.append(
            f"S3 remaining: {plan.s3_n} TIFFs "
            f"(~{format_bytes(plan.estimated_s3_bytes)} transfer{reclaim})"
        )
        lines.append(
            f"Disk remaining: {plan.remaining_n} "
            f"(~{format_bytes(plan.estimated_remaining_bytes)}, {plan.estimate_source})"
        )
    lines.append(f"Free disk: {format_bytes(plan.free_bytes)}")
    if plan.remaining_n and plan.estimated_remaining_bytes > plan.free_bytes * 0.9:
        lines.append("WARNING: estimated remaining size is close to or above free disk.")
    if spec.kind == "raw" and plan.estimated_remaining_bytes > 50 * 1024**3:
        lines.append(
            "WARNING: --codec raw stores uncompressed TIFFs. "
            "Use the default jpegxl_mq unless you have a multi-TB disk."
        )
    return "\n".join(lines)


def confirm_download(plan: DownloadPlan, *, yes: bool, interactive: bool | None = None) -> bool:
    if plan.remaining_n == 0 or yes:
        return True
    if interactive is None:
        interactive = os.isatty(0)
    if not interactive:
        raise SystemExit(
            "Refusing to download without confirmation (not a TTY). Re-run with --yes."
        )
    try:
        reply = input("Proceed with download? [y/N] ")
    except EOFError:
        return False
    return reply.strip().lower() in {"y", "yes"}


def _decode_s3_tiff(payload: bytes) -> np.ndarray:
    return np.asarray(tifffile.imread(BytesIO(payload)))


def _persist_encoded(array: np.ndarray, dest: Path, spec: ImageCodec) -> None:
    payload = encode_image(array, spec)
    write_bytes_atomic(dest, payload)


def _run_downloads(
    remaining: list[FileTask],
    *,
    jobs: int,
    estimated_bytes: int,
    spec: ImageCodec,
) -> dict[str, int]:
    lock = threading.Lock()
    use_bytes = estimated_bytes > 0
    pbar = tqdm(
        total=estimated_bytes if use_bytes else len(remaining),
        unit="B" if use_bytes else "img",
        unit_scale=use_bytes,
        unit_divisor=1024 if use_bytes else 1000,
        desc=spec.name,
        dynamic_ncols=True,
    )
    pbar.set_postfix(files=f"0/{len(remaining)}", refresh=False)

    def _progress(n: int) -> None:
        if use_bytes:
            with lock:
                pbar.update(n)

    def _one(task: FileTask) -> None:
        if spec.kind == "raw":
            download_s3_file(
                GALLERY_BUCKET,
                task.s3_key,
                task.dest,
                progress=_progress,
            )
            return
        leftover = part_path(task.dest)
        if leftover.exists():
            leftover.unlink()
        if task.local_tiff is not None and task.local_tiff.exists():
            array = np.asarray(tifffile.imread(task.local_tiff))
            _persist_encoded(array, task.dest, spec)
            task.local_tiff.unlink(missing_ok=True)
            return
        payload = read_s3_bytes(GALLERY_BUCKET, task.s3_key, progress=_progress)
        _persist_encoded(_decode_s3_tiff(payload), task.dest, spec)

    errors: list[BaseException] = []
    done_n = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            inflight = set()
            for task in remaining:
                inflight.add(pool.submit(_one, task))
                while len(inflight) >= max(jobs * 4, jobs):
                    finished, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        try:
                            fut.result()
                            done_n += 1
                        except Exception as exc:  # noqa: BLE001 — per-file, resume later
                            errors.append(exc)
                        if not use_bytes:
                            pbar.update(1)
                        pbar.set_postfix(
                            files=f"{done_n + len(errors)}/{len(remaining)}",
                            refresh=False,
                        )
            while inflight:
                finished, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in finished:
                    try:
                        fut.result()
                        done_n += 1
                    except Exception as exc:  # noqa: BLE001 — per-file, resume later
                        errors.append(exc)
                    if not use_bytes:
                        pbar.update(1)
                    pbar.set_postfix(
                        files=f"{done_n + len(errors)}/{len(remaining)}",
                        refresh=False,
                    )
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Finished files are kept; leftover *.part files are "
            "rewritten on the next run.",
            flush=True,
        )
        raise
    finally:
        pbar.close()
    if errors:
        print(
            f"{len(errors)} files failed after retries (first: {errors[0]}). "
            "Finished files are kept; re-run download-images to fill gaps.",
            flush=True,
        )
        raise SystemExit(1)
    return {"ok": done_n, "failed": len(errors)}


def download_tiffs(
    dest: Path | None = None,
    max_sites: int | None = None,
    max_wells: int | None = None,
    site_set: str = "all",
    sources: list[str] | None = None,
    site_keys: list[str] | None = None,
    plates: list[str] | None = None,
    jobs: int = 32,
    dry_run: bool = False,
    yes: bool = False,
    index_out: Path | None = None,
    confirm: bool = True,
    codec: str = DEFAULT_CODEC,
) -> dict[str, object]:
    """Stream Orig JUMP TIFFs for JUMP-lite wells; persist ``codec`` files."""
    cfg = load_data_config()
    dest = Path(dest or resolve(cfg["layout"]["images"]))
    dest.mkdir(parents=True, exist_ok=True)
    spec = get_codec(codec)
    if not dry_run:
        nested = migrate_flat_images(dest)
        if nested:
            print(f"Nested {nested} flat files into source/batch/plate/well/.")

    index = build_tiff_index(
        site_set=site_set,
        max_sites=max_sites,
        max_wells=max_wells,
        sources=sources,
        site_keys=site_keys,
        plates=plates,
    )
    index_path = write_tiff_index(index, index_out)
    plan = plan_tiff_download(index, dest, index_path, jobs=jobs, codec=spec.name)
    summary: dict[str, object] = {
        **plan.stats,
        "index": str(plan.index_path),
        "dest": str(plan.dest),
        "codec": spec.name,
        "complete": plan.complete_n,
        "remaining": plan.remaining_n,
        "s3_remaining": plan.s3_n,
        "estimated_remaining_bytes": plan.estimated_remaining_bytes,
        "estimated_remaining": format_bytes(plan.estimated_remaining_bytes),
        "estimated_s3_bytes": plan.estimated_s3_bytes,
        "estimated_s3": format_bytes(plan.estimated_s3_bytes),
        "complete_bytes": plan.complete_bytes,
        "free_bytes": plan.free_bytes,
        "estimate_source": plan.estimate_source,
        "dry_run": dry_run,
        "downloaded": [],
    }
    print(format_download_plan(plan))
    if dry_run:
        return summary
    if plan.remaining_n == 0:
        print("Nothing to download.")
        return summary
    if confirm and not confirm_download(plan, yes=yes):
        raise SystemExit("Download cancelled.")
    remaining = [t for t in plan.tasks if t.status != "complete"]
    if spec.kind == "raw":
        progress_total = plan.estimated_remaining_bytes
    else:
        progress_total = plan.estimated_s3_bytes
    _run_downloads(remaining, jobs=jobs, estimated_bytes=progress_total, spec=spec)
    if spec.kind != "raw":
        for task in plan.tasks:
            for leftover in tiff_candidates(dest, task.site_key, task.channel):
                if leftover.exists() and leftover != task.dest:
                    leftover.unlink()
    summary["downloaded"] = plan.remaining_n
    return summary


def download_paper_cellprofiler(dest: Path | None = None, dry_run: bool = False) -> Path | dict:
    """Fetch the assembled CellProfiler profiles used in the paper (~13.5 GB)."""
    cfg = load_data_config()
    dest = Path(dest or resolve(cfg["layout"]["paper_cp"]))
    gallery = cfg["cellpainting_gallery"]
    bucket = gallery["bucket"]
    key = gallery["paper_cellprofiler"]
    expected = int(gallery["paper_cellprofiler_bytes"])
    if dry_run:
        info = head_s3(bucket, key)
        return {
            "s3": f"s3://{bucket}/{key}",
            "bytes": info["ContentLength"],
            "expected": expected,
            "dest": str(dest),
        }
    return download_s3_file(bucket, key, dest, expected_size=expected)


def _s3_from_url(url: str) -> tuple[str, str] | None:
    try:
        return parse_s3_uri(url)
    except ValueError:
        return None
