from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tifffile

from jumpbench.config import load_data_config
from jumpbench.data.compress import ImageCodec, get_codec
from jumpbench.data.metadata import parse_site_key
from jumpbench.data.metadata import site_key as make_site_key

CHANNEL_FILES = ("AGP", "DNA", "ER", "Mito", "RNA")
IMAGE_SUFFIXES = (".jxl", ".jpg", ".jpeg", ".tif", ".tiff")


def image_stem(site_key: str, channel: str) -> str:
    return f"{site_key}__{channel}"


def nested_relpath(site_key: str, channel: str, suffix: str) -> Path:
    """source/batch/plate/well/{site}__{channel}{suffix}."""
    parsed = parse_site_key(site_key)
    return (
        Path(parsed["source"])
        / parsed["batch"]
        / parsed["plate"]
        / parsed["well"]
        / f"{parsed['site']}__{channel}{suffix}"
    )


def flat_relpath(site_key: str, channel: str, suffix: str) -> Path:
    return Path(f"{image_stem(site_key, channel)}{suffix}")


def tiff_stem(site_key: str, channel: str) -> str:
    return f"{image_stem(site_key, channel)}.tif"


def image_path(
    images_root: Path,
    site_key: str,
    channel: str,
    codec: ImageCodec | str,
) -> Path:
    spec = get_codec(codec) if isinstance(codec, str) else codec
    return Path(images_root) / nested_relpath(site_key, channel, spec.suffix)


def tiff_path(images_root: Path, site_key: str, channel: str) -> Path:
    return Path(images_root) / nested_relpath(site_key, channel, ".tif")


def legacy_image_path(
    images_root: Path,
    site_key: str,
    channel: str,
    suffix: str,
) -> Path:
    return Path(images_root) / flat_relpath(site_key, channel, suffix)


def candidate_paths(
    images_root: Path, site_key: str, channel: str, suffix: str
) -> tuple[Path, Path]:
    root = Path(images_root)
    return (
        root / nested_relpath(site_key, channel, suffix),
        root / flat_relpath(site_key, channel, suffix),
    )


def tiff_candidates(images_root: Path, site_key: str, channel: str) -> tuple[Path, Path]:
    return candidate_paths(images_root, site_key, channel, ".tif")


def find_channel_path(images_root: Path, site_key: str, channel: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        for path in candidate_paths(images_root, site_key, channel, suffix):
            if path.exists():
                return path
    return None


def site_has_images(images_root: Path, site_key: str, channel: str = "DNA") -> bool:
    """True if at least one channel file exists. Does not walk the tree."""
    return find_channel_path(images_root, site_key, channel) is not None


def find_tiff(images_root: Path, site_key: str, channel: str) -> Path | None:
    for path in tiff_candidates(images_root, site_key, channel):
        if path.exists():
            return path
    return None


def read_image_file(path: Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return np.asarray(tifffile.imread(path))
    from jumpbench.data.compress import decode_image

    payload = path.read_bytes()
    if suffix == ".jxl":
        return decode_image(payload, "jpegxl_mq")
    return decode_image(payload, "jpeg")


def load_site_images(
    images_root: Path,
    site_key: str,
    channels: tuple[str, ...] = CHANNEL_FILES,
) -> np.ndarray:
    """Load a site as uint16 array with shape (C, H, W) in JUMP-lite channel order."""
    arrays = []
    for channel in channels:
        path = find_channel_path(images_root, site_key, channel)
        if path is None:
            raise FileNotFoundError(tiff_path(images_root, site_key, channel))
        arrays.append(read_image_file(path))
    stacked = np.stack(arrays, axis=0)
    if stacked.dtype != np.uint16:
        stacked = stacked.astype(np.uint16, copy=False)
    return stacked


def load_site_tiffs(
    images_root: Path,
    site_key: str,
    channels: tuple[str, ...] = CHANNEL_FILES,
) -> np.ndarray:
    """Backward-compatible alias; loads JPEG XL / JPEG / TIFF."""
    return load_site_images(images_root, site_key, channels)


def _site_from_filename(root: Path, path: Path) -> str | None:
    stem = path.stem
    if "__" not in stem:
        return None
    name, channel = stem.rsplit("__", 1)
    if channel not in CHANNEL_FILES:
        return None
    try:
        parse_site_key(name)
        return name
    except ValueError:
        pass
    try:
        rel = path.parent.relative_to(root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 4:
        return None
    try:
        return make_site_key(parts[0], parts[1], parts[2], parts[3], name)
    except ValueError:
        return None


def iter_local_sites(images_root: Path) -> Iterator[str]:
    """Yield site keys as files are discovered. Prefer this over a full-tree scan."""
    seen: set[str] = set()
    root = Path(images_root)
    if not root.exists():
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        for filename in filenames:
            suffix = Path(filename).suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                continue
            site = _site_from_filename(root, base / filename)
            if site and site not in seen:
                seen.add(site)
                yield site


def list_local_sites(images_root: Path) -> list[str]:
    return sorted(iter_local_sites(images_root))


def iter_flat_image_entries(images_root: Path) -> Iterator[os.DirEntry]:
    """Top-level files only (the slow flat layout). Does not recurse."""
    root = Path(images_root)
    if not root.is_dir():
        return
    with os.scandir(root) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            name = entry.name
            if name.endswith(".part"):
                name = name[: -len(".part")]
            suffix = Path(name).suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                yield entry


def migrate_flat_images(images_root: Path, *, show_progress: bool = True) -> int:
    """Move top-level ``{site}__{channel}.*`` files into source/batch/plate/well/.

    Returns the number of files moved. Completes already-nested files are ignored.
    """
    from tqdm import tqdm

    from jumpbench.data.s3util import part_path

    root = Path(images_root)
    entries = list(iter_flat_image_entries(root))
    if not entries:
        return 0
    moved = 0
    pbar = tqdm(entries, desc="nest images", unit="file", disable=not show_progress)
    for entry in pbar:
        name = entry.name
        is_part = name.endswith(".part")
        core = name[: -len(".part")] if is_part else name
        path = Path(entry.path)
        suffix = Path(core).suffix.lower()
        stem = Path(core).stem
        if "__" not in stem:
            continue
        site, channel = stem.rsplit("__", 1)
        try:
            parse_site_key(site)
        except ValueError:
            continue
        dest = root / nested_relpath(site, channel, suffix)
        if is_part:
            dest = part_path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            path.unlink()
            moved += 1
            continue
        path.replace(dest)
        moved += 1
    pbar.close()
    return moved


def default_images_root() -> Path:
    from jumpbench.paths import resolve

    cfg = load_data_config()
    return resolve(cfg["layout"]["images"])
