"""Which sites to embed: CRISPR vs all wells, 4-site vs all FOVs, batch/plate filters."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import polars as pl

from jumpbench.data.images import iter_local_sites, site_has_images
from jumpbench.data.masks import mask_sites
from jumpbench.data.metadata import (
    JOIN_WELL,
    filter_wells,
    load_perturbations,
    load_wells,
    parse_site_key,
    well_id,
)
from jumpbench.profiles.cellprofiler import filter_crispr_wells

SITE_SETS = ("all", "jump_lite")
SUBSETS = ("all", "crispr")
CELL_CROPS = ("cell_fixed", "cell_bbox")


def _status(message: str) -> None:
    import sys

    print(message, file=sys.stderr, flush=True)


def _as_list(values: Iterable[str] | None) -> list[str] | None:
    if values is None:
        return None
    out = [str(v) for v in values]
    return out or None


def selected_wells(
    *,
    subset: str | None = None,
    max_wells: int | None = None,
    sources: Iterable[str] | None = None,
    plates: Iterable[str] | None = None,
    batches: Iterable[str] | None = None,
    site_keys: Iterable[str] | None = None,
) -> pl.DataFrame | None:
    """Wells to keep, or None when grid should walk every local site."""
    sources_l = _as_list(sources)
    plates_l = _as_list(plates)
    batches_l = _as_list(batches)
    keys_l = _as_list(site_keys)
    filtered = bool(
        subset == "crispr" or max_wells is not None or sources_l or plates_l or batches_l or keys_l
    )
    if not filtered:
        return None
    if subset not in {None, "all", "crispr"}:
        raise ValueError(f"subset must be 'all' or 'crispr', got {subset!r}")
    if subset == "crispr":
        wells = filter_crispr_wells(load_perturbations())
    else:
        wells = load_wells()
    return filter_wells(
        wells,
        max_wells=max_wells,
        sources=sources_l,
        plates=plates_l,
        batches=batches_l,
        site_keys=keys_l,
    )


def iter_local_sites_for_wells(images_root: Path, wells: pl.DataFrame) -> Iterator[str]:
    """Yield local site keys that belong to ``wells`` without walking the whole tree."""
    images_root = Path(images_root)
    seen: set[str] = set()
    for row in wells.select(JOIN_WELL).unique(maintain_order=True).iter_rows(named=True):
        well_dir = (
            images_root
            / row["Metadata_Source"]
            / row["Metadata_Batch"]
            / row["Metadata_Plate"]
            / row["Metadata_Well"]
        )
        if not well_dir.is_dir():
            continue
        for path in well_dir.iterdir():
            if not path.is_file():
                continue
            stem = path.stem
            if "__" not in stem:
                continue
            site_token, _channel = stem.rsplit("__", 1)
            key = (
                well_id(
                    row["Metadata_Source"],
                    row["Metadata_Batch"],
                    row["Metadata_Plate"],
                    row["Metadata_Well"],
                )
                + f"__{site_token}"
            )
            try:
                parse_site_key(key)
            except ValueError:
                continue
            if key not in seen:
                seen.add(key)
                yield key


def iter_embed_sites(
    images_root: Path,
    crop: str,
    *,
    site_keys: Iterable[str] | None = None,
    subset: str | None = None,
    site_set: str = "all",
    max_wells: int | None = None,
    sources: Iterable[str] | None = None,
    plates: Iterable[str] | None = None,
    batches: Iterable[str] | None = None,
) -> Iterator[str]:
    """Yield sites that should be embedded.

    Cell crops and ``site_set=jump_lite`` use frozen 4-site keys. Grid
    ``site_set=all`` uses local Orig FOVs, optionally restricted to CRISPR /
    batch / plate.
    """
    images_root = Path(images_root)
    if site_set not in SITE_SETS:
        raise ValueError(f"site_set must be one of {SITE_SETS}, got {site_set!r}")
    if crop in CELL_CROPS and site_set == "all":
        raise ValueError("cell crops are JUMP-lite 4-site only; use --sites jump_lite")

    explicit_keys = _as_list(site_keys)
    if crop == "grid" and explicit_keys is not None and subset is None and site_set == "all":
        if max_wells is None and not sources and not plates and not batches:
            yield from explicit_keys
            return

    use_frozen = crop in CELL_CROPS or site_set == "jump_lite"
    if use_frozen:
        frozen_subset = subset or "crispr"
        _status(f"Loading {frozen_subset} 4-site keys (site_set={site_set})...")
        masked = mask_sites(
            subset=frozen_subset,
            max_wells=max_wells,
            sources=_as_list(sources),
            plates=_as_list(plates),
            batches=_as_list(batches),
            site_keys=explicit_keys,
        )
        candidates = masked["Metadata_Site_Key"].to_list()
        _status(f"{len(candidates)} mask sites; probing which have local images...")
        n_checked = 0
        n_hit = 0
        for key in candidates:
            n_checked += 1
            if n_checked == 1 or n_checked % 500 == 0:
                _status(f"  probed {n_checked}/{len(candidates)} ({n_hit} local so far)")
            if site_has_images(images_root, key):
                n_hit += 1
                if n_hit == 1:
                    _status(f"  first local site: {key}")
                yield key
        _status(f"{n_hit} of {len(candidates)} mask sites have local images")
        return

    wells = selected_wells(
        subset=subset,
        max_wells=max_wells,
        sources=sources,
        plates=plates,
        batches=batches,
        site_keys=explicit_keys,
    )
    if wells is None:
        _status(f"Scanning local sites under {images_root} (streaming, no pre-count)...")
        n = 0
        for key in iter_local_sites(images_root):
            n += 1
            if n == 1 or n % 1000 == 0:
                _status(f"  found {n} local sites...")
            yield key
        _status(f"Found {n} local sites")
        return

    _status(f"Scanning local FOVs for {wells.height} wells (subset={subset or 'all'})...")
    n = 0
    for key in iter_local_sites_for_wells(images_root, wells):
        n += 1
        if n == 1 or n % 1000 == 0:
            _status(f"  found {n} local sites...")
        yield key
    _status(f"Found {n} local sites")
