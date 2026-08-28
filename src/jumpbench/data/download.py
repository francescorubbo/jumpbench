"""Download JUMP TIFFs and the paper's assembled CellProfiler profiles.

JUMP-lite compressed Zarr stores may still be awaiting Cell Painting Gallery
promotion (mid-September 2026). Original JUMP TIFFs are already public; the
frozen site manifest in this repo points at those objects.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import polars as pl
from tqdm import tqdm

from jumpbench.config import load_data_config
from jumpbench.data.images import tiff_path
from jumpbench.data.metadata import load_sites
from jumpbench.data.s3util import download_s3_file
from jumpbench.paths import resolve


def _guess_tiff_url_columns(df: pl.DataFrame) -> list[str]:
    names = [c for c in df.columns if "url" in c.lower() or "uri" in c.lower() or "s3" in c.lower()]
    channel_named = [c for c in df.columns if any(ch.lower() in c.lower() for ch in ("agp", "dna", "er", "mito", "rna"))]
    return names or channel_named


def _s3_from_url(url: str) -> tuple[str, str] | None:
    url = url.strip()
    if url.startswith("s3://"):
        parsed = urlparse(url)
        return parsed.netloc, parsed.path.lstrip("/")
    if "cellpainting-gallery.s3" in url or "s3.amazonaws.com/cellpainting-gallery" in url:
        parsed = urlparse(url)
        path = parsed.path.lstrip("/")
        if path.startswith("cellpainting-gallery/"):
            path = path.split("/", 1)[1]
        return "cellpainting-gallery", path
    return None


def download_tiffs(
    dest: Path | None = None,
    max_sites: int | None = 32,
    sources: list[str] | None = None,
    site_keys: list[str] | None = None,
) -> list[Path]:
    """Download original JUMP TIFFs for a frozen JUMP-lite site subset.

    The site manifest schema varies across JUMP_lite snapshots. We look for
    either a site-key column plus per-channel URLs, or reconstruct keys from
    Metadata_* columns.
    """
    cfg = load_data_config()
    dest = Path(dest or resolve(cfg["layout"]["images"]))
    dest.mkdir(parents=True, exist_ok=True)
    sites = load_sites()

    key_col = next(
        (
            c
            for c in (
                "Metadata_Site_Key",
                "site_key",
                "site",
                "Metadata_SiteKey",
                "key",
            )
            if c in sites.columns
        ),
        None,
    )
    if key_col is None and {"Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"}.issubset(sites.columns):
        site_col = next((c for c in ("Metadata_Site", "site") if c in sites.columns), None)
        if site_col is None:
            raise ValueError(f"Cannot build site keys from columns {sites.columns}")
        sites = sites.with_columns(
            (
                pl.col("Metadata_Source").cast(pl.Utf8)
                + "__"
                + pl.col("Metadata_Batch").cast(pl.Utf8)
                + "__"
                + pl.col("Metadata_Plate").cast(pl.Utf8)
                + "__"
                + pl.col("Metadata_Well").cast(pl.Utf8)
                + "__"
                + pl.col(site_col).cast(pl.Utf8)
            ).alias("site_key")
        )
        key_col = "site_key"

    if key_col is None:
        raise ValueError(
            "Site manifest has no site key. Inspect metadata/jump_lite_v1_site_manifest.parquet."
        )

    if site_keys:
        sites = sites.filter(pl.col(key_col).is_in(site_keys))
    if sources:
        source_col = next((c for c in ("Metadata_Source", "source") if c in sites.columns), None)
        if source_col:
            sites = sites.filter(pl.col(source_col).is_in(sources))
    if max_sites is not None:
        sites = sites.head(max_sites)

    url_cols = _guess_tiff_url_columns(sites)
    downloaded: list[Path] = []
    if not url_cols:
        listing = dest / "requested_sites.csv"
        sites.select(key_col).write_csv(listing)
        print(
            "The frozen site manifest in this repo has plate/well/site keys but not "
            "original TIFF URLs.\n"
            f"Wrote {listing} ({sites.height} sites).\n"
            "To fetch pixels:\n"
            "  1. Original JUMP TIFFs: clone https://github.com/afermg/JUMP_lite and run\n"
            "     `just build-jl-index download-raw` (writes a URI manifest, then S3).\n"
            "  2. JUMP-lite JPEG XL zarr: Cell Painting Gallery path in configs/data.yaml\n"
            "     (public promotion expected mid-September 2026).\n"
            "  3. `jumpbench smoke` exercises the embedding CLI on synthetic images."
        )
        return downloaded

    rows = sites.select([key_col, *url_cols]).to_dicts()
    for row in tqdm(rows, desc="Downloading TIFFs"):
        site = str(row[key_col])
        for col in url_cols:
            url = row[col]
            if url is None:
                continue
            parsed = _s3_from_url(str(url))
            if parsed is None:
                continue
            bucket, key = parsed
            channel = Path(key).stem.split("_")[-1]
            # Prefer explicit channel names in the column.
            for ch in ("AGP", "DNA", "ER", "Mito", "RNA"):
                if ch.lower() in col.lower() or f"_{ch}" in Path(key).name:
                    channel = ch
                    break
            dest_path = tiff_path(dest, site, channel)
            download_s3_file(bucket, key, dest_path)
            downloaded.append(dest_path)
    return downloaded


def download_paper_cellprofiler(dest: Path | None = None) -> Path:
    """Fetch the assembled CellProfiler profiles used in the paper (~13.5 GB).

    These profiles are aggregated from 6–9 sites/well. Using them against
    4-site embeddings is the paper's comparison, not a fair one.
    """
    cfg = load_data_config()
    dest = Path(dest or resolve(cfg["layout"]["paper_cp"]))
    gallery = cfg["cellpainting_gallery"]
    return download_s3_file(
        gallery["bucket"],
        gallery["paper_cellprofiler"],
        dest,
        expected_size=int(gallery["paper_cellprofiler_bytes"]),
    )
