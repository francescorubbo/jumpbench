#!/usr/bin/env python3
"""Ranked Raw-vs-MQ waterfall of sweep_paper_dl_v11_lite (sorted by Raw NAP)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from jumpbench.paths import repo_root, resolve

NORM_ORDER = ("standardize", "none", "robustmad")
FIT_ORDER = (False, True)

NORM_LABEL = {
    "standardize": "z-score",
    "none": "none",
    "robustmad": "RobustMAD",
}
FAMILY_LABEL = {
    ("standardize", False): "z-score · all wells",
    ("standardize", True): "z-score · negcons",
    ("none", False): "none",
    ("none", True): "none",
    ("robustmad", False): "RobustMAD · all wells",
    ("robustmad", True): "RobustMAD · negcons",
}

# Colorblind-safe: Raw/MQ series, then normalize strip, then fit strip.
RAW_COLOR = "#0072B2"
MQ_COLOR = "#D55E00"
SIMPLE_RAW = "#56B4E9"
SIMPLE_MQ = "#E69F00"
NORM_COLORS = {
    "standardize": "#0072B2",
    "none": "#999999",
    "robustmad": "#009E73",
}
FIT_COLORS = {False: "#F0E442", True: "#CC79A7"}


def _load_summary(path: Path, value_name: str) -> pl.DataFrame:
    rows = []
    for d in pl.read_csv(path).to_dicts():
        o = json.loads(d["overrides"])
        norm = o.get("normalize")
        if norm is None:
            norm = "none"
        rows.append(
            {
                "config_id": d["config_id"],
                value_name: float(d["mean_nap"]),
                "normalize": norm,
                "fit_on_controls": bool(o["fit_on_controls"]),
                "prune_correlated": bool(o["prune_correlated"]),
                "tvn_epsilon": float(o["tvn_epsilon"]),
                "pca_components": int(o["pca_components"]),
            }
        )
    return pl.DataFrame(rows)


def _simple_pca100(path: Path) -> float:
    return float(json.loads(Path(path).read_text())["pa"]["mean_nap"])


def _contiguous_runs(keys: list[tuple]) -> list[tuple[tuple, int, int]]:
    runs: list[tuple[tuple, int, int]] = []
    start = 0
    prev = keys[0]
    for i, key in enumerate(keys[1:], start=1):
        if key != prev:
            runs.append((prev, start, i - 1))
            prev = key
            start = i
    runs.append((prev, start, len(keys) - 1))
    return runs


def _family_key(normalize: str, fit: bool) -> tuple[str, bool]:
    if normalize == "none":
        return ("none", False)
    return (normalize, fit)


def join_sweeps(raw_csv: Path, mq_csv: Path) -> pl.DataFrame:
    raw = _load_summary(raw_csv, "raw_nap")
    mq = _load_summary(mq_csv, "mq_nap")
    return (
        raw.join(mq.select("config_id", "mq_nap"), on="config_id", how="inner")
        .sort("raw_nap", descending=True)
        .with_row_index("rank", offset=1)
    )


def plot_waterfall(
    table: pl.DataFrame,
    *,
    simple_raw: float,
    simple_mq: float,
    dest_png: Path,
    dest_svg: Path,
) -> None:
    n = table.height
    x = np.arange(n)
    raw = table["raw_nap"].to_numpy()
    mq = table["mq_nap"].to_numpy()
    norms = table["normalize"].to_list()
    fits = table["fit_on_controls"].to_list()
    families = [_family_key(n_, f) for n_, f in zip(norms, fits, strict=True)]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12.5, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [5.4, 0.28, 0.28], "hspace": 0.04},
        layout="constrained",
    )
    ax, ax_norm, ax_fit = axes

    ax.bar(x, raw, width=1.0, color=RAW_COLOR, align="edge", linewidth=0, label="Raw (sorted)")
    ax.plot(x + 0.5, mq, color=MQ_COLOR, lw=1.1, label="MQ (same config)")
    ax.axhline(simple_raw, color=SIMPLE_RAW, ls="--", lw=1.3, label="simple_pca100 Raw")
    ax.axhline(simple_mq, color=SIMPLE_MQ, ls="--", lw=1.3, label="simple_pca100 MQ")
    ax.set_ylabel("CRISPR PA mean NAP")
    ax.set_ylim(-0.05, 0.52)
    ax.set_xlim(0, n)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_title("DL process sweep, ranked by Raw NAP (Run1 XL cell-96, n=420)")

    long_runs = [(k, a, b) for k, a, b in _contiguous_runs(families) if (b - a + 1) >= 20]
    for _key, a, b in long_runs:
        ax.axvline(a, color="0.85", lw=0.6)
        ax.axvline(b + 1, color="0.85", lw=0.6)

    norm_idx = np.array([NORM_ORDER.index(v) for v in norms], dtype=float).reshape(1, -1)
    fit_idx = np.array([FIT_ORDER.index(v) for v in fits], dtype=float).reshape(1, -1)
    ax_norm.imshow(
        norm_idx,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap([NORM_COLORS[k] for k in NORM_ORDER]),
        vmin=0,
        vmax=2,
        extent=(0, n, 0, 1),
    )
    ax_norm.set_yticks([])
    ax_norm.set_ylabel("norm.", rotation=0, ha="right", va="center", fontsize=8)
    ax_fit.imshow(
        fit_idx,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap([FIT_COLORS[k] for k in FIT_ORDER]),
        vmin=0,
        vmax=1,
        extent=(0, n, 0, 1),
    )
    ax_fit.set_yticks([])
    ax_fit.set_ylabel("fit", rotation=0, ha="right", va="center", fontsize=8)

    ticks = []
    labels = []
    for key, a, b in long_runs:
        ticks.append((a + b + 1) / 2)
        labels.append(FAMILY_LABEL[key])
    ax_fit.set_xticks(ticks)
    ax_fit.set_xticklabels(labels, fontsize=8)
    ax_fit.set_xlabel("Configs sorted by Raw NAP (high → low). Prune / TVN ε / PCA rank jitter inside blocks.")

    strip_handles = [
        Patch(facecolor=NORM_COLORS["standardize"], label="z-score"),
        Patch(facecolor=NORM_COLORS["none"], label="none"),
        Patch(facecolor=NORM_COLORS["robustmad"], label="RobustMAD"),
        Patch(facecolor=FIT_COLORS[False], label="fit on all wells"),
        Patch(facecolor=FIT_COLORS[True], label="fit on negcons"),
    ]
    ax_norm.legend(
        handles=strip_handles,
        loc="center left",
        bbox_to_anchor=(1.01, -0.1),
        frameon=False,
        fontsize=7,
        title="Strips",
        title_fontsize=7,
    )

    dest_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest_png, dpi=160)
    fig.savefig(dest_svg)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    root = repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw",
        default="data/results/campaign/timm_run1_xl_c96_raw_dl_sweep/summary.csv",
    )
    p.add_argument(
        "--mq",
        default="data/results/campaign/timm_run1_xl_c96_mq_dl_sweep/summary.csv",
    )
    p.add_argument(
        "--simple-raw",
        default="data/results/campaign/run1_xl_c96_raw_simple_pca100.json",
    )
    p.add_argument(
        "--simple-mq",
        default="data/results/campaign/run1_xl_c96_mq_simple_pca100.json",
    )
    p.add_argument("--png", default="docs/figures/wave_r_sweep_waterfall.png")
    p.add_argument("--svg", default="docs/figures/wave_r_sweep_waterfall.svg")
    args = p.parse_args(argv)
    table = join_sweeps(resolve(args.raw, root), resolve(args.mq, root))
    if table.height != 420:
        raise SystemExit(f"expected 420 matched configs, got {table.height}")
    plot_waterfall(
        table,
        simple_raw=_simple_pca100(resolve(args.simple_raw, root)),
        simple_mq=_simple_pca100(resolve(args.simple_mq, root)),
        dest_png=resolve(args.png, root),
        dest_svg=resolve(args.svg, root),
    )
    print(resolve(args.png, root))
    print(resolve(args.svg, root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
