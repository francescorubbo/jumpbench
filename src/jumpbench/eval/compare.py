"""Compare representations under explicit fairness modes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from jumpbench.eval.metrics import evaluate_profiles
from jumpbench.profiles.cellprofiler import comparison_is_fair
from jumpbench.provenance import now_iso, write_json

PAPER_HEADLINE = {
    "morphem": 0.944,
    "cellprofiler_paper": 0.936,
    "dinov2": 0.721,
    "subcell": 0.715,
    "openphenom": 0.688,
    "cell_count": 0.402,
    "dinov2_random": 0.252,
}

UNFAIR_REASONS = {
    "paper_as_published": (
        "CellProfiler features come from Cell Painting Gallery assembled "
        "profiles (6–9 sites/well). Embeddings use the JUMP-lite 4-site cohort "
        "(S1.2.7). MorphEM and OpenPhenom were pretrained on JUMP."
    ),
    "wells_aligned_only": (
        "Well universe is JUMP-lite, but CellProfiler values are still 6–9-site "
        "aggregates. Embeddings remain 4-site."
    ),
}


def compare_runs(
    profiles: dict[str, pl.DataFrame],
    mode: str,
    output: Path | None = None,
) -> pl.DataFrame:
    """Score each named profile table and attach fairness metadata.

    `profiles` maps a model name to a well-level processed dataframe.
    """
    rows = []
    for name, df in profiles.items():
        metrics = evaluate_profiles(df, tasks=("pa", "pc"))
        rows.append(
            {
                "model": name,
                "comparison_mode": mode,
                "fair": comparison_is_fair(mode),
                "pa_mean_nap": metrics.get("pa", {}).get("mean_nap"),
                "pc_mean_nap": metrics.get("pc", {}).get("mean_nap") if isinstance(metrics.get("pc"), dict) else None,
                "balanced_pa_pc": metrics.get("balanced_pa_pc"),
                "n_wells": df.height,
                "n_features": sum(not c.startswith("Metadata_") for c in df.columns),
                "paper_headline_mean": PAPER_HEADLINE.get(name),
                "caveat": UNFAIR_REASONS.get(mode, ""),
            }
        )
    table = pl.DataFrame(rows)
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        table.write_csv(output)
        write_json(
            output.with_suffix(".json"),
            {
                "created_at": now_iso(),
                "mode": mode,
                "fair": comparison_is_fair(mode),
                "rows": table.to_dicts(),
            },
        )
    return table
