"""Cartesian process-config grids and sweep gather.

Each grid is a mapping of preset keys to value lists in configs/process.yaml.
Jobs are independent: process one config, then evaluate CRISPR PA.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import polars as pl

from jumpbench.config import load_process_config
from jumpbench.eval.metrics import PAPER_PA_CRISPR, evaluate_path
from jumpbench.paths import resolve
from jumpbench.profiles.normalize import process_path

_RESERVED = {"presets", "evaluation"}

_SLUG_KEYS = {
    "normalize": None,
    "fit_on_controls": "fitctrl",
    "corr_threshold": "corr",
    "tvn_epsilon": "eps",
    "tvn_n_components": "npc",
    "pca_components": "pca",
    "prune_correlated": "prune",
}


def _fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if value is None:
        return "none"
    return str(value)


def config_id(overrides: dict[str, Any]) -> str:
    """Stable slug from swept keys, e.g. robustmad_fitctrl-true_corr-0.9_eps-0.5_npc-128."""
    parts: list[str] = []
    for key, value in overrides.items():
        short = _SLUG_KEYS.get(key, key.replace("_", "-"))
        token = _fmt_value(value)
        parts.append(token if short is None else f"{short}-{token}")
    return "_".join(parts) or "default"


def load_grid(name: str, process_cfg: dict[str, Any] | None = None) -> dict[str, list[Any]]:
    cfg = process_cfg or load_process_config()
    if name not in cfg or name in _RESERVED:
        known = [k for k in cfg if k not in _RESERVED and isinstance(cfg[k], dict)]
        raise KeyError(f"Unknown grid {name!r}. Known: {', '.join(known)}")
    raw = cfg[name]
    if not isinstance(raw, dict):
        raise TypeError(f"Grid {name!r} must be a mapping of axes to lists")
    grid = {k: v for k, v in raw.items() if isinstance(v, list)}
    if not grid:
        raise ValueError(f"Grid {name!r} has no list-valued axes")
    return grid


def expand_grid(
    grid: dict[str, list[Any]] | None = None,
    *,
    name: str | None = None,
    process_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Expand a Cartesian product into ``{config_id, overrides}`` rows."""
    if grid is None:
        if name is None:
            raise ValueError("Pass grid= or name=")
        grid = load_grid(name, process_cfg=process_cfg)
    keys = list(grid.keys())
    configs: list[dict[str, Any]] = []
    for combo in product(*(grid[k] for k in keys)):
        overrides = dict(zip(keys, combo, strict=True))
        configs.append({"config_id": config_id(overrides), "overrides": overrides})
    return configs


def select_configs(
    configs: list[dict[str, Any]],
    *,
    index: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    chosen = configs[:limit] if limit is not None else configs
    if index is None:
        return chosen
    if index < 0 or index >= len(chosen):
        raise IndexError(f"--index {index} out of range for {len(chosen)} configs")
    return [chosen[index]]


def result_path(results_dir: Path, config_id_str: str) -> Path:
    return Path(results_dir) / f"{config_id_str}.json"


def processed_path(processed_dir: Path, config_id_str: str) -> Path:
    return Path(processed_dir) / f"{config_id_str}.parquet"


def load_result_jsons(results_dir: str | Path) -> list[dict[str, Any]]:
    root = resolve(results_dir)
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("*.json")):
        if path.name in {"summary.json", "gather.json"}:
            continue
        data = json.loads(path.read_text())
        pa = data.get("pa") or {}
        rows.append(
            {
                "config_id": data.get("config_id", path.stem),
                "path": str(path),
                "mean_nap": pa.get("mean_nap"),
                "n_perturbations": pa.get("n_perturbations"),
                "paper_nap": pa.get("paper_nap", PAPER_PA_CRISPR),
                "delta_vs_paper": pa.get("delta_vs_paper"),
                "overrides": json.dumps(data.get("overrides") or {}, default=str),
            }
        )
    return rows


def gather_results(results_dir: str | Path) -> pl.DataFrame:
    """Rank completed CRISPR PA JSONs by mean_nap descending."""
    rows = load_result_jsons(results_dir)
    if not rows:
        return pl.DataFrame(
            schema={
                "rank": pl.Int64,
                "config_id": pl.Utf8,
                "mean_nap": pl.Float64,
                "n_perturbations": pl.Int64,
                "paper_nap": pl.Float64,
                "delta_vs_paper": pl.Float64,
                "overrides": pl.Utf8,
                "path": pl.Utf8,
            }
        )
    table = pl.DataFrame(rows).filter(pl.col("mean_nap").is_not_null())
    if table.height == 0:
        return table
    table = table.sort("mean_nap", descending=True).with_row_index("rank", offset=1)
    return table.select(
        "rank",
        "config_id",
        "mean_nap",
        "n_perturbations",
        "paper_nap",
        "delta_vs_paper",
        "overrides",
        "path",
    )


def winner_row(table: pl.DataFrame) -> dict[str, Any] | None:
    if table.height == 0:
        return None
    return table.row(0, named=True)


def override_set_flags(overrides: dict[str, Any]) -> list[str]:
    """CLI --set KEY=VALUE flags for a config's preset overrides."""
    flags: list[str] = []
    for key, value in overrides.items():
        dumped = json.dumps(value)
        if dumped in {"true", "false", "null"}:
            dumped = {"true": "true", "false": "false", "null": "null"}[dumped]
        flags.append(f"{key}={dumped}")
    return flags


def format_shard_commands(
    *,
    input_path: Path,
    processed: Path,
    result: Path,
    preset: str,
    overrides: dict[str, Any],
    device: str,
    tasks: str,
    subset: str,
) -> list[str]:
    sets = " ".join(f"--set {flag}" for flag in override_set_flags(overrides))
    process = (
        f"jumpbench process --input {input_path} --output {processed} "
        f"--preset {preset} --device {device}"
    )
    if sets:
        process = f"{process} {sets}"
    evaluate = (
        f"jumpbench evaluate --input {processed} --output {result} "
        f"--tasks {tasks} --subset {subset}"
    )
    return [process, evaluate]


def run_shard(
    *,
    input_path: str | Path,
    processed: str | Path,
    result: str | Path,
    preset: str,
    overrides: dict[str, Any],
    config_id_str: str,
    index: int,
    grid: str,
    device: str = "cpu",
    tasks: str = "pa",
    subset: str = "crispr",
    force: bool = False,
) -> Path:
    """Process one config and evaluate CRISPR PA. Skip if the result JSON exists."""
    result_file = Path(result)
    if result_file.exists() and not force:
        return result_file
    task_list = tuple(part.strip() for part in tasks.split(",") if part.strip())
    process_path(
        Path(input_path),
        Path(processed),
        preset=preset,
        preset_overrides=overrides,
        device=device,
    )
    eval_kwargs: dict[str, Any] = {}
    if subset == "crispr":
        eval_kwargs["subset"] = "crispr"
        eval_kwargs["group_col"] = None
        eval_kwargs["paper_ref"] = "crispr"
        if tasks == "pa,pc":
            task_list = ("pa",)
    scored = evaluate_path(Path(processed), tasks=task_list, **eval_kwargs)
    printable = {k: v for k, v in scored.items() if not k.startswith("_")}
    payload = {
        "config_id": config_id_str,
        "index": index,
        "grid": grid,
        "preset": preset,
        "overrides": overrides,
        "device": device,
        **printable,
    }
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return result_file


def _run_shard_payload(payload: dict[str, Any]) -> str:
    """Picklable worker for ProcessPoolExecutor."""
    path = run_shard(**payload)
    return str(path)
