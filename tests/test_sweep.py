"""Process-config grid expansion and CRISPR-PA gather."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jumpbench.cli import main
from jumpbench.profiles.sweep import (
    config_id,
    expand_grid,
    gather_results,
    select_configs,
    winner_row,
)


def test_cp_grid_has_280_configs():
    configs = expand_grid(name="sweep_paper_cp_v11")
    assert len(configs) == 280
    first = configs[0]
    assert first["config_id"] == "robustmad_fitctrl-true_corr-0.9_eps-0.05_npc-96"
    assert first["overrides"]["normalize"] == "robustmad"
    assert first["overrides"]["tvn_n_components"] == 96
    last = configs[-1]
    assert last["config_id"] == "standardize_fitctrl-false_corr-0.95_eps-1_npc-304"


def test_config_id_is_stable():
    assert (
        config_id(
            {
                "normalize": "robustmad",
                "fit_on_controls": True,
                "corr_threshold": 0.9,
                "tvn_epsilon": 0.5,
                "tvn_n_components": 128,
            }
        )
        == "robustmad_fitctrl-true_corr-0.9_eps-0.5_npc-128"
    )


def test_select_index_out_of_range():
    configs = expand_grid(name="sweep_paper_cp_v11")
    with pytest.raises(IndexError, match="out of range"):
        select_configs(configs, index=280)
    one = select_configs(configs, index=17)
    assert len(one) == 1
    limited = select_configs(configs, limit=10, index=0)
    assert limited[0]["config_id"] == configs[0]["config_id"]


def test_dl_grid_kept():
    configs = expand_grid(name="sweep_paper_dl_v11_lite")
    assert len(configs) == 420


def test_gather_ranks_by_crispr_mean_nap(tmp_path: Path):
    for name, nap in (("low", 0.4), ("mid", 0.6), ("high", 0.8)):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "config_id": name,
                    "overrides": {"tvn_epsilon": 0.5},
                    "pa": {
                        "mean_nap": nap,
                        "n_perturbations": 10,
                        "paper_nap": 0.815,
                        "delta_vs_paper": nap - 0.815,
                    },
                }
            )
        )
    table = gather_results(tmp_path)
    assert table["config_id"].to_list() == ["high", "mid", "low"]
    win = winner_row(table)
    assert win is not None
    assert win["config_id"] == "high"
    assert win["mean_nap"] == 0.8


def test_sweep_list_cli(capsys):
    assert main(["sweep", "list", "--grid", "sweep_paper_cp_v11", "--limit", "3"]) == 0
    out = capsys.readouterr().out
    assert "robustmad_fitctrl-true_corr-0.9_eps-0.05_npc-96" in out
    assert out.count("\n") == 3


def test_sweep_run_dry_run(capsys, tmp_path: Path):
    rc = main(
        [
            "sweep",
            "run",
            "--grid",
            "sweep_paper_cp_v11",
            "--preset",
            "paper_cp_default",
            "--input",
            str(tmp_path / "in.parquet"),
            "--processed-dir",
            str(tmp_path / "processed"),
            "--results-dir",
            str(tmp_path / "results"),
            "--index",
            "0",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "jumpbench process" in out
    assert "--set normalize=" in out
    assert "jumpbench evaluate" in out
    assert "--subset crispr" in out
    assert "--tasks pa" in out


def test_sweep_index_cli_out_of_range(tmp_path: Path):
    with pytest.raises(SystemExit, match="out of range"):
        main(
            [
                "sweep",
                "run",
                "--grid",
                "sweep_paper_cp_v11",
                "--preset",
                "paper_cp_default",
                "--input",
                str(tmp_path / "in.parquet"),
                "--processed-dir",
                str(tmp_path / "processed"),
                "--results-dir",
                str(tmp_path / "results"),
                "--index",
                "999",
                "--dry-run",
            ]
        )
