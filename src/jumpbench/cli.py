from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from jumpbench.config import apply_overrides, load_models_config, resolve_model
from jumpbench.data.download import download_paper_cellprofiler, download_tiffs
from jumpbench.data.images import default_images_root
from jumpbench.embed.generate import generate_embeddings
from jumpbench.eval.compare import compare_runs
from jumpbench.eval.metrics import evaluate_path
from jumpbench.paths import repo_root, resolve
from jumpbench.profiles.aggregate import aggregate_path
from jumpbench.profiles.cellprofiler import load_paper_cellprofiler
from jumpbench.profiles.normalize import process_path


def _add_overrides(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Dotted YAML override, e.g. channel_recipe=paper_table_s3 or models.dinov2.tile_size=256",
    )


def cmd_models(args: argparse.Namespace) -> int:
    cfg = apply_overrides(load_models_config(), args.overrides)
    if args.name:
        card = resolve_model(args.name, cfg)
        print(json.dumps(card, indent=2, default=str))
        return 0
    print(f"channel_recipe: {cfg.get('channel_recipe')}")
    for name, card in cfg["models"].items():
        print(f"  {name:20s}  family={card.get('family')}  tile={card.get('tile_size', '-')}")
    return 0


def cmd_download_images(args: argparse.Namespace) -> int:
    paths = download_tiffs(dest=args.dest, max_sites=args.max_sites, site_keys=args.site)
    print(f"Downloaded {len(paths)} files to {args.dest or default_images_root()}")
    return 0


def cmd_download_paper_cp(args: argparse.Namespace) -> int:
    path = download_paper_cellprofiler(args.dest)
    print(f"Paper CellProfiler profiles: {path}")
    print("These are 6–9 sites/well. Do not treat as a fair comparator to 4-site embeddings.")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    cfg = apply_overrides(load_models_config(), args.overrides)
    out = generate_embeddings(
        model=args.model,
        images_root=Path(args.images),
        output_dir=Path(args.output),
        site_keys=args.site or None,
        models_cfg=cfg,
        codec=args.codec,
    )
    print(out)
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    path = aggregate_path(Path(args.input), Path(args.output), how=args.how)
    print(path)
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    path = process_path(Path(args.input), Path(args.output), preset=args.preset)
    print(path)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    result = evaluate_path(Path(args.input), tasks=tuple(args.tasks.split(",")))
    printable = {k: v for k, v in result.items() if not k.startswith("_")}
    print(json.dumps(printable, indent=2, default=str))
    if args.output:
        Path(args.output).write_text(json.dumps(printable, indent=2, default=str) + "\n")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    profiles = {}
    for spec in args.profile:
        if "=" not in spec:
            raise SystemExit("--profile must be NAME=PATH")
        name, path = spec.split("=", 1)
        profiles[name] = pl.read_parquet(path)
    table = compare_runs(profiles, mode=args.mode, output=Path(args.output) if args.output else None)
    print(table)
    if not table["fair"][0]:
        print("\nThis comparison is tagged UNFAIR. See README.md § Fairness.", file=sys.stderr)
    return 0


def cmd_paper_cp_align(args: argparse.Namespace) -> int:
    df = load_paper_cellprofiler(args.input, align_wells=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output)
    print(f"{df.height} wells written to {args.output} (still 6–9-site CellProfiler values)")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """End-to-end dummy embeddings on synthetic images; no network, no weights."""
    import numpy as np
    import tifffile

    root = Path(args.workdir)
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sites = [
        "source_2__batchA__plate1__A01__1",
        "source_2__batchA__plate1__A01__2",
        "source_2__batchA__plate1__A02__1",
        "source_2__batchA__plate1__A02__2",
    ]
    for site in sites:
        for ch in ("AGP", "DNA", "ER", "Mito", "RNA"):
            arr = rng.integers(0, 4096, size=(256, 256), dtype=np.uint16)
            tifffile.imwrite(images / f"{site}__{ch}.tif", arr)

    cfg = apply_overrides(load_models_config(), ["models.dummy.tile_size=64"])
    site_path = generate_embeddings("dummy", images, root / "embeddings", models_cfg=cfg, codec="synth")
    well_path = aggregate_path(site_path, root / "profiles" / "dummy.parquet")
    print(f"smoke site embeddings: {site_path}")
    print(f"smoke well profiles:   {well_path}  rows={pl.read_parquet(well_path).height}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jumpbench",
        description="JUMP-lite reproduction: controllable embeddings vs CellProfiler.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("models", help="List or dump a resolved model card")
    m.add_argument("name", nargs="?")
    _add_overrides(m)
    m.set_defaults(func=cmd_models)

    d = sub.add_parser("download-images", help="Download original JUMP TIFFs for a site subset")
    d.add_argument("--dest", type=Path)
    d.add_argument("--max-sites", type=int, default=32)
    d.add_argument("--site", action="append", default=[])
    d.set_defaults(func=cmd_download_images)

    c = sub.add_parser("download-paper-cp", help="Download assembled CellProfiler profiles (unfair baseline)")
    c.add_argument("--dest", type=Path)
    c.set_defaults(func=cmd_download_paper_cp)

    e = sub.add_parser("embed", help="Generate per-site embeddings with an explicit model card")
    e.add_argument("--model", required=True)
    e.add_argument("--images", default=str(default_images_root()))
    e.add_argument("--output", default=str(resolve("data/embeddings")))
    e.add_argument("--codec", default="raw_tiff")
    e.add_argument("--site", action="append", default=[])
    _add_overrides(e)
    e.set_defaults(func=cmd_embed)

    a = sub.add_parser("aggregate", help="Median-aggregate tiles/sites to wells")
    a.add_argument("--input", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--how", choices=("median", "mean"), default="median")
    a.set_defaults(func=cmd_aggregate)

    pr = sub.add_parser("process", help="RobustMAD / PCA / TVN-EFAAR profile processing")
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", required=True)
    pr.add_argument("--preset", default="paper_dl_default")
    pr.set_defaults(func=cmd_process)

    ev = sub.add_parser("evaluate", help="Phenotypic activity / consistency")
    ev.add_argument("--input", required=True)
    ev.add_argument("--tasks", default="pa,pc")
    ev.add_argument("--output")
    ev.set_defaults(func=cmd_evaluate)

    cmp_ = sub.add_parser("compare", help="Score multiple representations; tags unfair modes")
    cmp_.add_argument("--profile", action="append", required=True, help="NAME=parquet")
    cmp_.add_argument("--mode", default="paper_as_published")
    cmp_.add_argument("--output")
    cmp_.set_defaults(func=cmd_compare)

    al = sub.add_parser("align-paper-cp", help="Inner-join assembled CP onto JUMP-lite wells")
    al.add_argument("--input", required=True)
    al.add_argument("--output", required=True)
    al.set_defaults(func=cmd_paper_cp_align)

    sm = sub.add_parser("smoke", help="Synthetic-image dummy pipeline")
    sm.add_argument("--workdir", default=str(repo_root() / "data" / "smoke"))
    sm.set_defaults(func=cmd_smoke)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
