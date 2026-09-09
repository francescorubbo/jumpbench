from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from jumpbench.config import apply_overrides, load_models_config, resolve_model
from jumpbench.data.compress import CODECS, DEFAULT_CODEC
from jumpbench.data.download import download_paper_cellprofiler, download_tiffs
from jumpbench.data.images import default_images_root, migrate_flat_images
from jumpbench.data.index import build_tiff_index, site_set_summary, write_tiff_index
from jumpbench.embed.generate import generate_embeddings
from jumpbench.eval.compare import compare_runs
from jumpbench.eval.metrics import PAPER_PA_CRISPR, evaluate_path
from jumpbench.paths import repo_root, resolve
from jumpbench.profiles.aggregate import aggregate_path
from jumpbench.profiles.cellprofiler import align_paper_cellprofiler
from jumpbench.profiles.normalize import DEVICES, process_path, resolve_device
from jumpbench.profiles.sweep import (
    _run_shard_payload,
    expand_grid,
    format_shard_commands,
    gather_results,
    processed_path,
    result_path,
    select_configs,
    winner_row,
)


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


def _site_filters(args: argparse.Namespace) -> dict:
    site_set = getattr(args, "sites", "all")
    explicit_subset = bool(args.site or args.source or args.plate)
    full = getattr(args, "full_cohort", False) or getattr(args, "all", False)
    if full:
        max_sites = None
        max_wells = None
    else:
        max_sites = args.max_sites
        max_wells = getattr(args, "max_wells", None)
        if max_sites is None and max_wells is None and not explicit_subset:
            if site_set == "jump_lite":
                max_sites = 32
            else:
                max_wells = 4
        elif (
            site_set == "all"
            and max_wells is None
            and max_sites is not None
            and not explicit_subset
        ):
            max_wells = max_sites
    return {
        "site_set": site_set,
        "max_sites": max_sites,
        "max_wells": max_wells,
        "sources": args.source or None,
        "site_keys": args.site or None,
        "plates": getattr(args, "plate", None) or None,
    }


def _print_index_stats(index: pl.DataFrame, path: Path | None = None) -> None:
    stats = site_set_summary(index)
    extra = f" → {path}" if path else ""
    print(
        f"{stats['site_set']}: {stats['sites']} sites / {stats['wells']} wells "
        f"(FOVs per well min/median/max "
        f"{stats['sites_per_well_min']}/{stats['sites_per_well_median']:.0f}/"
        f"{stats['sites_per_well_max']}), {stats['files']} channel URIs{extra}"
    )


def cmd_index_images(args: argparse.Namespace) -> int:
    index = build_tiff_index(**_site_filters(args))
    path = write_tiff_index(index, args.output)
    _print_index_stats(index, path)
    if args.show:
        print(index.head(args.show).select(["Metadata_Site_Key", "channel", "uri"]))
    return 0


def cmd_download_images(args: argparse.Namespace) -> int:
    filters = _site_filters(args)
    if filters["max_sites"] is None and filters["max_wells"] is None:
        if args.codec == "raw":
            print(
                "No well/site cap with --codec raw: JUMP-lite wells × all Orig FOVs "
                "is several times the paper's ~10 TB 4-site TIFF cohort.",
                file=sys.stderr,
            )
        else:
            print(
                "No well/site cap: S3 still transfers ~10–20 TB of Orig TIFF, but "
                f"--codec {args.codec} persists ~1–3% of that on disk. "
                "Pass --max-wells N for a subset.",
                file=sys.stderr,
            )
    summary = download_tiffs(
        dest=args.dest,
        jobs=args.jobs,
        dry_run=args.dry_run,
        yes=args.yes,
        index_out=args.index,
        codec=args.codec,
        **filters,
    )
    if args.dry_run:
        return 0
    print(f"URI index: {summary['index']}")
    return 0


def cmd_migrate_images(args: argparse.Namespace) -> int:
    dest = Path(args.dest) if args.dest else default_images_root()
    n = migrate_flat_images(dest)
    print(f"Nested {n} files under {dest}/{{source}}/{{batch}}/{{plate}}/{{well}}/")
    return 0


def cmd_download_paper_cp(args: argparse.Namespace) -> int:
    result = download_paper_cellprofiler(args.dest, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(result, indent=2))
        return 0
    print(f"Paper CellProfiler profiles: {result}")
    print(
        "Assembled CellProfiler is 6–9 sites/well. "
        "Match that with --sites all (default) on embeddings."
    )
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
    path = process_path(
        Path(args.input),
        Path(args.output),
        preset=args.preset,
        overrides=getattr(args, "overrides", None),
        device=getattr(args, "device", "cpu"),
    )
    print(path)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    kwargs: dict = {}
    tasks = tuple(part.strip() for part in args.tasks.split(",") if part.strip())
    if getattr(args, "subset", None) == "crispr":
        kwargs["subset"] = "crispr"
        kwargs["group_col"] = None
        kwargs["paper_ref"] = "crispr"
        if args.tasks == "pa,pc":
            tasks = ("pa",)
    result = evaluate_path(Path(args.input), tasks=tasks, **kwargs)
    printable = {k: v for k, v in result.items() if not k.startswith("_")}
    print(json.dumps(printable, indent=2, default=str))
    pa = printable.get("pa") or {}
    if pa.get("paper_nap") is not None:
        print(
            f"CRISPR PA NAP {pa['mean_nap']:.4f} vs paper {PAPER_PA_CRISPR:.3f} "
            f"(delta {pa['delta_vs_paper']:+.4f})",
            file=sys.stderr,
        )
    if args.output:
        Path(args.output).write_text(json.dumps(printable, indent=2, default=str) + "\n")
    return 0


def cmd_sweep_list(args: argparse.Namespace) -> int:
    try:
        configs = expand_grid(name=args.grid)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.limit is not None:
        configs = configs[: args.limit]
    print(f"{len(configs)} configs ({args.grid})", file=sys.stderr)
    for i, row in enumerate(configs):
        print(f"{i}\t{row['config_id']}")
    return 0


def _shard_payloads(args: argparse.Namespace) -> tuple[str, list[dict]]:
    full = expand_grid(name=args.grid)
    selected = select_configs(full, index=args.index, limit=args.limit)
    device = resolve_device(args.device)
    processed_dir = Path(args.processed_dir)
    results_dir = Path(args.results_dir)
    limited = full[: args.limit] if args.limit is not None else full
    index_by_id = {row["config_id"]: i for i, row in enumerate(limited)}
    payloads = []
    for row in selected:
        cid = row["config_id"]
        payloads.append(
            {
                "input_path": str(Path(args.input)),
                "processed": str(processed_path(processed_dir, cid)),
                "result": str(result_path(results_dir, cid)),
                "preset": args.preset,
                "overrides": row["overrides"],
                "config_id_str": cid,
                "index": index_by_id[cid],
                "grid": args.grid,
                "device": device,
                "tasks": args.tasks,
                "subset": args.subset,
                "force": args.force,
            }
        )
    return device, payloads


def cmd_sweep_run(args: argparse.Namespace) -> int:
    try:
        device, payloads = _shard_payloads(args)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    jobs = int(args.jobs)
    if device == "mps" and jobs > 1:
        print("MPS is a single GPU; forcing --jobs 1", file=sys.stderr)
        jobs = 1
    if args.dry_run:
        for payload in payloads:
            for line in format_shard_commands(
                input_path=Path(payload["input_path"]),
                processed=Path(payload["processed"]),
                result=Path(payload["result"]),
                preset=payload["preset"],
                overrides=payload["overrides"],
                device=payload["device"],
                tasks=payload["tasks"],
                subset=payload["subset"],
            ):
                print(line)
            print()
        print(f"{len(payloads)} shards (device={device}, jobs={jobs})", file=sys.stderr)
        return 0
    if jobs <= 1 or len(payloads) == 1:
        for payload in payloads:
            print(_run_shard_payload(payload))
        return 0
    failed = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_run_shard_payload, payload): payload for payload in payloads}
        for fut in as_completed(futures):
            payload = futures[fut]
            try:
                print(fut.result())
            except Exception as exc:
                failed += 1
                print(f"{payload['config_id_str']} failed: {exc}", file=sys.stderr)
    return 1 if failed else 0


def cmd_sweep_gather(args: argparse.Namespace) -> int:
    table = gather_results(args.results_dir)
    if table.height == 0:
        print(f"No completed results in {args.results_dir}", file=sys.stderr)
        return 1
    out = Path(args.output) if args.output else Path(args.results_dir) / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(out)
    print(table)
    winner = winner_row(table)
    if winner is not None:
        mean_nap = float(winner["mean_nap"])
        print(
            f"winner {winner['config_id']} CRISPR PA NAP {mean_nap:.4f} "
            f"vs paper {PAPER_PA_CRISPR:.3f} "
            f"(delta {mean_nap - PAPER_PA_CRISPR:+.4f}; CRISPR-PA selection, not paper PA×PC)",
            file=sys.stderr,
        )
    print(out)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    profiles = {}
    for spec in args.profile:
        if "=" not in spec:
            raise SystemExit("--profile must be NAME=PATH")
        name, path = spec.split("=", 1)
        profiles[name] = pl.read_parquet(path)
    table = compare_runs(
        profiles, mode=args.mode, output=Path(args.output) if args.output else None
    )
    print(table)
    if not table["fair"][0]:
        print("\nThis comparison is tagged UNFAIR. See README.md § Fairness.", file=sys.stderr)
    return 0


def cmd_paper_cp_align(args: argparse.Namespace) -> int:
    path = align_paper_cellprofiler(args.input, args.output, subset=args.subset)
    n = pl.scan_parquet(path).select(pl.len()).collect().item()
    subset = args.subset or "all"
    print(f"{n} wells written to {path} ({subset}; still 6–9-site CellProfiler values)")
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
    site_path = generate_embeddings(
        "dummy", images, root / "embeddings", models_cfg=cfg, codec="synth"
    )
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

    def _add_site_filters(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--sites",
            choices=("all", "jump_lite"),
            default="all",
            help="all = every Orig FOV on JUMP-lite wells (default, 6–9 typical). "
            "jump_lite = paper's frozen 4-site sample.",
        )
        parser.add_argument(
            "--max-wells",
            type=int,
            default=None,
            help="Cap JUMP-lite wells (keeps every FOV of those wells). "
            "Default 4 when --sites all and no other filter.",
        )
        parser.add_argument(
            "--max-sites",
            type=int,
            default=None,
            help="Cap FOVs after site selection. Default 32 only for --sites jump_lite.",
        )
        parser.add_argument(
            "--all",
            dest="full_cohort",
            action="store_true",
            help="No well/site cap (full JUMP-lite well set; huge).",
        )
        parser.add_argument(
            "--site",
            action="append",
            default=[],
            help="Metadata_Site_Key (repeatable)",
        )
        parser.add_argument(
            "--source",
            action="append",
            default=[],
            help="JUMP source, e.g. source_13 (repeatable)",
        )
        parser.add_argument(
            "--plate",
            action="append",
            default=[],
            help="Metadata_Plate (repeatable)",
        )

    idx = sub.add_parser(
        "index-images",
        help="Resolve Orig TIFF URIs from JUMP load_data CSVs (no pixel download)",
    )
    _add_site_filters(idx)
    idx.add_argument("--output", type=Path)
    idx.add_argument("--show", type=int, default=0, help="Print this many index rows")
    idx.set_defaults(func=cmd_index_images)

    d = sub.add_parser(
        "download-images",
        help="Stream Orig JUMP TIFFs from S3; persist JPEG XL by default",
    )
    _add_site_filters(d)
    d.add_argument("--dest", type=Path)
    d.add_argument(
        "--jobs",
        type=int,
        default=32,
        help="Parallel GET workers. Orig TIFFs are ~2.7 MiB; more jobs hide us-east-1 RTT.",
    )
    d.add_argument(
        "--codec",
        choices=tuple(CODECS),
        default=DEFAULT_CODEC,
        help="On-disk format. jpegxl_mq is paper MQ (~100× vs TIFF). "
        "raw writes uncompressed TIFFs (multi-TB).",
    )
    d.add_argument(
        "--dry-run",
        action="store_true",
        help="Index + size estimate only; do not transfer pixels",
    )
    d.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Do not prompt for confirmation",
    )
    d.add_argument("--index", type=Path, help="Where to write tiff_uris.parquet")
    d.set_defaults(func=cmd_download_images)

    nest = sub.add_parser(
        "migrate-images",
        help="Move flat {site}__{channel}.* files into source/batch/plate/well/",
    )
    nest.add_argument("--dest", type=Path)
    nest.set_defaults(func=cmd_migrate_images)

    c = sub.add_parser(
        "download-paper-cp",
        help="Download assembled CellProfiler profiles (unfair baseline)",
    )
    c.add_argument("--dest", type=Path)
    c.add_argument("--dry-run", action="store_true")
    c.set_defaults(func=cmd_download_paper_cp)

    e = sub.add_parser("embed", help="Generate per-site embeddings with an explicit model card")
    e.add_argument("--model", required=True)
    e.add_argument("--images", default=str(default_images_root()))
    e.add_argument("--output", default=str(resolve("data/embeddings")))
    e.add_argument(
        "--codec",
        default=DEFAULT_CODEC,
        help="Label for the embedding output folder (match download --codec)",
    )
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
    pr.add_argument(
        "--device",
        choices=DEVICES,
        default="cpu",
        help="cpu (default), mps for Apple GPU corrcoef/PCA, auto = MPS if available",
    )
    _add_overrides(pr)
    pr.set_defaults(func=cmd_process)

    ev = sub.add_parser("evaluate", help="Phenotypic activity / consistency")
    ev.add_argument("--input", required=True)
    ev.add_argument("--tasks", default="pa,pc")
    ev.add_argument(
        "--subset",
        choices=("all", "crispr"),
        default="all",
        help="crispr = keep CRISPR wells plus plate-matched negcons, then "
        "score PA against the paper CRISPR NAP (0.815)",
    )
    ev.add_argument("--output")
    ev.set_defaults(func=cmd_evaluate)

    sw = sub.add_parser(
        "sweep",
        help="Expand a process-config grid; run/gather CRISPR PA shards",
    )
    sw_sub = sw.add_subparsers(dest="sweep_cmd", required=True)

    sw_list = sw_sub.add_parser("list", help="Print config_id for each grid index")
    sw_list.add_argument("--grid", required=True)
    sw_list.add_argument("--limit", type=int)
    sw_list.set_defaults(func=cmd_sweep_list)

    sw_run = sw_sub.add_parser("run", help="Process + evaluate one index or a local --jobs pool")
    sw_run.add_argument("--grid", required=True)
    sw_run.add_argument("--preset", required=True)
    sw_run.add_argument("--input", required=True)
    sw_run.add_argument("--processed-dir", required=True)
    sw_run.add_argument("--results-dir", required=True)
    sw_run.add_argument("--index", type=int, help="0-based shard (job array contract)")
    sw_run.add_argument(
        "--jobs", type=int, default=1, help="Local process pool. Forced to 1 on MPS."
    )
    sw_run.add_argument("--limit", type=int, help="Smoke prefix of the grid")
    sw_run.add_argument("--device", choices=DEVICES, default="cpu")
    sw_run.add_argument("--tasks", default="pa")
    sw_run.add_argument(
        "--subset",
        choices=("all", "crispr"),
        default="crispr",
        help="Default crispr: rank configs by CRISPR PA",
    )
    sw_run.add_argument("--force", action="store_true", help="Redo shards that already have a JSON")
    sw_run.add_argument("--dry-run", action="store_true")
    sw_run.set_defaults(func=cmd_sweep_run)

    sw_gather = sw_sub.add_parser("gather", help="Rank completed CRISPR PA JSONs")
    sw_gather.add_argument("--results-dir", required=True)
    sw_gather.add_argument("--output", help="CSV path (default: RESULTS_DIR/summary.csv)")
    sw_gather.set_defaults(func=cmd_sweep_gather)

    cmp_ = sub.add_parser("compare", help="Score multiple representations; tags unfair modes")
    cmp_.add_argument("--profile", action="append", required=True, help="NAME=parquet")
    cmp_.add_argument("--mode", default="paper_as_published")
    cmp_.add_argument("--output")
    cmp_.set_defaults(func=cmd_compare)

    al = sub.add_parser("align-paper-cp", help="Inner-join assembled CP onto JUMP-lite wells")
    al.add_argument("--input", required=True)
    al.add_argument("--output", required=True)
    al.add_argument(
        "--subset",
        choices=("all", "crispr"),
        default="all",
        help="crispr = CRISPR wells plus plate-matched negcons (cheaper PA check)",
    )
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
