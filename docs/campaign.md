# JUMP-lite ranking campaign

Runbook for the CRISPR PA + timm experiments declared in
[hypotheses.md](hypotheses.md). Protocol details stay in [protocol.md](protocol.md).

Endpoint: **CRISPR phenotypic activity (mean NAP)** after `paper_dl_default`
PCA/TVN, fit on the same CRISPR wells that were embedded. No PC, MOTIVE,
11-task mean, MorphEM/DINOv2/SubCell re-embed, or Raw/HQ/MQ restudy.

Images are already on disk (`data/images/`, JPEG XL MQ, all Orig FOVs). New
embeddings write under `data/embeddings/timm/...` (gitignored). Results go in
`data/results/campaign/` (also gitignored). Hypothesis **Status** / **Decision**
in `docs/hypotheses.md` is the durable record.

## Declared baseline (B0)

Change one axis per arm.

| Knob | B0 value |
|---|---|
| Model | `timm` / `tf_efficientnet_b0` / pretrained / bag-of-channels |
| Channels | all 5 (`jump_lite_as_run` / `five_stain`) |
| Crop | `grid`, `tile_size=224` (`crop_size` defaults to `tile_size`) |
| Sites | `--sites jump_lite` (frozen 4-site) |
| Cohort | CRISPR `--subset crispr --batch 20220914_Run1` |
| Images | `jpegxl_mq` |
| Pool | `--pool site` |
| Process | `paper_dl_default`, CRISPR wells only |
| Score | `evaluate --tasks pa --subset crispr` |

Pilot materiality: an arm **moves** B0 if \|ΔNAP\| ≥ **0.03** on Run1.
Full-CRISPR confirm: **material** if \|ΔNAP\| ≥ **0.02**.

Run directories must not collide. Always pass `--run-dir`.

```text
data/embeddings/timm/run1/grid_jump_lite_224_efficientnet_b0_5ch/
data/embeddings/timm/run1/grid_jump_lite_224_efficientnet_b0_dinov2_as_run/
data/embeddings/timm/run1/cell_fixed_jump_lite_96_efficientnet_b0_5ch/
data/embeddings/timm/run1/cell_fixed_jump_lite_128_efficientnet_b0_5ch/
data/embeddings/timm/crispr/grid_jump_lite_224_efficientnet_b0_5ch/
```

## Wave 0 — Tooling (done in this repo)

`jumpbench embed` now honors `--subset`, `--sites {all,jump_lite}`, `--batch`,
`--source`, `--plate`, `--pool {crop,site}`, and `--run-dir`. Site shards resume
after a crash (`shards/` + `completed_sites.txt`). Cell crops remain 4-site only.

Named timm recipes: `five_stain`, `dinov2_as_run`, `table_s3_dinov2`,
`subcell_as_run` (plus the existing `jump_lite_as_run` / `paper_table_s3`).

## Wave 0b — One-plate smoke

Pick the first Run1 plate that exists locally, e.g. `CP-CC9-R1-01`.

```bash
jumpbench embed --model dummy --images data/images \
  --subset crispr --sites jump_lite --batch 20220914_Run1 \
  --plate CP-CC9-R1-01 --pool site \
  --run-dir data/embeddings/dummy/smoke_run1_plate

# Real backbone, one plate (weights download on first run)
jumpbench embed --model timm --images data/images \
  --subset crispr --sites jump_lite --batch 20220914_Run1 \
  --plate CP-CC9-R1-01 --pool site \
  --run-dir data/embeddings/timm/smoke_run1_plate \
  --set runtime.batch_size=8

jumpbench aggregate \
  --input data/embeddings/timm/smoke_run1_plate/site_embeddings.parquet \
  --output data/profiles/timm_smoke_run1_plate.parquet
jumpbench process --preset paper_dl_default \
  --input data/profiles/timm_smoke_run1_plate.parquet \
  --output data/processed/timm_smoke_run1_plate.parquet
jumpbench evaluate --tasks pa --subset crispr \
  --input data/processed/timm_smoke_run1_plate.parquet \
  --output data/results/campaign/smoke_run1_plate.json

# Masks for later cell-crop arms (Run1 4-site CRISPR)
jumpbench download-masks --subset crispr --batch 20220914_Run1 --jobs 16
```

One-plate PA is not a campaign score. It only checks the pipeline.

## Wave 1 — B0 on Run1

```bash
jumpbench embed --model timm --images data/images \
  --subset crispr --sites jump_lite --batch 20220914_Run1 \
  --pool site \
  --run-dir data/embeddings/timm/run1/grid_jump_lite_224_efficientnet_b0_5ch \
  --set runtime.batch_size=16

jumpbench aggregate \
  --input data/embeddings/timm/run1/grid_jump_lite_224_efficientnet_b0_5ch/site_embeddings.parquet \
  --output data/profiles/timm_run1_b0.parquet
jumpbench process --preset paper_dl_default \
  --input data/profiles/timm_run1_b0.parquet \
  --output data/processed/timm_run1_b0.parquet
jumpbench evaluate --tasks pa --subset crispr \
  --input data/processed/timm_run1_b0.parquet \
  --output data/results/campaign/run1_b0.json
```

Matched-well CP comparator (still 6–9-site features; tag unfair on site count):

```python
import polars as pl
from pathlib import Path

src = Path("data/profiles/cellprofiler_paper_crispr.parquet")
out = Path("data/profiles/cellprofiler_paper_crispr_run1.parquet")
pl.read_parquet(src).filter(pl.col("Metadata_Batch") == "20220914_Run1").write_parquet(out)
```

```bash
jumpbench process --preset paper_cp_default \
  --input data/profiles/cellprofiler_paper_crispr_run1.parquet \
  --output data/processed/cellprofiler_paper_crispr_run1.parquet
jumpbench evaluate --tasks pa --subset crispr \
  --input data/processed/cellprofiler_paper_crispr_run1.parquet \
  --output data/results/campaign/run1_cp.json
```

## Wave 2 — Cheap OFAT (same Run1, 4-site, `paper_dl_default`)

Each arm is B0 plus one `--set` / `--sites` change. Reuse the Wave 1 aggregate
→ process → evaluate block with a new `--run-dir` and profile names.

| Arm | Hypothesis | Extra flags |
|---|---|---|
| C3a | H4, H8, H10 | `--set channel_recipe=dinov2_as_run` |
| C3b | H4, H8 | `--set channel_recipe=table_s3_dinov2` |
| C4 | H4, H10 | `--set channel_recipe=subcell_as_run` |
| T448 | H3, H9 | `--set models.timm.tile_size=448` |
| T256 | H3, H9 | `--set models.timm.tile_size=256` |
| ViT | H12 | `--set models.timm.architecture=vit_small_patch14_dinov2.lvd142m --set runtime.batch_size=8` |
| Psweep | H6 | no re-embed; see below |

```bash
jumpbench sweep run --grid sweep_paper_dl_v11_lite --preset paper_dl_default \
  --input data/profiles/timm_run1_b0.parquet \
  --processed-dir data/processed/timm_run1_b0_dl_sweep \
  --results-dir data/results/campaign/timm_run1_b0_dl_sweep \
  --jobs 4 --subset crispr --tasks pa
jumpbench sweep gather --results-dir data/results/campaign/timm_run1_b0_dl_sweep
```

Report `paper_dl_default` **and** the sweep winner. Do not treat best-of-sweep
as the representation score.

## Wave 2.5 — Cell-crop window OFAT (H14)

Do **not** start until Wave 2 is scored. Grid T256/T448 stay the H3/H9
native-window sweep (EfficientNet does not resize). 224 is Wave 3 CELL, not
re-run here.

| Arm | Hypothesis | Extra flags |
|---|---|---|
| C96 | H14 | `--crop cell_fixed --crop-size 96` |
| C128 | H14 | `--crop cell_fixed --crop-size 128` |

```bash
jumpbench embed --model timm --images data/images \
  --subset crispr --sites jump_lite --batch 20220914_Run1 \
  --crop cell_fixed --crop-size 96 --pool site \
  --run-dir data/embeddings/timm/run1/cell_fixed_jump_lite_96_efficientnet_b0_5ch \
  --set runtime.batch_size=16
# same with --crop-size 128 and run-dir .../cell_fixed_jump_lite_128_...
```

Record `n_objects_skipped_edge` and cells kept. Compare C96 / C128 to B0
grid **and** to Wave 3 CELL@224.

## Wave 3 — Expensive OFAT

| Arm | Hypothesis | Extra flags |
|---|---|---|
| ALL | H2 | `--sites all` (grid only) |
| CELL | H1, H14 | `--crop cell_fixed` (`crop_size=224` unless C96 or C128 \|ΔNAP\| ≥ 0.03 vs this arm) |
| BBOX | H1 | `--crop cell_bbox` only if CELL \|ΔNAP\| ≥ 0.03; inherit Wave 2.5 `crop_size` if CELL did |
| XL | H12 | `--set models.timm.architecture=tf_efficientnetv2_xl.in21k --set runtime.batch_size=4` |

If C96 or C128 beats CELL@224 by \|ΔNAP\| ≥ 0.03, CELL/BBOX inherit that
`crop_size` for any later re-run or Wave 4 promotion.

XL keeps `tile_size=224`. Conv nets do not resize; ViT-S vs XL is not
capacity-matched.

## Wave 4 — Full CRISPR confirmation

Promote at most two cards to all 142 CRISPR plates (`--subset crispr`, drop
`--batch`), still `--sites jump_lite` unless ALL was the only mover:

1. B0
2. The single largest-effect arm from Waves 2–3 (including Wave 2.5)

```bash
jumpbench compare --mode paper_as_published \
  --profile timm_b0=data/processed/timm_crispr_b0.parquet \
  --profile cellprofiler_paper=data/processed/cellprofiler_paper_crispr.parquet \
  --output data/results/campaign/compare_paper_as_published.csv
```

Do not call a 4-site timm vs 6–9-site CP comparison `fair_all_sites`.

## Observational (no new pixels)

H5, H7, H11, H13 stay **open** with Decision “not tested in this study.”
Do not mark them `falsified`.

## Closing a hypothesis

After the relevant wave, edit that hypothesis in [hypotheses.md](hypotheses.md):
**Status**, dated **Decision**, command, comparison mode, NAP. Add a new
hypothesis instead of silently expanding an old one.
