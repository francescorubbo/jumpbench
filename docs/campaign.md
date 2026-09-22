# JUMP-lite ranking campaign

Runbook for the CRISPR PA + timm experiments declared in
[hypotheses.md](hypotheses.md). Protocol details stay in [protocol.md](protocol.md).

Endpoint: **CRISPR phenotypic activity (mean NAP)** after `simple_pca100`
(PCA-100, plate negcon z-score), fit on the same CRISPR wells that were embedded. No PC, MOTIVE,
11-task mean, MorphEM/DINOv2/SubCell re-embed, or Table 1 Raw/HQ/MQ/D20
restudy. Wave R is a **declared Raw-vs-MQ timm analogue** (H13), not that
restudy.

MQ for 4-site arms is JUMP-lite `jpegxl_lossy_mq.zarr` streamed from CPG
(`--image-source s3_mq`). Local `data/images/` JPEG XL remains for already
downloaded all-FOV files; do not delete it. Raw Orig TIFFs are **streamed**
at embed time (`--image-source s3`); do not write them next to `.jxl`. New
embeddings write under
`data/embeddings/timm/...` (gitignored). Results go in
`data/results/campaign/` (also gitignored). Hypothesis **Status** /
**Decision** in `docs/hypotheses.md` is the durable record.

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
| Process | `simple_pca100`, CRISPR wells only |
| Score | `evaluate --tasks pa --subset crispr` |

Pilot materiality: an arm **moves** B0 if \|ΔNAP\| ≥ **0.03** on Run1.
Full-CRISPR confirm: **material** if \|ΔNAP\| ≥ **0.02**.

Run directories must not collide. Always pass `--run-dir`.

```text
data/embeddings/timm/run1/grid_jump_lite_224_efficientnet_b0_5ch/
data/embeddings/timm/run1/grid_jump_lite_224_efficientnet_b0_dinov2_as_run/
data/embeddings/timm/run1/cell_fixed_jump_lite_96_efficientnetv2_xl_5ch_raw/
data/embeddings/timm/run1/cell_fixed_jump_lite_96_efficientnetv2_xl_5ch_jpegxl_mq/
data/embeddings/timm/run1/grid_jump_lite_224_efficientnetv2_xl_5ch_raw/
data/embeddings/timm/run1/grid_jump_lite_224_efficientnetv2_xl_5ch_jpegxl_mq/
data/embeddings/timm/run1/grid_jump_lite_224_vit_small_dinov2_5ch_raw/
data/embeddings/timm/run1/grid_jump_lite_224_vit_small_dinov2_5ch_jpegxl_mq/
data/embeddings/timm/crispr/grid_jump_lite_224_efficientnet_b0_5ch/
```

## Wave 0 — Tooling (done in this repo)

`jumpbench embed` now honors `--subset`, `--sites {all,jump_lite}`, `--batch`,
`--source`, `--plate`, `--pool {crop,site}`, `--run-dir`, `--image-source
{local,s3,s3_mq}`, and `--prefetch-jobs`. `--image-source s3` streams Orig
TIFFs and does not probe local `.jxl`. `--image-source s3_mq` streams
JUMP-lite `jpegxl_lossy_mq.zarr` (4-site only). Site shards resume after a
crash (`shards/` + `completed_sites.txt`). Cell crops remain 4-site only.

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
jumpbench process --preset simple_pca100 \
  --input data/profiles/timm_smoke_run1_plate.parquet \
  --output data/processed/timm_smoke_run1_plate.parquet
jumpbench evaluate --tasks pa --subset crispr \
  --input data/processed/timm_smoke_run1_plate.parquet \
  --output data/results/campaign/smoke_run1_plate.json

# Masks for later cell-crop arms (Run1 4-site CRISPR)
jumpbench download-masks --subset crispr --batch 20220914_Run1 --jobs 16
```

One-plate PA is not a campaign score. It only checks the pipeline.

S3 cell-crop smoke (dummy, one well, Orig TIFF stream; do not persist TIFFs):

```bash
jumpbench embed --model dummy --image-source s3 \
  --subset crispr --sites jump_lite --batch 20220914_Run1 \
  --plate CP-CC9-R1-01 --max-wells 1 \
  --crop cell_fixed --crop-size 96 --pool site \
  --run-dir data/embeddings/dummy/smoke_s3_cell96_raw

jumpbench embed --model dummy --image-source s3_mq \
  --subset crispr --sites jump_lite --batch 20220914_Run1 \
  --plate CP-CC9-R1-01 --max-wells 1 \
  --crop cell_fixed --crop-size 96 --pool site \
  --run-dir data/embeddings/dummy/smoke_s3_mq_cell96
```

## Wave R — Raw champion, then MQ twin (H13)

Write-up of the scores: [wave_r_process_and_mq.md](wave_r_process_and_mq.md).

XL at 96 px is not the Wave 3 224 px memory case: use `runtime.batch_size=64`
(raise to 128 if VRAM allows). `batch_size=4` is ~350 CUDA syncs per site.

```bash
# Run1 champion (stream Orig TIFF)
jumpbench embed --model timm --image-source s3 \
  --subset crispr --sites jump_lite --batch 20220914_Run1 \
  --crop cell_fixed --crop-size 96 --pool site \
  --run-dir data/embeddings/timm/run1/cell_fixed_jump_lite_96_efficientnetv2_xl_5ch_raw \
  --set models.timm.architecture=tf_efficientnetv2_xl.in21k \
  --set runtime.batch_size=64

# Run1 MQ twin (stream JUMP-lite jpegxl_lossy_mq.zarr)
jumpbench embed --model timm --image-source s3_mq \
  --subset crispr --sites jump_lite --batch 20220914_Run1 \
  --crop cell_fixed --crop-size 96 --pool site \
  --run-dir data/embeddings/timm/run1/cell_fixed_jump_lite_96_efficientnetv2_xl_5ch_jpegxl_mq \
  --set models.timm.architecture=tf_efficientnetv2_xl.in21k \
  --set runtime.batch_size=64

# Intersect site_key both ways so CPG zarr/TIFF holes drop on the peer
# (no re-embed). A well must not be 4-site Raw vs 3-site MQ.
jumpbench aggregate \
  --input data/embeddings/timm/run1/cell_fixed_jump_lite_96_efficientnetv2_xl_5ch_raw/site_embeddings.parquet \
  --keep-sites-from data/embeddings/timm/run1/cell_fixed_jump_lite_96_efficientnetv2_xl_5ch_jpegxl_mq \
  --output data/profiles/timm_run1_xl_c96_raw.parquet
jumpbench process --preset simple_pca100 \
  --input data/profiles/timm_run1_xl_c96_raw.parquet \
  --output data/processed/timm_run1_xl_c96_raw_simple_pca100.parquet
jumpbench evaluate --tasks pa --subset crispr \
  --input data/processed/timm_run1_xl_c96_raw_simple_pca100.parquet \
  --output data/results/campaign/run1_xl_c96_raw_simple_pca100.json

jumpbench aggregate \
  --input data/embeddings/timm/run1/cell_fixed_jump_lite_96_efficientnetv2_xl_5ch_jpegxl_mq/site_embeddings.parquet \
  --keep-sites-from data/embeddings/timm/run1/cell_fixed_jump_lite_96_efficientnetv2_xl_5ch_raw \
  --output data/profiles/timm_run1_xl_c96_mq.parquet
jumpbench process --preset simple_pca100 \
  --input data/profiles/timm_run1_xl_c96_mq.parquet \
  --output data/processed/timm_run1_xl_c96_mq_simple_pca100.parquet
jumpbench evaluate --tasks pa --subset crispr \
  --input data/processed/timm_run1_xl_c96_mq_simple_pca100.parquet \
  --output data/results/campaign/run1_xl_c96_mq_simple_pca100.json
```

If \|ΔNAP\| (Raw vs MQ) ≥ 0.03, **stop MQ OFAT** and ablate on Raw/stream
only. If not, later OFAT may stay on MQ. Full CRISPR (drop `--batch`) only
if the Run1 champion is material vs B0-on-MQ or vs this MQ twin.

`simple_pca100` Δ is −0.120 (stop MQ OFAT). Independently swept TVN winners
are Δ −0.017; matched high-NAP configs still have MQ below Raw.

Score both arms on the **intersection** of embedded `site_key`s. MQ can skip
CPG zarr holes (`n_sites_skipped_no_image`, `skipped_sites_no_image.txt`);
`--keep-sites-from` drops those sites from Raw well aggregation so a well is
not 4-site Raw vs 3-site MQ. Filter both ways in case Raw also missed TIFFs.

### Processing first (H6), then compression (H13)

On those Raw wells vs the MQ twin, CRISPR PA mean NAP (same 336 perturbations):

| Process | Raw | MQ | Δ (MQ−Raw) |
|---|---:|---:|---:|
| `simple_pca100` | 0.453 | 0.332 | −0.120 |
| `sweep_paper_dl_v11_lite` winner | 0.420 | 0.404 | −0.017 |
| matched Raw-winner config on MQ | 0.420 | 0.393 | −0.028 |
| `paper_dl_default` | 0.038 | 0.116 | +0.078 |

Sweep winners are independently selected (`standardize`, fit-on-all-wells, ε=0.05; Raw prune+PCA-170, MQ no-prune+PCA-304). Every config with Raw NAP ≥ 0.38 is worse on MQ (n=70, median Δ −0.025, range −0.011 to −0.036). The 0.12 `simple_pca100` drop is larger than the TVN-grid drop; `paper_dl_default` still flips sign because it tanks Raw.

```bash
jumpbench sweep run --grid sweep_paper_dl_v11_lite --preset paper_dl_default \
  --input data/profiles/timm_run1_xl_c96_mq.parquet \
  --processed-dir data/processed/timm_run1_xl_c96_mq_dl_sweep \
  --results-dir data/results/campaign/timm_run1_xl_c96_mq_dl_sweep \
  --jobs 16 --subset crispr --tasks pa
jumpbench sweep gather --results-dir data/results/campaign/timm_run1_xl_c96_mq_dl_sweep
```

The DL Raw grid spans 0.030–0.420 (420 configs). RobustMAD fit on negcons never
exceeds 0.052 on Raw. Sweep `--jobs` > 1 pins one BLAS thread per process.

## Wave R.5 — Grid 224 XL / ViT, Raw and MQ (H1, H3, H12, H13)

Same Run1 CRISPR 4-site keys as Wave R cell-96 XL (35,398 `site_key`s via
`--keep-sites-from` that run). Grid embeddings have no Cellpose skip, so
the filter drops 1,726 Raw / 1,607 MQ extra sites. `simple_pca100` only
(no DL sweep). Full table: [wave_r_process_and_mq.md](wave_r_process_and_mq.md).

| Crop | Model | Raw NAP | MQ NAP | Δ MQ−Raw |
|---|---|---:|---:|---:|
| `cell_fixed` 96 | EfficientNetV2-XL | 0.453 | 0.332 | **−0.120** |
| `grid` 224 | EfficientNetV2-XL | 0.403 | 0.329 | **−0.073** |
| `grid` 224 | ViT-S DINOv2 (timm) | 0.397 | 0.333 | **−0.063** |

Material movers on Run1 (\|ΔNAP\| ≥ 0.03): cell-96 vs grid-224 on XL **Raw**
(+0.050); Raw vs MQ on every card. XL vs ViT on grid is **not** material
(Raw Δ +0.006). Cell vs grid on **MQ** is not material (Δ +0.003). Confounded
with H14 (`crop_size` 96 vs `tile_size` 224). H12 here is `simple_pca100`,
not the planned `paper_dl_default`.

```bash
CELL=data/embeddings/timm/run1/cell_fixed_jump_lite_96_efficientnetv2_xl_5ch_raw
EMB=data/embeddings/timm/run1

# example: XL grid Raw; same block for xl_g224_mq, vit_g224_raw, vit_g224_mq
jumpbench aggregate \
  --input $EMB/grid_jump_lite_224_efficientnetv2_xl_5ch_raw/site_embeddings.parquet \
  --keep-sites-from $CELL \
  --output data/profiles/timm_run1_xl_g224_raw.parquet
jumpbench process --preset simple_pca100 \
  --input data/profiles/timm_run1_xl_g224_raw.parquet \
  --output data/processed/timm_run1_xl_g224_raw_simple_pca100.parquet
jumpbench evaluate --tasks pa --subset crispr \
  --input data/processed/timm_run1_xl_g224_raw_simple_pca100.parquet \
  --output data/results/campaign/timm_run1_xl_g224_raw_simple_pca100.json
```

Embed recipe: `scripts/run_grid_embeds.sh` (tmux `jumpbench-grid`).

Fallback if streaming is flaky **and** ≥0.5 TiB is free (separate root so
`.jxl` cannot win):

```bash
jumpbench download-images --codec raw --dest data/images_raw \
  --subset crispr --sites jump_lite --batch 20220914_Run1 --yes
```

## Wave 1 — B0 on Run1

Paused until Wave R (H13) is scored.

```bash
jumpbench embed --model timm --images data/images \
  --subset crispr --sites jump_lite --batch 20220914_Run1 \
  --pool site \
  --run-dir data/embeddings/timm/run1/grid_jump_lite_224_efficientnet_b0_5ch \
  --set runtime.batch_size=16

jumpbench aggregate \
  --input data/embeddings/timm/run1/grid_jump_lite_224_efficientnet_b0_5ch/site_embeddings.parquet \
  --output data/profiles/timm_run1_b0.parquet
jumpbench process --preset simple_pca100 \
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

## Wave 2 — Cheap OFAT (same Run1, 4-site, `simple_pca100`)

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
as the representation score. Wave R already ran this grid on XL cell-96 Raw
(`data/results/campaign/timm_run1_xl_c96_raw_dl_sweep/`).

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

H5, H7, H11 stay **open** with Decision “not tested in this study.”
H13 is **partial** (Wave R analogue); still do not mark it `falsified` by
omission, and do not expand it into HQ/D20.

## Closing a hypothesis

After the relevant wave, edit that hypothesis in [hypotheses.md](hypotheses.md):
**Status**, dated **Decision**, command, comparison mode, NAP. Add a new
hypothesis instead of silently expanding an old one.
