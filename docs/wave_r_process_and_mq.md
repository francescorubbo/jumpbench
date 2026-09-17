# Wave R: processing vs JUMP-lite MQ on timm cell crops

Date: 2026-09-17.
Card: EfficientNetV2-XL, 96 px `cell_fixed` crops, five stains, site-median
pool, Run1 CRISPR, JUMP-lite 4-site keys.
Endpoint: CRISPR phenotypic activity, mean NAP (`evaluate --tasks pa --subset
crispr`). Random retrieval is 0. Material on Run1 if \|ΔNAP\| ≥ 0.03.

This is the declared Raw-vs-MQ **timm analogue** ([H13](hypotheses.md#h13--compression-robustness-headline-inherits-flawed-embedding-evaluation)),
not a Table 1 Raw/HQ/MQ/D20 restudy and not the paper’s 0.815 CRISPR PA
figure. Living verdicts stay in [hypotheses.md](hypotheses.md). Commands
also live in [campaign.md](campaign.md).

## Setup

| Knob | Value |
|---|---|
| Model | `timm` / `tf_efficientnetv2_xl.in21k` / pretrained / bag-of-channels |
| Channels | AGP, DNA, ER, Mito, RNA (`jump_lite_as_run`) |
| Crop | `--crop cell_fixed --crop-size 96 --pool site` |
| Sites | `--sites jump_lite` |
| Cohort | `--subset crispr --batch 20220914_Run1` |
| Raw pixels | streamed Orig TIFF (`--image-source s3`) |
| MQ pixels | streamed JUMP-lite `jpegxl_lossy_mq.zarr` (`--image-source s3_mq`) |
| Wells scored | 336 CRISPR perturbations after PCA/TVN or `simple_pca100` |
| Sites in both parquets | 35,398 (inner-join on `site_key`) |

MQ skipped 119 CPG zarr holes (`n_sites_skipped_no_image`). Aggregate both
arms with `--keep-sites-from` the peer run. On this pair the filter dropped
**0** extra sites (Raw already lacked those 119).

Campaign readout is now `simple_pca100` (PCA-100, then per-plate z-score
on negcons; no TVN). `paper_dl_default` and `sweep_paper_dl_v11_lite` are
comparators so process and codec are not changed at the same time.

## Findings

### 1. The paper DL recipe tanks this card (H6, mixed)

Same Raw well profiles:

| Process | Mean NAP | Median NAP |
|---|---:|---:|
| `simple_pca100` | 0.453 | 0.398 |
| Sweep winner | 0.420 | — |
| `paper_dl_default` | 0.038 | −0.149 |

`sweep_paper_dl_v11_lite` is 420 configs (normalize × fit-on-controls ×
prune × TVN ε × PCA rank). On Raw it spans **0.030–0.420** (median 0.208).
`paper_dl_default` (RobustMAD fit on negcons, PCA-128, TVN ε=0.5) is rank
**393/420**. Every RobustMAD-on-negcons config is ≤ 0.052. Z-score fit on
all wells sits at 0.38–0.42.

The reachable H6 test (“does the DL grid move *timm* CRISPR PA?”) is yes,
by ≫ 0.03. CP INT / independent-set prune is still untested.

Raw sweep winner:
`standardize`, `fit_on_controls=false`, prune, `tvn_epsilon=0.05`,
PCA-170
(`standardize_fitctrl-false_prune-true_eps-0.05_pca-170`).

### 2. MQ loses signal on the campaign readout (H13, mixed)

| Process | Raw | MQ | Δ (MQ−Raw) |
|---|---:|---:|---:|
| `simple_pca100` | 0.453 | 0.332 | **−0.120** |
| Sweep winners (picked separately) | 0.420 | 0.404 | −0.017 |
| Raw-winner config, scored on MQ | 0.420 | 0.393 | −0.028 |
| `paper_dl_default` (sweep shard) | 0.038 | 0.116 | +0.078 |

`simple_pca100` Δ is material. Stop further MQ OFAT on this card; later
OFAT stays on Raw/stream.

Do **not** use an older `data/results/campaign/run1_xl_c96_mq.json` (NAP
0.340). That file is `paper_dl_default` from before the site-intersected
profiles. The matched shard on these profiles is 0.116.

### 3. The MQ drop is not an artifact of switching process

Independently selected TVN winners miss the 0.03 bar (Δ −0.017). Matched
configs that actually keep signal still have MQ below Raw:

- All **70** configs with Raw NAP ≥ 0.38 are worse on MQ (median Δ −0.025,
  range −0.011 to −0.036; 19 of them ≤ −0.03).
- MQ sweep winner
  (`standardize_fitctrl-false_prune-false_eps-0.05_pca-304`) is 0.404;
  that same config is 0.418 on Raw (Δ −0.014).
- `paper_dl_default` still flips sign because it destroys Raw, not because
  MQ is better.

So: MQ is consistently a bit worse when the process preserves phenotype;
`simple_pca100` amplifies that past 0.03; the paper DL default is not a
fair codec comparison.

MQ sweep range: 0.086–0.404 (median 0.223).

![DL process sweep ranked by Raw NAP, MQ overlaid, simple_pca100 dashed](figures/wave_r_sweep_waterfall.png)

420 `sweep_paper_dl_v11_lite` configs, sorted by Raw CRISPR PA mean NAP; MQ is the same process config (not re-ranked). Dashed lines are `simple_pca100`. The x-axis is the `normalize` × fit-on-negcons blocks that appear after that sort; prune, TVN ε, and PCA rank jitter inside blocks. Generate with `python scripts/plot_sweep_waterfall.py`.

## Result files

All under `data/` (gitignored):

```text
data/profiles/timm_run1_xl_c96_{raw,mq}.parquet
data/profiles/timm_run1_xl_c96_{raw,mq}.parquet.site_filter.json
data/processed/timm_run1_xl_c96_raw_simple_pca100.parquet
data/processed/timm_run1_xl_c96_mq_simple_pca100.parquet
data/results/campaign/run1_xl_c96_raw_simple_pca100.json
data/results/campaign/run1_xl_c96_mq_simple_pca100.json
data/results/campaign/timm_run1_xl_c96_raw_dl_sweep/summary.csv
data/results/campaign/timm_run1_xl_c96_mq_dl_sweep/summary.csv
```

## Reproduce

Needs GPU for embed, CPU for process/evaluate/sweep. Streamed S3; do not
write Orig TIFFs next to local `.jxl`. `--jobs 16` pins one BLAS thread
per process.

### Embed (skip if the run dirs already have `site_embeddings.parquet`)

```bash
jumpbench embed --model timm --image-source s3 \
  --subset crispr --sites jump_lite --batch 20220914_Run1 \
  --crop cell_fixed --crop-size 96 --pool site \
  --run-dir data/embeddings/timm/run1/cell_fixed_jump_lite_96_efficientnetv2_xl_5ch_raw \
  --set models.timm.architecture=tf_efficientnetv2_xl.in21k \
  --set runtime.batch_size=64

jumpbench embed --model timm --image-source s3_mq \
  --subset crispr --sites jump_lite --batch 20220914_Run1 \
  --crop cell_fixed --crop-size 96 --pool site \
  --run-dir data/embeddings/timm/run1/cell_fixed_jump_lite_96_efficientnetv2_xl_5ch_jpegxl_mq \
  --set models.timm.architecture=tf_efficientnetv2_xl.in21k \
  --set runtime.batch_size=64
```

### Same sites, then `simple_pca100`

```bash
RAW=data/embeddings/timm/run1/cell_fixed_jump_lite_96_efficientnetv2_xl_5ch_raw
MQ=data/embeddings/timm/run1/cell_fixed_jump_lite_96_efficientnetv2_xl_5ch_jpegxl_mq

jumpbench aggregate \
  --input $RAW/site_embeddings.parquet \
  --keep-sites-from $MQ \
  --output data/profiles/timm_run1_xl_c96_raw.parquet

jumpbench aggregate \
  --input $MQ/site_embeddings.parquet \
  --keep-sites-from $RAW \
  --output data/profiles/timm_run1_xl_c96_mq.parquet

jumpbench process --preset simple_pca100 \
  --input data/profiles/timm_run1_xl_c96_raw.parquet \
  --output data/processed/timm_run1_xl_c96_raw_simple_pca100.parquet
jumpbench evaluate --tasks pa --subset crispr \
  --input data/processed/timm_run1_xl_c96_raw_simple_pca100.parquet \
  --output data/results/campaign/run1_xl_c96_raw_simple_pca100.json

jumpbench process --preset simple_pca100 \
  --input data/profiles/timm_run1_xl_c96_mq.parquet \
  --output data/processed/timm_run1_xl_c96_mq_simple_pca100.parquet
jumpbench evaluate --tasks pa --subset crispr \
  --input data/processed/timm_run1_xl_c96_mq_simple_pca100.parquet \
  --output data/results/campaign/run1_xl_c96_mq_simple_pca100.json
```

Expect Raw CRISPR PA NAP ≈ 0.453 and MQ ≈ 0.332.

### Same profiles, paper DL default (optional comparator)

```bash
jumpbench process --preset paper_dl_default \
  --input data/profiles/timm_run1_xl_c96_raw.parquet \
  --output data/processed/timm_run1_xl_c96_raw.parquet
jumpbench evaluate --tasks pa --subset crispr \
  --input data/processed/timm_run1_xl_c96_raw.parquet \
  --output data/results/campaign/run1_xl_c96_raw.json

jumpbench process --preset paper_dl_default \
  --input data/profiles/timm_run1_xl_c96_mq.parquet \
  --output data/processed/timm_run1_xl_c96_mq.parquet
jumpbench evaluate --tasks pa --subset crispr \
  --input data/processed/timm_run1_xl_c96_mq.parquet \
  --output data/results/campaign/run1_xl_c96_mq.json
```

Expect Raw ≈ 0.038. MQ on these profiles ≈ 0.116 (sweep shard), not 0.340.

### Same profiles, 420-config DL sweep

Grid parent is still `--preset paper_dl_default`; axes override it
(`configs/process.yaml` `sweep_paper_dl_v11_lite`).

```bash
jumpbench sweep run --grid sweep_paper_dl_v11_lite --preset paper_dl_default \
  --input data/profiles/timm_run1_xl_c96_raw.parquet \
  --processed-dir data/processed/timm_run1_xl_c96_raw_dl_sweep \
  --results-dir data/results/campaign/timm_run1_xl_c96_raw_dl_sweep \
  --jobs 16 --subset crispr --tasks pa
jumpbench sweep gather --results-dir data/results/campaign/timm_run1_xl_c96_raw_dl_sweep

jumpbench sweep run --grid sweep_paper_dl_v11_lite --preset paper_dl_default \
  --input data/profiles/timm_run1_xl_c96_mq.parquet \
  --processed-dir data/processed/timm_run1_xl_c96_mq_dl_sweep \
  --results-dir data/results/campaign/timm_run1_xl_c96_mq_dl_sweep \
  --jobs 16 --subset crispr --tasks pa
jumpbench sweep gather --results-dir data/results/campaign/timm_run1_xl_c96_mq_dl_sweep
```

Completed JSONs are skipped unless `--force`. On this machine each sweep was
about 2.5 hours at 16 jobs.

Matched Raw-winner on MQ (no extra embed):

```bash
jumpbench process --preset paper_dl_default \
  --set normalize=standardize \
  --set fit_on_controls=false \
  --set prune_correlated=true \
  --set tvn_epsilon=0.05 \
  --set pca_components=170 \
  --input data/profiles/timm_run1_xl_c96_mq.parquet \
  --output data/processed/timm_run1_xl_c96_mq_raw_winner.parquet
jumpbench evaluate --tasks pa --subset crispr \
  --input data/processed/timm_run1_xl_c96_mq_raw_winner.parquet \
  --output data/results/campaign/run1_xl_c96_mq_raw_winner.json
```

That shard already exists as
`data/results/campaign/timm_run1_xl_c96_mq_dl_sweep/standardize_fitctrl-false_prune-true_eps-0.05_pca-170.json`
(NAP 0.393).

### Waterfall figure

```bash
python scripts/plot_sweep_waterfall.py
# docs/figures/wave_r_sweep_waterfall.png
# docs/figures/wave_r_sweep_waterfall.svg
```
