# JUMP-lite evaluation protocol (as implemented here)

Source of truth for *their* numbers: [afermg/JUMP_lite](https://github.com/afermg/JUMP_lite)
and [arXiv:2608.07632](https://arxiv.org/abs/2608.07632). This file is the
protocol this repo actually runs.

## Cohort

Frozen JUMP-lite v1.0: 163,776 wells, ≤4 sites/well, 655,101 sites, 6 JUMP
sources. Manifests are in `metadata/jump_lite_v1_*.parquet`.

## Images (original JUMP, not JUMP-lite zarr)

Pixels are original JUMP TIFFs from Cell Painting Gallery
(`s3://cellpainting-gallery/cpg0016-jump/...`). URI discovery copies
JUMP_lite `prep/build_jl_index.sql`:

1. For each needed `(source, batch, plate)`, fetch
   `workspace/load_data_csv/{batch}/{plate}/load_data_with_illum.csv`.
2. Keep `URL_Orig{AGP,DNA,ER,Mito,RNA}` (illumination functions are not applied).
3. Restrict to JUMP-lite **wells**. By default keep **every Orig FOV** on
   those wells (typically 6–9, the same support assembled CellProfiler used).
   `--sites jump_lite` instead inner-joins the frozen 4-site keys.
4. **Stream** each Orig TIFF from S3 and persist JPEG XL (paper MQ,
   `Jpegxl(lossless=False, distance=3.0)`) at
   `data/images/{source}/{batch}/{plate}/{well}/{site}__{channel}.jxl`.

```bash
jumpbench download-images --max-wells 1          # all FOVs of 1 well (often 9)
jumpbench download-images --sites jump_lite --max-wells 1   # paper's 4 sites
jumpbench download-images --codec raw --max-wells 1         # uncompressed TIFFs
```

The paper's embedding run used 4 sites; their CellProfiler numbers used 6–9.
`--sites all` is therefore the default. Full well set × all FOVs is tens of
TB of S3 TIFF and ~200–300 GB of JPEG XL MQ on disk.

## Representations

| Name | What this repo generates | Paper JUMP-lite source |
|---|---|---|
| CellProfiler paper | `jumpbench download-paper-cp` then `align-paper-cp` | Assembled CPG profiles, **6–9 sites/well** (S1.2.7) |
| CellProfiler fair | `cp_measure` on the same pixels as the embeddings | Not reported at JUMP-lite scale |
| DINOv2 / MorphEM / OpenPhenom / SubCell | `jumpbench embed --model …` | Paper: 4 sites. This repo default: **all Orig FOVs** |
| Cell count | object-count columns only | Same |

## Embedding generation (controllable)

Resolved from `configs/models.yaml`:

1. Load site as `(C,H,W)` uint16 from JPEG XL / JPEG / TIFF, channel order **AGP, DNA, ER, Mito, RNA**.
2. Select `channels` for the active `channel_recipe`.
3. Optional `model_channel_order` reorder (SubCell `rybg`).
4. Apply `preprocess` ops in order (clip / minmax / 8-bit / per-tile `standard`).
5. Non-overlapping `crop` tiles of `tile_size` (Aliby `kind: crop`; remainder dropped).
6. Forward through the backend (`checkpoint`, `pretrained`, `batch_size`, `device`).
7. Write per-site features + `provenance.json`.

Override any of this without editing YAML:

```bash
jumpbench embed --model dinov2 --set channel_recipe=paper_table_s3 \
  --set models.dinov2.tile_size=256 --set models.dinov2.pretrained=false
```

`jump_lite_as_run` is the default because that is what `prep/aliby_featurize.py`
and `cpg_upload/JUMP_LITE_README.md` actually used. Table S3 disagrees for
DINOv2 and SubCell.

## Aggregation

Median over tiles, then median over sites → one vector per well. Matches
JUMP_lite `src/extract_features.py` for long-form DL outputs. `--how mean` is
available as a control.

## Profile processing

CPU reimplementation of the variance-first recipe (`src/norm_3`): drop NA >
30%, low-variance filter, RobustMAD (or z-score) fit on negative controls per
plate, optional inverse-normal (CellProfiler only), correlation prune
(independent-set for CP, greedy for DL), then TVN-EFAAR. For CellProfiler,
PCA is fit on controls *inside* TVN-EFAAR (JUMP_lite v11-lite CP). Paper S1.2.6
swept 280 engineered-feature configs and kept the max min-max-rescaled PA×PC.
This repo's `sweep_paper_cp_v11` is that grid, but **ranks by CRISPR PA only**.
`paper_cp_default` is the grid center, not the paper winner.

GPU RAPIDS numbers from JUMP_lite will not be bit-identical. Optional
`--device mps` runs corrcoef and PCA on Apple Silicon; CORAL and copairs stay
on CPU. MPS forces `--jobs 1`.

```bash
jumpbench sweep run --grid sweep_paper_cp_v11 --preset paper_cp_default \
  --input data/profiles/cellprofiler_paper_all.parquet \
  --processed-dir data/processed/cp_v11 \
  --results-dir data/results/cp_v11 \
  --jobs 4
jumpbench sweep gather --results-dir data/results/cp_v11
```

`evaluate --tasks pa --subset crispr` keeps CRISPR wells plus plate-matched
negcons. Process can be fit on all JUMP-lite wells (the paper) or on a
CRISPR-only `align-paper-cp --subset crispr` slice (cheaper). The published
0.815 is a directional check: it came from the PA×PC-selected config.

## Metrics

- **PA**: copairs average precision, perturbation replicates vs plate-matched
  DMSO/negcon; report mean NAP (random = 0).
- **PC**: same-target vs different-target compounds from RefChem (more than
  one supporting record, gene-encoded proteins).
- **MOTIVE recall@k%**: not in the first cut of the CLI; annotations and the
  allowlist are shipped (`metadata/motive_eval_compounds.parquet`) for a later
  port of `src/motive/evaluate_motive.py`.

## Fairness modes

| `--mode` | Fair? | Meaning |
|---|---|---|
| `paper_as_published` | no | Their table: CP 6–9 sites vs embeddings 4 sites |
| `wells_aligned_only` | no | Same wells, embeddings still 4-site |
| `fair_all_sites` | yes | Embeddings aggregated over all Orig FOVs vs assembled CP (both 6–9) |
| `fair_same_sites` | yes | Embeddings and `cp_measure` both on the same FOV set |

`jumpbench compare` refuses to silently call the paper comparison fair.
