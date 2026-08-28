# JUMP-lite evaluation protocol (as implemented here)

Source of truth for *their* numbers: [afermg/JUMP_lite](https://github.com/afermg/JUMP_lite)
and [arXiv:2608.07632](https://arxiv.org/abs/2608.07632). This file is the
protocol this repo actually runs.

## Cohort

Frozen JUMP-lite v1.0: 163,776 wells, ≤4 sites/well, 655,101 sites, 6 JUMP
sources. Manifests are in `metadata/jump_lite_v1_*.parquet`.

## Representations

| Name | What this repo generates | Paper JUMP-lite source |
|---|---|---|
| CellProfiler paper | `jumpbench download-paper-cp` then `align-paper-cp` | Assembled CPG profiles, **6–9 sites/well** (S1.2.7) |
| CellProfiler fair | `cp_measure` on JUMP-lite images + deposited Cellpose masks | Not reported at JUMP-lite scale |
| DINOv2 / MorphEM / OpenPhenom / SubCell | `jumpbench embed --model …` | Aliby + Nahual on 4 sites; per-site parquets on CPG |
| Cell count | object-count columns only | Same |

## Embedding generation (controllable)

Resolved from `configs/models.yaml`:

1. Load site as `(C,H,W)` uint16, channel order **AGP, DNA, ER, Mito, RNA**.
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
plate, optional inverse-normal (CellProfiler only), greedy correlation prune,
PCA, CORAL TVN-EFAAR. Paper swept 48 DL configs and kept the max balanced
PA×PC. This repo defaults to the grid center (`configs/process.yaml`) so a
single run is interpretable; sweep knobs are listed in that file.

GPU RAPIDS numbers from JUMP_lite will not be bit-identical.

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
| `paper_as_published` | no | Reproduce their comparison, including 6–9 vs 4 sites |
| `wells_aligned_only` | no | Same wells, CP still 6–9-site values |
| `fair_same_sites` | yes | Embeddings and `cp_measure` both on the JUMP-lite 4-site set |

`jumpbench compare` refuses to silently call the paper comparison fair.
