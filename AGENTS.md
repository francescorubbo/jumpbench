# Agent notes

This repo generates controllable **timm** embeddings and scores **CRISPR PA**
after PCA/TVN. [JUMP_lite](https://github.com/afermg/JUMP_lite) remains the
source of truth for *their* numbers. `jumpbench compare --mode paper_as_published`
is tagged unfair on purpose.

Ignore the paper CRISPR PA NAP **0.815**. It is JUMP_lite’s PA×PC-selected
full-CRISPR config, not this study’s endpoint. We rank CRISPR-subset PA
(mean NAP) against our own arms. Run1 is a smaller slice, so 0.815 is even
less comparable. Do not write `paper_nap` / `delta_vs_paper` into result JSON
or treat 0.815 as a target.

## Ranking fairness

Living preregistration: [docs/hypotheses.md](docs/hypotheses.md).

When an analysis supports or kills a claim, update that hypothesis’s **Status**
and **Decision** in place (date + evidence). Add a new hypothesis instead of
silently expanding an old one. Do not mark an `observational` hypothesis
`falsified` because we chose not to run the experiment.

## Study constraints

Do not propose work that violates these:

1. Cellpose masks exist only for the JUMP-lite **4-site** subset.
   `--crop cell_fixed` / `cell_bbox` is 4-site-only. All-FOV (`--sites all`)
   runs are **grid tiles** only.
2. Endpoint is **CRISPR PA after PCA/TVN** (`paper_dl_default` or a declared
   DL sweep). No PC, no MOTIVE, no 11-task Figure 5 mean.
3. New representations come only from `jumpbench embed --model timm`. Do not
   re-extract CellProfiler, `cp_measure`, or MorphEM from pixels. Frozen
   comparators: paper headline numbers and assembled CPG CellProfiler profiles
   already on disk. Swapping `models.timm.architecture` (EfficientNet vs a
   DINOv2-class ViT) is in-scope for H12; that is not a re-run of paper
   DINOv2. H13 has a declared Raw-vs-MQ **timm analogue** (Wave R: cell-crop
   EfficientNetV2-XL on streamed Orig TIFF vs streamed JUMP-lite
   `jpegxl_lossy_mq.zarr`). That is not a Table 1 four-codec restudy; do not
   start HQ/D20 or re-embed paper DL families.

Campaign runbook: [docs/campaign.md](docs/campaign.md).

## Pointers

- [docs/protocol.md](docs/protocol.md) — PA/PC protocol and fairness modes
- [configs/models.yaml](configs/models.yaml) — channel recipes, tile sizes, preprocess
