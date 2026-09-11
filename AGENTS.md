# Agent notes

This repo generates controllable **timm** embeddings and scores **CRISPR PA**
after PCA/TVN. [JUMP_lite](https://github.com/afermg/JUMP_lite) remains the
source of truth for *their* numbers. `jumpbench compare --mode paper_as_published`
is tagged unfair on purpose.

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
   DINOv2. H13 (compression-robustness inheriting embedding flaws) is
   observational; do not start a Raw/HQ/MQ/D20 re-benchmark unless asked.

Campaign runbook: [docs/campaign.md](docs/campaign.md).

## Pointers

- [docs/protocol.md](docs/protocol.md) — PA/PC protocol and fairness modes
- [configs/models.yaml](configs/models.yaml) — channel recipes, tile sizes, preprocess
