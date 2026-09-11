# JUMP-lite reproduction: CellProfiler vs embeddings

Reproduce [Muñoz, Fredin Haslum, Shen, Carpenter, Singh, 2026](https://arxiv.org/abs/2608.07632)
(*JUMP-lite: Compact, reproducible benchmarking of cell representations*) with
**explicit control over embedding generation**.

Their code: [afermg/JUMP_lite](https://github.com/afermg/JUMP_lite). This repo
does not vendor that tree. It copies the frozen v1.0 metadata, reimplements
the evaluation protocol in a linear CLI, and replaces Nahual/Aliby IPC with a
model card you can edit.

## Why rebuild this

The paper’s headline is that classical CellProfiler features and MorphEM lead a
weaker mid-tier of DINOv2 / SubCell / OpenPhenom (Figure 5, uncompressed):

| Representation | Mean score (column-normalized across 11 tasks) |
|---|---:|
| MorphEM | 0.944 |
| CellProfiler | 0.936 |
| DINOv2 | 0.721 |
| SubCell | 0.715 |
| OpenPhenom | 0.688 |
| Cell count | 0.402 |
| Random ViT | 0.252 |

That ranking is **not** an apples-to-apples embedding comparison:

1. **Different site counts (S1.2.7).** Full JUMP-lite CellProfiler numbers use
   precomputed Cell Painting Gallery profiles from **6–9 sites/well**. Every
   deep model used the JUMP-lite **4-site** cohort. More sites typically
   stabilize well-level profiles.
2. **Different feature extractors.** Target-2 compression experiments used
   `cp_measure` on the same images. JUMP-lite used assembled CellProfiler, not
   `cp_measure` on JUMP-lite pixels.
3. **Train/test overlap.** MorphEM and OpenPhenom were pretrained on data that
   includes JUMP. The paper flags this.
4. **Best-of-sweep post-processing.** Each model×codec kept the Hydra config
   with the best balanced PA×PC (48 DL configs). Rankings mix representation
   quality with who got the luckiest recipe.
5. **Table S3 ≠ the released driver.** DINOv2 in the manuscript is
   Nuclei/AGP/Mito; `prep/aliby_featurize.py` feeds zarr channels `[0,1,2]` =
   **AGP, DNA, ER**. SubCell Table S3 is Nuclei/AGP/Mito/RNA; the CPG README is
   Mito/ER/DNA/AGP (`rybg`). Both recipes are in `configs/models.yaml`.

If CellProfiler still wins after (1) and (2) are equalized, that is a much
stronger result. This repo is built to run that check. Living preregistration
of these and further ranking-fairness claims: [docs/hypotheses.md](docs/hypotheses.md).

## What you control

Every embedding run writes `provenance.json` next to the parquet:

- checkpoint / architecture / `pretrained`
- channel recipe (`jump_lite_as_run` vs `paper_table_s3`) and reorder
- preprocess ops (clip percentiles, 8-bit, minmax, per-channel `standard`, percentile min-max)
- `preprocess_scope` (`site` on the FOV vs `tile` per crop)
- tile size, non-overlapping crop
- batch size, device, seed
- tile and site aggregation (`median` default, `mean` optional)

```bash
jumpbench models dinov2
jumpbench embed --model dinov2 --images data/images \
  --set channel_recipe=paper_table_s3 \
  --set models.dinov2.tile_size=256 \
  --set models.dinov2.checkpoint=facebookresearch/dinov2 \
  --set runtime.batch_size=8
```

Swap in a local weight file with `--set models.morphem.checkpoint=/path/to/ckpt`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# GPU embedding backends
pip install -e ".[embed]"
# optional: JUMP-lite zarr readers + cp_measure
pip install -e ".[images,cellprofiler]"
```

## Pipeline

```text
JUMP TIFFs streamed from S3 → JPEG XL on disk
        │
        ▼
jumpbench embed --model {dinov2,morphem,openphenom,subcell,timm,dummy}
        │  per-site tiles, provenance.json
        ▼
jumpbench aggregate --how median
        │  well-level raw features
        ▼
jumpbench process --preset paper_dl_default
        │  RobustMAD → PCA → TVN-EFAAR (CPU)
        ▼
jumpbench evaluate --tasks pa,pc
jumpbench compare --mode fair_all_sites --profile morphem=... --profile cellprofiler_paper=...
```

Paper-as-published CellProfiler, CRISPR phenotypic activity. Process fits on
all JUMP-lite wells. `evaluate --subset crispr` keeps CRISPR wells plus
plate-matched negcons. Ranking a process-config grid uses CRISPR PA only
(not the paper's min-max-rescaled PA×PC):

```bash
jumpbench download-paper-cp          # ~13.5 GB assembled CPG profiles
jumpbench align-paper-cp --input data/paper_cp/profiles.parquet \
  --output data/profiles/cellprofiler_paper_all.parquet \
  --subset all
jumpbench sweep list --grid sweep_paper_cp_v11
jumpbench sweep run --grid sweep_paper_cp_v11 --preset paper_cp_default \
  --input data/profiles/cellprofiler_paper_all.parquet \
  --processed-dir data/processed/cp_v11 \
  --results-dir data/results/cp_v11 \
  --jobs 4
# job array: --index N  (N in 0..279)
# Apple GPU corrcoef/PCA: --device mps --jobs 1
jumpbench sweep gather --results-dir data/results/cp_v11
```

Embeddings reuse the same evaluate job with a different `--input`, `--grid`,
and `--preset` (for example `sweep_paper_dl_v11_lite` / `paper_dl_default`).

`--subset crispr` on `align-paper-cp` still exists for a cheaper CRISPR-only
process. The paper's CRISPR PA NAP is **0.815** (Figure 5, the config that
won on balanced PA×PC). This sweep picks the max CRISPR PA, so 0.815 is a
directional check, not a same-recipe target. A single CPU `paper_cp_default`
run will not match it either.

Full-cohort `compare --mode paper_as_published` is tagged **unfair** in the
output on purpose.

## Data

Frozen v1.0 manifests and RefChem matches are in this checkout
(`metadata/`, `data/refchemdb/`), copied from JUMP_lite. Images and
CellProfiler profiles come from the **original public JUMP objects**, not
from JUMP-lite's still-unpublished JPEG XL zarrs.

The paper's uncompressed ranking (Figure 5) compared 4-site embeddings to
assembled CellProfiler that still averages 6–9 sites/well. Because this repo
streams Orig TIFFs directly, **the default is all FOVs per JUMP-lite well**
so that comparison can be site-matched. `--sites jump_lite` restores their
4-site sample.

Orig TIFFs are ~2.6 MiB per channel. The full well set × all FOVs is
**tens of TB** — it will not fit on a 1 TB disk. `download-images` therefore
**streams each TIFF from S3 and only writes JPEG XL** (paper MQ, Butteraugli
distance 3.0, ~100× smaller; JUMP-lite's 4-site MQ corpus is ~116 GB). Leftover
local `.tif` files from earlier runs are transcoded and deleted. Use
`--codec raw` only if you actually want uncompressed TIFFs.

```bash
# All Orig FOVs on 4 JUMP-lite wells (typically 6–9 sites each).
# Prints S3-transfer vs on-disk estimates and asks to confirm; --yes skips.
# Re-run to resume: complete .jxl files are skipped.
# Files land in data/images/{source}/{batch}/{plate}/{well}/ — not one flat folder.
jumpbench download-images --max-wells 4
jumpbench migrate-images   # once, if an older run wrote flat {site}__{channel}.jxl
jumpbench download-images --dry-run --max-wells 4   # estimate only

# Paper's 4-site embedding cohort
jumpbench download-images --sites jump_lite --max-wells 4

# Uncompressed TIFFs (needs a multi-TB disk)
jumpbench download-images --codec raw --max-wells 1

# Assembled CellProfiler (~13.5 GB, 6–9 sites/well)
jumpbench download-paper-cp --dry-run
jumpbench download-paper-cp

# Cellpose masks for --crop cell_fixed / cell_bbox (cached under data/masks/)
jumpbench download-masks --max-wells 4
# embed also prefetches any missing masks before the GPU loop
jumpbench embed --model timm --crop cell_fixed --images data/images
```

`--max-wells` defaults to 4 when `--sites all` and you pass no other filter.
`--sites jump_lite` defaults to 32 FOVs. Full cohort S3 transfer is still
~10–20 TB; on-disk JPEG XL MQ is roughly 200–300 GB (`--all`).

| Artifact | Status | How |
|---|---|---|
| Site / well / perturbation / RefChem tables | shipped | `metadata/jump_lite_v1_*.parquet` |
| Original JUMP pixels | public now | `jumpbench download-images` streams TIFF, persists JPEG XL MQ |
| Cellpose instance masks | public now | `jumpbench download-masks` (also prefetched by `--crop cell_*` embed) |
| Assembled CellProfiler | public now ~13.5 GB | `jumpbench download-paper-cp` (`cpg0016-jump-assembled` v1.0c) |
| JUMP-lite JPEG XL zarr (MQ 92 GB, HQ 238 GB, …) | CPG promotion mid-Sept 2026 | compression study only; not required for Figure 5-style raw embeddings |
| Their per-site embeddings | CPG `workspace_dl/embeddings/` | optional; this repo regenerates them |

Smoke test (no network, no weights):

```bash
jumpbench smoke
```

## Layout

```text
configs/           model cards, S3 paths, processing presets
docs/protocol.md   exact PA/PC / fairness rules
docs/hypotheses.md living ranking-fairness preregistration
AGENTS.md          study constraints for agents working in this repo
metadata/          frozen JUMP-lite v1.0 cohort
src/jumpbench/     download → embed → aggregate → process → evaluate
tests/             channel recipes, tiling, aggregation, fairness flags
```

Upstream analysis, Hydra GPU sweeps, and figure scripts stay in
[JUMP_lite](https://github.com/afermg/JUMP_lite). Point at that repo when you
want bit-identical RAPIDS sweeps; use this one when you want to change how
embeddings are made.

## License

MIT. JUMP / Cell Painting Gallery data terms still apply to images and
assembled profiles.
