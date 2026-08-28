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
stronger result. This repo is built to run that check.

## What you control

Every embedding run writes `provenance.json` next to the parquet:

- checkpoint / architecture / `pretrained`
- channel recipe (`jump_lite_as_run` vs `paper_table_s3`) and reorder
- preprocess ops (clip percentiles, 8-bit, minmax, per-channel `standard`)
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
# optional: JPEG XL zarr + cp_measure
pip install -e ".[images,cellprofiler]"
```

## Pipeline

```text
JUMP TIFFs or JUMP-lite zarr
        │
        ▼
jumpbench embed --model {dinov2,morphem,openphenom,subcell,dummy}
        │  per-site tiles, provenance.json
        ▼
jumpbench aggregate --how median
        │  well-level raw features
        ▼
jumpbench process --preset paper_dl_default
        │  RobustMAD → PCA → TVN-EFAAR (CPU)
        ▼
jumpbench evaluate --tasks pa,pc
jumpbench compare --mode fair_same_sites --profile morphem=... --profile cellprofiler_fair=...
```

Paper-as-published CellProfiler (unfair, for matching their table):

```bash
jumpbench download-paper-cp          # ~13.5 GB assembled CPG profiles
jumpbench align-paper-cp --input data/paper_cp/profiles.parquet \
  --output data/profiles/cellprofiler_paper.parquet
jumpbench compare --mode paper_as_published \
  --profile cellprofiler_paper=data/profiles/cellprofiler_paper.parquet \
  --profile morphem=data/processed/morphem.parquet
```

`compare --mode paper_as_published` is tagged **unfair** in the output on
purpose.

## Data

Frozen v1.0 manifests and RefChem matches are in this checkout
(`metadata/`, `data/refchemdb/`), copied from JUMP_lite.

| Artifact | Status | How |
|---|---|---|
| Site / well / perturbation / RefChem tables | shipped | `metadata/jump_lite_v1_*.parquet` |
| Original JUMP TIFFs | public | `jumpbench download-images --max-sites 32` |
| JUMP-lite JPEG XL zarr (MQ 92 GB, HQ 238 GB, …) | CPG promotion by mid-Sept 2026 | `configs/data.yaml` |
| Assembled CellProfiler | public ~13.5 GB | `jumpbench download-paper-cp` |
| Their per-site embeddings | CPG `workspace_dl/embeddings/` | optional; this repo prefers regenerating them |

Smoke test (no network, no weights):

```bash
jumpbench smoke
```

## Layout

```text
configs/           model cards, S3 paths, processing presets
docs/protocol.md   exact PA/PC / fairness rules
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
