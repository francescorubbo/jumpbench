# JUMP-lite ranking hypotheses (preregistration)

Living register of why the paper’s Figure 5 ranking may not be a fair
representation comparison, and why its compression-robustness headline may
inherit the same evaluation problems. Claims stay listed even when this
study cannot close them. Verdicts go here, not in the README.

Paper: [Muñoz et al., 2026](https://arxiv.org/abs/2608.07632)
(*JUMP-lite: Compact, reproducible benchmarking of cell representations*).
Their code: [afermg/JUMP_lite](https://github.com/afermg/JUMP_lite) v1.0.
Protocol this repo actually runs: [protocol.md](protocol.md).

## How to update

- **Status:** `open` | `verified` | `falsified` | `mixed` | `superseded`
- **Testability:** `in-scope` | `partial` | `observational`
  (`observational` = listed; no experiment in this study)
- Change `Status`, add a dated **Decision** with evidence (command, comparison
  mode, figure). Do not delete the original **Claim**.
- Add a new hypothesis instead of silently expanding an old one.
- Do not mark an observational hypothesis `falsified` because we chose not to
  run the experiment.

## Study constraints

These bind every planned test.

1. **Masks exist only for the JUMP-lite 4-site subset.** `--crop cell_fixed` /
   `cell_bbox` is 4-site-only. All-FOV (`--sites all`) work is **grid tiles**
   only.
2. **Endpoint is CRISPR phenotypic activity after PCA/TVN.** No PC, no MOTIVE,
   no 11-task column-normalized mean. Process presets stay in the DL family
   (`paper_dl_default` or a declared sweep); that is part of the measurement,
   not an extra analysis.
3. **No new CellProfiler, `cp_measure`, or MorphEM features from pixels.** New
   representations come only from `jumpbench embed --model timm`. Frozen
   comparators allowed: paper headline numbers and already-assembled CPG
   CellProfiler well profiles on disk (no re-extraction). MorphEM is not
   re-embedded.

What this study *can* vary, all inside timm + CRISPR PA + PCA/TVN:

- grid vs cell crops (cell crops: 4-site masks only)
- `--sites all` vs `--sites jump_lite` (grid only for all-FOV)
- `tile_size` / remainder coverage
- `cell_fixed` `--crop-size` (native cell window; H14). Grid native window is
  already H3/H9 via `tile_size` (EfficientNet does not resize).
- channel recipe (keep / drop stains) and bag-of-channels concat (timm’s
  native forward)
- `models.timm.architecture` (conv vs DINOv2-class ViT; H12)

## Summary

| ID | Claim (short) | Testability | Status |
|---|---|---|---|
| H1 | CP pathway conflates segmentation, features, and post-processing | partial | open |
| H2 | 4-site embeddings vs 6–9-site CellProfiler | partial | open |
| H3 | Grid tiles lack cell inductive bias; tile size differs by model | partial | open |
| H4 | Channel handling is inconsistent (concat vs stack + drops) | partial | open |
| H5 | MorphEM / OpenPhenom train/test overlap with JUMP | observational | open |
| H6 | Asymmetric post-processing grids and config selection | partial | open |
| H7 | Illumination correction and extractor mismatch | observational | open |
| H8 | Table S3 ≠ released embedding driver | partial | open |
| H9 | Tile remainder changes spatial coverage | in-scope | open |
| H10 | Bag-of-channels concat makes embedding dim incomparable | partial | open |
| H11 | Cell Count^ inherits extra CP sites | observational | open |
| H12 | DINOv2 is a weak natural-image baseline vs convnets (EfficientNet) | in-scope | open |
| H13 | Compression-robustness headline inherits flawed embedding evaluation | observational | open |
| H14 | Default 224 px cell windows are not cell-scale | in-scope | open |

---

## H1 — CP pathway conflates distinct steps

**Claim.** The Figure 5 CellProfiler row is not an embedding extractor
comparable to DINOv2. It is a full classical pipeline: (1) instance
segmentation, (2) per-cell engineered features, (3) well aggregation and
CP-specific post-processing.

**Why it would bias the ranking.** A CellProfiler win can come from cell-level
inductive bias, from the handcrafted feature set, from production-pipeline
effects (illumination, aggregation over more sites), or from extra processing
(inverse-normal transform, independent-set prune)—not from “classical features
beat learned embeddings.”

**What the paper / JUMP_lite / this repo actually did.**

1. **Segmentation.** Target-2 used Cellpose + `cp_measure` (Figure 1c, S1.6.3).
   Full JUMP-lite ranking used archived CellProfiler, which also segments.
   Learned models used Aliby `tile.kind = crop` on the FOV
   (`prep/aliby_featurize.py`). This repo exposes `--crop cell_fixed` /
   `cell_bbox` in `src/jumpbench/embed/generate.py`.
2. **Feature extraction.** Full-scale CP is assembled CPG `ALL/v1.0c`, not
   `cp_measure` on JUMP-lite pixels (S1.6.8;
   JUMP_lite `prep/fetch_cp_profiles.py`). Target-2 *did* run `cp_measure` on
   the same images; that is not the Figure 5 CP number.
3. **Post-processing.** Both families get PCA-TVN. CP uniquely also gets
   inverse-normal transform and independent-set correlation prune (S1.6.5;
   `configs/process.yaml` `paper_cp_default` vs `paper_dl_default`). JUMP
   production CellProfiler applies illumination correction; this repo’s
   embeddings do not (`docs/protocol.md`).

**Testability:** partial. We cannot re-extract CP or MorphEM, so we cannot
unconfound CP’s feature set from its segmentation, nor give MorphEM cell
crops.

**Planned test.** On the 4-site CRISPR cohort, compare timm **grid** vs timm
**cell_fixed / cell_bbox**, same channel recipe, same PCA/TVN, CRISPR PA.
Optionally compare those to frozen assembled-CP CRISPR PA on the same wells
(still 6–9-site CP features — do not call that site-matched).

**Falsifier.** Reachable: cell crops do not improve timm CRISPR PA vs grid on
the 4-site set. Then the “missing cell inductive bias” arm is not supported
*for timm*. The CP-extractor and CP-only-processing arms stay open.
Not reachable here: a same-pixel `cp_measure` vs embedding comparison.

**Status:** open

**Decision:** 

---

## H2 — 4-site embeddings vs 6–9-site CellProfiler

**Claim.** Figure 5 compares embeddings aggregated over the JUMP-lite 4-site
sample to CellProfiler well profiles that average **6–9** Orig FOVs per well
(not a fixed 9). JUMP-lite: 163,773 wells with 4 sites, 3 with 3 (S1.1).
Cell Count^ inherits the extra sites.

**Why it would bias the ranking.** More sites typically stabilize well-level
profiles. Extra FOVs can lift CP (and Cell Count) without any representation
advantage.

**What the paper / JUMP_lite / this repo actually did.** Figure 5 caption and
S1.6.8 flag CellProfiler* as more than four sites. Embeddings used four.
This repo: `--sites all` (default, all Orig FOVs on JUMP-lite wells) vs
`--sites jump_lite`; `compare --mode fair_all_sites` vs `paper_as_published`
(`docs/protocol.md`).

**Testability:** partial. We can add sites on the **timm grid** path only. We
cannot 4-site-subsample or re-extract CellProfiler, and cell crops cannot use
extra FOVs (no masks).

**Planned test.** timm grid `--sites all` vs `--sites jump_lite`, same card,
PCA/TVN, CRISPR PA. Frozen assembled CP remains the 6–9-site comparator
(`paper_as_published` / `wells_aligned_only`), tagged unfair on site count.

**Falsifier.** Reachable: all-FOV timm CRISPR PA is not materially higher than
4-site timm. Then extra sites do not explain a timm vs 4-site-timm gap. They
can still explain paper CP vs 4-site embeddings; that half stays open without
a 4-site CP.

**Status:** open

**Decision:**

---

## H3 — Grid tiles lack cell inductive bias; tile size differs

**Claim.** Learned models embed large non-overlapping grid tiles of the FOV,
not individual cells. Tile size also differs by model, so spatial support is
not matched.

| Model | Tile | Channels as run | Notes |
|---|---:|---|---|
| DINOv2 / MorphEM / ViT-rand | 224 | 3 or 5 | JUMP FOV ~1080×1280 → 4×5 tiles; remainder dropped |
| OpenPhenom | 256 | 5 | same 4×5 grid, different remainder |
| SubCell | 448 | 4 | 2×2 tiles, much more discarded |

**Why it would bias the ranking.** CellProfiler looks at instances. A 448 px
grid also sees a different fraction of the FOV than a 224 px grid (see H9).
Mid-tier spread among DINOv2 / SubCell / OpenPhenom can be a crop artifact.

**What the paper / JUMP_lite / this repo actually did.** Table S6,
`aliby_featurize.py` `tile_size`, CPG README, and `configs/models.yaml`.
Aliby crop drops right/bottom remainder (`src/jumpbench/embed/tiling.py`).

**Testability:** partial. Tile-size and cell-vs-grid tests are timm-only.
Cell vs grid is 4-site-only. We will not re-embed DINOv2 / OpenPhenom /
SubCell / MorphEM at a common tile size.

**Planned test.** (a) timm grid `tile_size` sweep (e.g. 224 / 256 / 448) on a
declared site set; (b) same as H1 cell vs grid on 4-site CRISPR PA.

**Falsifier.** Reachable: (a) matched tile sizes do not move timm CRISPR PA →
size is not causal *for timm*; (b) cell crops do not beat grid on 4-site timm
→ cell inductive bias is not causal *for timm*. Paper-model tile-size
confounding stays observational.

**Status:** open

**Decision:**

---

## H4 — Channel handling is inconsistent

**Claim.** Models do not see the same stains, and they do not combine stains
the same way. Concatenation (bag-of-channels) vs stacking, plus dropping
channels for RGB or 4-channel models, are mixed in one ranking.

As-run (`selected_channels` on zarr order AGP, DNA, ER, Mito, RNA):

- **DINOv2:** stack `[0,1,2]` = AGP/DNA/ER as RGB; drops Mito and RNA. Table S3
  lists Nuclei/AGP/Mito instead.
- **MorphEM:** bag-of-channels (1ch forward, concatenate embeddings) on all 5
  (`src/jumpbench/embed/backends.py` `MorphEmBackend`).
- **OpenPhenom:** 5-channel stack.
- **SubCell:** 4 channels with `rybg` reorder (as-run Mito/ER/DNA/AGP, drops
  RNA); Table S3 is Nuclei/AGP/Mito/RNA.

Preprocessing is also inconsistent (Table S6): DINOv2/MorphEM `standard`;
OpenPhenom clip 0.5% + 8-bit + standard; SubCell minmax (paper table also
lists clip 0.5%; CPG README vs Table S6 disagree for SubCell).

**Why it would bias the ranking.** Dropping organelles removes signal.
Concatenating five single-channel forwards is a different inductive bias from
feeding a 3-channel RGB stack into an ImageNet ViT. The ranking attributes
the gap to “the model.”

**What the paper / JUMP_lite / this repo actually did.**
`prep/aliby_featurize.py` `selected_channels`; CPG README; Table S6 vs Table
S3. This repo defaults to `channel_recipe: jump_lite_as_run`;
`paper_table_s3` is the manuscript recipe (`configs/models.yaml`). Timm is
bag-of-channels: each stain is repeated to RGB, forwarded, concatenated.

**Testability:** partial. We can drop/reorder stains and compare 3ch vs 5ch
concat under CRISPR PA. We cannot re-run DINOv2 stacking vs MorphEM concat vs
SubCell `rybg`.

**Planned test.** timm channel-recipe grid on a declared crop/site setting
(all 5 stains vs DINOv2-as-run 3 vs Table S3 3 vs SubCell-as-run 4), same
PCA/TVN, CRISPR PA.

**Falsifier.** Reachable: channel subset does not move timm CRISPR PA. Then
dropping stains is not a first-order effect *for this backbone*. The paper’s
concat-vs-stack contrast is not directly tested (timm does not stack five
microscopy channels as a single multi-channel tensor).

**Status:** open

**Decision:**

---

## H5 — Train/test overlap (MorphEM, OpenPhenom)

**Claim.** MorphEM was pretrained on CHAMMI-75, which includes JUMP (S1.6.6:
up to 11.7% of JUMP-lite wells / 47% of plates by acquisition identity).
OpenPhenom documents training on RxRx3 + JUMP; plates are unpublished.

**Why it would bias the ranking.** Leakage can inflate MorphEM (and possibly
OpenPhenom) relative to DINOv2 / SubCell / CellProfiler.

**What the paper / JUMP_lite / this repo actually did.** The paper flags this
and still saw MorphEM lead after dropping matched wells/plates (Figure S19).
That analysis uses acquisition identity as a proxy, not a leakage-free
evaluation.

**Testability:** observational. We will not re-embed MorphEM or drop
overlapping wells. Timm ImageNet / timm pretraining is not JUMP; that is a
qualitative contrast only.

**Planned test.** None — observational.

**Falsifier.** Not reachable here.

**Status:** open

**Decision:** 2026-09-11 — not tested in this study. Campaign constraint: no MorphEM / OpenPhenom re-embed. Status stays `open` (observational; not falsified by omission).

---

## H6 — Asymmetric post-processing and config selection

**Claim.** CellProfiler and learned embeddings were not processed with the
same recipe family. CP: 280 configs, inverse-normal, independent-set prune,
PCA-TVN. DL: up to 175 effective pipelines, no INT, PCA-TVN. Compression
analyses pick the max min-max-rescaled PA×PC. Figure 5 itself **averages**
across normalization configurations (caption)—it is not a best-of-sweep
table.

**Why it would bias the ranking.** Extra CP-only steps, different grids, and
(in other figures) picking the luckiest recipe mix representation quality
with processing. Averaging configs (Figure 5) is a different selection rule
than this repo’s CRISPR-PA ranking.

**What the paper / JUMP_lite / this repo actually did.** S1.6.5; Figure 5
caption; `configs/process.yaml` `sweep_paper_cp_v11` vs
`sweep_paper_dl_v11_lite`. This repo ranks CRISPR PA only.

**Testability:** partial. We will run PCA/TVN on timm and may sweep that grid
for CRISPR PA. We will not re-sweep CP or MorphEM, and we will not reproduce
the 11-task mean.

**Planned test.** Process timm well profiles with `paper_dl_default` (and
optionally `sweep_paper_dl_v11_lite`), score CRISPR PA. Frozen assembled CP
may be scored under its own preset as an unfair comparator, not as a matched
processing ablation.

**Falsifier.** Reachable only for “does the DL processing grid move *timm*
CRISPR PA.” Not reachable: whether CP’s INT / independent-set prune caused
the paper lead.

**Status:** open

**Decision:**

---

## H7 — Illumination and extractor mismatch

**Claim.** Paper embeddings (and this repo) use Orig TIFFs without applying
illumination functions (`docs/protocol.md`). Assembled CellProfiler comes
from JUMP production pipelines that include illumination correction,
segmentation, and CellProfiler measurement—not `cp_measure` on JUMP-lite
pixels.

**Why it would bias the ranking.** Illum-corrected, production-pipeline CP
features are a different input than raw-ish tiles fed to a network.

**What the paper / JUMP_lite / this repo actually did.** S1.6.8; JUMP FAQ on
CP pipelines; this repo keeps `URL_Orig*` only.

**Testability:** observational. The fair extractor control would be
`cp_measure` on JUMP-lite pixels (`cellprofiler_fair`) — out of scope (no new
CP features).

**Planned test.** None — observational.

**Falsifier.** Not reachable here.

**Status:** open

**Decision:** 2026-09-11 — not tested in this study. No `cp_measure` on JUMP-lite pixels. Status stays `open` (observational; not falsified by omission).

---

## H8 — Table S3 ≠ released embedding driver

**Claim.** Manuscript Table S6 / S3 channel lists disagree with
`prep/aliby_featurize.py` and the CPG embedding README for DINOv2 and
SubCell.

**Why it would bias the ranking.** Reproducing “the paper” from the table
yields a different input than the deposited embeddings.

**What the paper / JUMP_lite / this repo actually did.** Documented in
`configs/models.yaml` and tested in `tests/test_pipeline.py`
(`jump_lite_as_run` vs `paper_table_s3`).

**Testability:** partial analogue via timm channel recipes (H4). Direct
DINOv2 / SubCell re-featurization is out of scope.

**Planned test.** Covered by H4’s timm channel-recipe grid.

**Falsifier.** Same as H4 for timm. Paper-model mismatch stays a documented
discrepancy, not an experiment we will re-run.

**Status:** open

**Decision:**

---

## H9 — Tile remainder changes spatial coverage

**Claim.** Non-overlapping crop drops right/bottom remainder. On a typical
1080×1280 JUMP site, 224 and 256 px both yield a 4×5 grid but discard
different strips; 448 px yields 2×2 and discards much more.

**Why it would bias the ranking.** Models with larger tiles see fewer crops
and a smaller fraction of the FOV. Confounded with H3 tile size.

**What the paper / JUMP_lite / this repo actually did.** Aliby `kind: crop`;
`src/jumpbench/embed/tiling.py`.

**Testability:** in-scope for timm (same as H3a). Confounded with crop mode
on 4-site cell runs.

**Planned test.** timm grid `tile_size` sweep; record tiles per site and
fraction of FOV kept in provenance.

**Falsifier.** Reachable: CRISPR PA is stable across tile sizes that change
coverage. Then remainder is not first-order *for timm*.

**Status:** open

**Decision:**

---

## H10 — Bag-of-channels concat makes embedding dim incomparable

**Claim.** MorphEM (and this repo’s timm) concatenates per-channel forwards,
so raw dimension scales with stain count (~5× a single-forward ViT). PCA-TVN
then projects families to similar rank; covariance structure before TVN still
differs.

**Why it would bias the ranking.** Apparent MorphEM strength can be
“five independently embedded stains” rather than a better architecture.
Comparing pre-TVN distances across models is not apples-to-apples.

**What the paper / JUMP_lite / this repo actually did.** MorphEM
`bag_of_channels: true`; timm same pattern in
`src/jumpbench/embed/backends.py`.

**Testability:** partial. Timm concat dim scales with stain count (H4). We
will not re-embed MorphEM.

**Planned test.** Same as H4: 3 vs 4 vs 5 stain concat on timm, CRISPR PA
after PCA/TVN. Optionally record pre-TVN dim in provenance.

**Falsifier.** Reachable: stain-count (hence dim) does not move post-TVN
timm CRISPR PA.

**Status:** open

**Decision:**

---

## H11 — Cell Count^ inherits extra CP sites

**Claim.** Figure 5 Cell Count^ is derived from archived CellProfiler well
profiles and therefore had access to the same additional sites as
CellProfiler*. MQ Cellpose Counts† (Table S3b) is the 4-site analogue.

**Why it would bias the ranking.** The “simple baseline” is not matched to
the 4-site embedding cohort. Paper already notes cell count matches mid-tier
DL on CRISPR PA.

**What the paper / JUMP_lite / this repo actually did.** Figure 5 caption;
Table S3.

**Testability:** observational. No new CP-derived counts planned.

**Planned test.** None — observational.

**Falsifier.** Not reachable here.

**Status:** open

**Decision:** 2026-09-11 — not tested in this study. No new CP-derived cell counts. Status stays `open` (observational; not falsified by omission).

---

## H12 — DINOv2 is a weak natural-image baseline vs convnets

**Claim.** The paper’s DINOv2 row is a weak natural-image baseline for Cell
Painting. ViT/DINO features are a poor match to cell morphology;
**convolutional** encoders (EfficientNet as the named example) should do
better because they capture **texture at multiple resolutions**. Figure 5’s
mid-tier “deep learning” bucket is partly “we picked a transformer pretrained
on natural RGB photos,” not “learned embeddings cannot match CellProfiler.”

**Why it would bias the ranking.** Treating DINOv2 as *the* generalist
vision baseline understates what natural-image encoders can do. A stronger
conv baseline can shrink or close the gap to MorphEM / CellProfiler without
any microscopy-specific pretraining.

**What the paper / JUMP_lite / this repo actually did.** Paper DINOv2 is
`dinov2_vits14` on a 3-channel RGB stack (as-run AGP/DNA/ER), not
bag-of-channels (H4). This repo’s timm card defaults to `tf_efficientnet_b0`
and swaps backbones with `--set models.timm.architecture=...`
(`configs/models.yaml`). Conv backbones keep native crop size; ViTs resize
(`src/jumpbench/embed/backends.py`).

**Testability:** in-scope under the timm recipe; partial vs the paper DINOv2
run. The reachable test is architecture with **same** channels,
bag-of-channels, crop, sites, and PCA/TVN. It does not re-run JUMP_lite’s
stacked DINOv2 (H4).

**Planned test.** Hold crop, site set, channel recipe, and `paper_dl_default`
PCA/TVN fixed. Compare CRISPR PA for:

- default conv: `tf_efficientnet_b0` (campaign B0)
- conv champion: `tf_efficientnetv2_xl.in21k` (ImageNet-21k; not capacity-matched
  to ViT-S)
- transformer: `vit_small_patch14_dinov2.lvd142m` (DINOv2-class timm id, not
  the Nahual `dinov2` family)

**Falsifier.** Reachable: EfficientNetV2-XL does not beat the DINOv2-class
timm backbone on CRISPR PA. Then “conv nets are better at morphology texture”
is not supported *under this protocol*. Not reachable: re-ranking paper
Figure 5’s stacked DINOv2 row.

**Status:** open

**Decision:**

---

## H13 — Compression-robustness headline inherits flawed embedding evaluation

**Claim.** The paper’s central storage claim — that moderate JPEG XL
compression causes only marginal to negligible downstream loss (HQ mean
−1.2%, MQ −6.5% vs Raw; Table 1) — is estimated from the same learned
embeddings whose ranking evaluation is potentially flawed (H1–H4, H13).
Models that do not see cells, drop stains, or lack multi-scale texture
inductive bias can look compression-robust because they never used the
signal that lossy codecs destroy.

**Why it would bias the ranking.** If the readout is insensitive to fine
morphology, “compression preserves phenotypic signal” is a property of the
readout, not of the pixels. The headline then overstates how safe MQ/HQ are
for Cell Painting profiling in general. JUMP-lite CellProfiler and Cell
Count have **no** compression sweep (Raw-only; Discussion, Figure 5
caption). Target-2 `cp_measure` checks feature correlation and segmentation
AP, not JUMP-lite PA/PC. Table 1 pools MorphEM, SubCell, DINOv2, and
OpenPhenom only.

**What the paper / JUMP_lite / this repo actually did.** Section 3.4 and
Table 1: per-task %Δ vs lossless Zstd on the four learned families.
Figure 3c Target-2 pools those families plus `cp_measure` over
normalization configs. S1.6.8: full-scale CP is archived Raw profiles.
This repo persists JPEG XL MQ from Orig TIFFs for embeddings; JUMP-lite
zarr codec variants are “compression study only” and not required for
Figure 5-style runs (`README.md`). We will not re-embed MorphEM / DINOv2 /
SubCell / OpenPhenom at Raw/HQ/MQ/D20.

**Testability:** observational for Table 1. An optional analogue — timm
CRISPR PA on a small Raw vs MQ subset, especially cell-crop EfficientNet
vs grid ViT — is not in the current vary-list and is not promised (raw
Orig TIFFs are tens of TB). Do not treat MQ-only timm scores as a
compression restudy.

**Planned test.** None — observational. If a declared Raw-vs-MQ timm subset
is run later, record it here as a partial analogue, not a Table 1
replication.

**Falsifier.** Not reachable here. A later analogue would falsify the
“insensitive readout” arm if cell-crop convnets show the same small MQ
delta as grid ViTs.

**Status:** open

**Decision:** 2026-09-11 — not tested in this study. No Raw/HQ/MQ/D20 re-benchmark. MQ-only timm scores are not a Table 1 analogue. Status stays `open` (observational; not falsified by omission).

---

## H14 — Default 224 px cell windows are not cell-scale

**Claim.** Campaign cell crops inherit the paper grid tile (`crop_size=224`)
unless `--crop-size` is set. A 224 px centroid window is large versus a JUMP
cell: extra neighbor/context enters the crop, and `cell_fixed` drops any
window that leaves the FOV. Grid `tile_size` (H3, H9) does not test this.
On EfficientNet-B0, grid `tile_size` already *is* the native window because
convs do not resize; that is a different axis from a cell-centered window.

**Why it would bias the ranking.** A cell-vs-grid comparison at 224 px can
fail because the “cell” crop is still a large patch, not because cell
inductive bias is absent. Edge-cell dropout also changes which instances
enter the well profile. Wave 3 CELL/BBOX would then score the wrong window.

**What the paper / JUMP_lite / this repo actually did.** Paper embeddings use
Aliby `kind: crop` at model `tile_size` (224 / 256 / 448), not instance
windows. This repo: `--crop-size` is independent of `tile_size`
(`src/jumpbench/embed/generate.py`); `cell_fixed` skips out-of-FOV windows
(`src/jumpbench/embed/crops.py`). Campaign B0 and Wave 3 CELL default to 224.

**Testability:** in-scope. 4-site `cell_fixed` only. Grid native-window
sweep stays T256/T448 (H3, H9).

**Planned test.** On Run1 4-site CRISPR, `--crop cell_fixed` at **96** and
**128** (Wave 2.5) vs CELL@224 (Wave 3) and vs B0 grid. Same channel recipe,
PCA/TVN, CRISPR PA. Record `n_objects_skipped_edge` and cells kept in
provenance. If C96 or C128 beats CELL@224 by \|ΔNAP\| ≥ 0.03, Wave 3
CELL/BBOX inherit that `crop_size`.

**Falsifier.** Reachable: CRISPR PA is stable across 96 / 128 / 224
(`|ΔNAP| < 0.03` vs CELL@224). Then window size is not first-order *for
timm cell crops*. Grid remainder/tile size stays H3/H9.

**Status:** open

**Decision:**
