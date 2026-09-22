#!/usr/bin/env bash
# Sequential: XL cell-224 Raw, XL grid-96 Raw, then DL process sweeps for
# any Run1 timm embedding that does not already have 420 CRISPR PA shards.
# Resume-safe (embed completed_sites.txt; sweep skips existing JSON).
set -euo pipefail
export PATH="/home/francesco/jumpbench/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
cd /home/francesco/jumpbench

ROOT=data/embeddings/timm/run1
CELL96=$ROOT/cell_fixed_jump_lite_96_efficientnetv2_xl_5ch_raw
KEEP=$CELL96
SWEEP_GRID=sweep_paper_dl_v11_lite
SWEEP_PRESET=paper_dl_default
SWEEP_JOBS="${SWEEP_JOBS:-16}"
N_SWEEP=420
LOG=$ROOT/xl_c224_g96_sweep_chain.log
mkdir -p "$ROOT"
exec >>"$LOG" 2>&1

log() { echo "===== $(date -Is) $* ====="; }

stem_for() {
  case "$1" in
    cell_fixed_jump_lite_96_efficientnetv2_xl_5ch_raw) echo timm_run1_xl_c96_raw ;;
    cell_fixed_jump_lite_96_efficientnetv2_xl_5ch_jpegxl_mq) echo timm_run1_xl_c96_mq ;;
    cell_fixed_jump_lite_224_efficientnetv2_xl_5ch_raw) echo timm_run1_xl_c224_raw ;;
    grid_jump_lite_224_efficientnetv2_xl_5ch_raw) echo timm_run1_xl_g224_raw ;;
    grid_jump_lite_224_efficientnetv2_xl_5ch_jpegxl_mq) echo timm_run1_xl_g224_mq ;;
    grid_jump_lite_96_efficientnetv2_xl_5ch_raw) echo timm_run1_xl_g96_raw ;;
    grid_jump_lite_224_vit_small_dinov2_5ch_raw) echo timm_run1_vit_g224_raw ;;
    grid_jump_lite_224_vit_small_dinov2_5ch_jpegxl_mq) echo timm_run1_vit_g224_mq ;;
    *) echo "timm_run1_$1" ;;
  esac
}

n_sweep_json() {
  local dir="$1"
  [[ -d "$dir" ]] || { echo 0; return; }
  find "$dir" -maxdepth 1 -type f -name '*.json' ! -name 'summary.json' ! -name 'gather.json' | wc -l
}

embed_xl() {
  log "embed $*"
  jumpbench embed --model timm --image-source s3 \
    --subset crispr --sites jump_lite --batch 20220914_Run1 \
    --pool site \
    --set models.timm.architecture=tf_efficientnetv2_xl.in21k \
    "$@"
  log "embed done"
}

ensure_profile() {
  local rundir="$1" stem="$2"
  local parquet="data/profiles/${stem}.parquet"
  local sites="$ROOT/$rundir/site_embeddings.parquet"
  if [[ -f "$parquet" ]]; then
    log "profile exists $parquet"
    return
  fi
  log "aggregate $rundir -> $stem"
  if [[ -d "$KEEP" ]]; then
    jumpbench aggregate --input "$sites" --keep-sites-from "$KEEP" --output "$parquet"
  else
    jumpbench aggregate --input "$sites" --output "$parquet"
  fi
}

sweep_if_needed() {
  local rundir="$1"
  local sites="$ROOT/$rundir/site_embeddings.parquet"
  [[ -f "$sites" ]] || { log "skip $rundir (no site_embeddings.parquet)"; return; }
  local stem results n
  stem=$(stem_for "$rundir")
  results="data/results/campaign/${stem}_dl_sweep"
  n=$(n_sweep_json "$results")
  if [[ "$n" -ge "$N_SWEEP" ]]; then
    if [[ ! -f "$results/summary.csv" ]]; then
      log "gather $stem ($n shards, no summary)"
      jumpbench sweep gather --results-dir "$results"
    else
      log "skip sweep $stem (already $n/$N_SWEEP)"
    fi
    return
  fi
  ensure_profile "$rundir" "$stem"
  log "sweep $stem ($n/$N_SWEEP done)"
  jumpbench sweep run --grid "$SWEEP_GRID" --preset "$SWEEP_PRESET" \
    --input "data/profiles/${stem}.parquet" \
    --processed-dir "data/processed/${stem}_dl_sweep" \
    --results-dir "$results" \
    --jobs "$SWEEP_JOBS" --subset crispr --tasks pa
  jumpbench sweep gather --results-dir "$results"
  log "sweep done $stem"
}

log "chain start SWEEP_JOBS=$SWEEP_JOBS"

# cell-224 Raw (Wave 3 CELL@224). "cell-244" in the request is this arm.
embed_xl --crop cell_fixed --crop-size 224 \
  --run-dir $ROOT/cell_fixed_jump_lite_224_efficientnetv2_xl_5ch_raw \
  --set runtime.batch_size=4

# grid-96 Raw (96 px tiles; unconfounds H1/H14 vs cell-96 / grid-224).
embed_xl --crop grid \
  --run-dir $ROOT/grid_jump_lite_96_efficientnetv2_xl_5ch_raw \
  --set models.timm.tile_size=96 \
  --set runtime.batch_size=64

log "sweeps for Run1 embeddings missing a full $N_SWEEP-config DL sweep"
shopt -s nullglob
for parquet in "$ROOT"/*/site_embeddings.parquet; do
  rundir=$(basename "$(dirname "$parquet")")
  sweep_if_needed "$rundir"
done

log "ALL EMBEDS AND SWEEPS DONE"
