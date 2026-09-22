#!/usr/bin/env bash
# Sequential grid embeds (XL Raw/MQ, ViT Raw/MQ). Resume-safe via completed_sites.txt.
set -euo pipefail
export PATH="/home/francesco/jumpbench/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
cd /home/francesco/jumpbench

log=data/embeddings/timm/run1/grid_embed_chain.log
mkdir -p data/embeddings/timm/run1
exec >>"$log" 2>&1

run() {
  echo "===== $(date -Is) $* ====="
  jumpbench embed "$@"
  echo "===== $(date -Is) done ====="
}

common=(
  --model timm
  --subset crispr --sites jump_lite --batch 20220914_Run1
  --crop grid --pool site
)

run "${common[@]}" --image-source s3 \
  --run-dir data/embeddings/timm/run1/grid_jump_lite_224_efficientnetv2_xl_5ch_raw \
  --set models.timm.architecture=tf_efficientnetv2_xl.in21k \
  --set runtime.batch_size=4

run "${common[@]}" --image-source s3_mq \
  --run-dir data/embeddings/timm/run1/grid_jump_lite_224_efficientnetv2_xl_5ch_jpegxl_mq \
  --set models.timm.architecture=tf_efficientnetv2_xl.in21k \
  --set runtime.batch_size=4

run "${common[@]}" --image-source s3 \
  --run-dir data/embeddings/timm/run1/grid_jump_lite_224_vit_small_dinov2_5ch_raw \
  --set models.timm.architecture=vit_small_patch14_dinov2.lvd142m \
  --set runtime.batch_size=8

run "${common[@]}" --image-source s3_mq \
  --run-dir data/embeddings/timm/run1/grid_jump_lite_224_vit_small_dinov2_5ch_jpegxl_mq \
  --set models.timm.architecture=vit_small_patch14_dinov2.lvd142m \
  --set runtime.batch_size=8

echo "===== $(date -Is) ALL GRID EMBEDS DONE ====="
