#!/bin/bash
# =============================================================================
# extract_embeddings.sh
# Chạy trích xuất embeddings từ trained backbone.
#
# Usage:
#   bash extract_embeddings.sh                        # dùng giá trị mặc định
#   bash extract_embeddings.sh --split train          # chỉ chạy train split
#   bash extract_embeddings.sh --seed 0 --run_time 20240601_120000
# =============================================================================

set -e  # dừng ngay nếu có lỗi

# ─────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────
DATASET_NAME="plantdoc"
MODEL_NAME="mobilenetv3small_torchvision"   # hoặc mobilenetv3small_timm
TYPE_MODEL="pretrain_baseline"                 # hoặc pretrain_baseline
TYPE_BACKBONE="pretrain_backbone"
RUN_TIME="run_20260629-101639"                           # tên folder timestamp trong seed dir
SPLIT="all"                                 # train | validation | test | all
SEED=42
BATCH_SIZE=64
NUM_WORKERS=4

# ─────────────────────────────────────────────
# Parse CLI overrides  (key=value style)
# ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset_name) DATASET_NAME="$2"; shift 2 ;;
        --model_name)   MODEL_NAME="$2";   shift 2 ;;
        --type_model)   TYPE_MODEL="$2";   shift 2 ;;
        --run_time)     RUN_TIME="$2";     shift 2 ;;
        --split)        SPLIT="$2";        shift 2 ;;
        --seed)         SEED="$2";         shift 2 ;;
        --batch_size)   BATCH_SIZE="$2";   shift 2 ;;
        --num_workers)  NUM_WORKERS="$2";  shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

# ─────────────────────────────────────────────
# Print config
# ─────────────────────────────────────────────
echo "============================================="
echo "  Extract Embeddings"
echo "============================================="
echo "  dataset_name : $DATASET_NAME"
echo "  model_name   : $MODEL_NAME"
echo "  type_model   : $TYPE_MODEL"
echo "  run_time     : $RUN_TIME"
echo "  split        : $SPLIT"
echo "  seed         : $SEED"
echo "  batch_size   : $BATCH_SIZE"
echo "  num_workers  : $NUM_WORKERS"
echo "============================================="

# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
cd src
python -m embedding.pretrain_backbone.image_embedding \
    --dataset_name "$DATASET_NAME" \
    --model_name   "$MODEL_NAME"   \
    --type_model   "$TYPE_MODEL"   \
    --type_backbone "$TYPE_BACKBONE" \
    --run_time     "$RUN_TIME"     \
    --split        "$SPLIT"        \
    --seed         "$SEED"         \
    --batch_size   "$BATCH_SIZE"   \
    --num_workers  "$NUM_WORKERS"

echo ""
echo "[Done] Embeddings saved."