#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python"

DATASET_NAME="plantdoc"
BACKBONE_TYPE="dense_global_residual_frozen_run140028"
CENTROID_BACKBONE_TYPE="pretrain_backbone_run140028"
BACKBONE_NAME="mobilenetv3small_torchvision"
MODEL_CLUSTERING_NAME="kmeans"
SEED=46
NUM_EXPERTS=4
TOP_K=2
METRIC="cosine"
TEMPERATURE=0.5
LR=3e-4
WEIGHT_DECAY=1e-2
LABEL_SMOOTHING=0.05
NUM_EPOCHS=400
BATCH_SIZE=32
BACKBONE_CHECKPOINT="$SCRIPT_DIR/checkpoints/plantdoc/pretrain_baseline/mobilenetv3small_torchvision/seed_46/run_20260701-140028/best_checkpoint.pth"

CENTROID_PATH="$SCRIPT_DIR/clustering_results/$DATASET_NAME/$CENTROID_BACKBONE_TYPE/${BACKBONE_NAME}_backbone/$MODEL_CLUSTERING_NAME/$METRIC/seed_$SEED/clusters_kmeans_G${NUM_EXPERTS}_seed${SEED}.npz"

if [[ ! -x "$PYTHON" ]]; then
    echo "[ERROR] Virtualenv Python not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$BACKBONE_CHECKPOINT" ]]; then
    echo "[ERROR] Dense checkpoint not found: $BACKBONE_CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "$CENTROID_PATH" ]]; then
    echo "[ERROR] Centroid not found: $CENTROID_PATH" >&2
    exit 1
fi
if ! "$PYTHON" -c \
    'import sys, torch; print(f"CUDA: {torch.cuda.is_available()}"); sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "[ERROR] CUDA is required." >&2
    exit 1
fi

echo "=============================================="
echo " Frozen Dense-global Residual Cluster-MoE Train"
echo "=============================================="
echo " Seed          : $SEED"
echo " Backbone ckpt : $BACKBONE_CHECKPOINT"
echo " Centroid      : $CENTROID_PATH"
echo " LR            : $LR"
echo " Output type   : $BACKBONE_TYPE"
echo "=============================================="

cd "$SCRIPT_DIR/src"
"$PYTHON" -m training.clustering_moe_global_residual \
    --seed "$SEED" \
    --num_experts "$NUM_EXPERTS" \
    --top_k "$TOP_K" \
    --distance_metric "$METRIC" \
    --temperature "$TEMPERATURE" \
    --pretrain_backbone \
    --freeze_dense_branch \
    --backbone_checkpoint "$BACKBONE_CHECKPOINT" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --label_smoothing "$LABEL_SMOOTHING" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --dataset_name "$DATASET_NAME" \
    --backbone_type "$BACKBONE_TYPE" \
    --centroid_backbone_type "$CENTROID_BACKBONE_TYPE" \
    --backbone_name "$BACKBONE_NAME" \
    --model_clustering_name "$MODEL_CLUSTERING_NAME"
