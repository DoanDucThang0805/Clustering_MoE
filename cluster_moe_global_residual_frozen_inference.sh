#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python"

DATASET_NAME="plantdoc"
BACKBONE_TYPE="dense_global_residual_frozen_pretrain_backbone"
CENTROID_BACKBONE_TYPE="pretrain_backbone"
BACKBONE_NAME="mobilenetv3small_torchvision"
MODEL_CLUSTERING_NAME="kmeans"
SEED=46
NUM_EXPERTS=4
TOP_K=2
METRIC="cosine"
TEMPERATURE=0.5
BATCH_SIZE=32
CHECKPOINT="best_checkpoint.pth"
RUN_TIME=""

SEED_DIR="$SCRIPT_DIR/checkpoints/$DATASET_NAME/clustering_moe/$BACKBONE_TYPE/${BACKBONE_NAME}_backbone/$MODEL_CLUSTERING_NAME/temperature_$TEMPERATURE/G${NUM_EXPERTS}_${METRIC}_top${TOP_K}/seed_$SEED"

if [[ ! -x "$PYTHON" ]]; then
    echo "[ERROR] Virtualenv Python not found: $PYTHON" >&2
    exit 1
fi

if [[ -z "$RUN_TIME" ]]; then
    for candidate in "$SEED_DIR"/run_*/"$CHECKPOINT"; do
        if [[ -f "$candidate" ]]; then
            RUN_TIME="$(basename "$(dirname "$candidate")")"
        fi
    done
fi

if [[ -z "$RUN_TIME" ]]; then
    echo "[ERROR] No $CHECKPOINT found under: $SEED_DIR" >&2
    exit 1
fi

CHECKPOINT_PATH="$SEED_DIR/$RUN_TIME/$CHECKPOINT"
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
fi

echo "=================================================="
echo " Frozen Dense-global Residual Cluster-MoE Inference"
echo "=================================================="
echo " Seed       : $SEED"
echo " Runtime    : $RUN_TIME"
echo " Checkpoint : $CHECKPOINT_PATH"
echo "=================================================="

cd "$SCRIPT_DIR/src"
"$PYTHON" -m inference.cluster_moe_global_residual.inference \
    --seed "$SEED" \
    --num_experts "$NUM_EXPERTS" \
    --top_k "$TOP_K" \
    --metric "$METRIC" \
    --temperature "$TEMPERATURE" \
    --pretrain_backbone \
    --batch_size "$BATCH_SIZE" \
    --dataset_name "$DATASET_NAME" \
    --backbone_type "$BACKBONE_TYPE" \
    --centroid_backbone_type "$CENTROID_BACKBONE_TYPE" \
    --backbone_name "$BACKBONE_NAME" \
    --model_clustering_name "$MODEL_CLUSTERING_NAME" \
    --checkpoint "$CHECKPOINT" \
    --runtime "$RUN_TIME"
