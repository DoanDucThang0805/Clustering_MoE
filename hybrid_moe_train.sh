#!/bin/bash

set -e
clear

# ==========================================================
# Common Config
# ==========================================================
DATASET_NAME="plantdoc"
BACKBONE_TYPE="non_pretrain_backbone"
BACKBONE_NAME="mobilenetv3small_torchvision"
MODEL_CLUSTERING_NAME="kmeans"

NUM_EXPERTS=4
TOP_K=2
METRIC="cosine"
TEMPERATURE=0.5
PRETRAIN_BACKBONE=false

LR=1e-3
WEIGHT_DECAY=1e-3
LAMBDA=0.5
MOE_ALPHA=0.05

NUM_EPOCHS=400
BATCH_SIZE=64

# ==========================================================
# Paths
# ==========================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

# ==========================================================
# Train Function
# ==========================================================
train_seed() {
    local SEED=$1

    echo ""
    echo "========================================"
    echo " Training Seed = $SEED"
    echo "========================================"

    cd "$SCRIPT_DIR/src"

    python -m training.hybrid_moe \
        --seed "$SEED" \
        --num_experts "$NUM_EXPERTS" \
        --top_k "$TOP_K" \
        --distance_metric "$METRIC" \
        --temperature "$TEMPERATURE" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --num_epochs "$NUM_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --dataset_name "$DATASET_NAME" \
        --backbone_type "$BACKBONE_TYPE" \
        --backbone_name "$BACKBONE_NAME" \
        --model_clustering_name "$MODEL_CLUSTERING_NAME" \
        --lambda_ "$LAMBDA" \
        --moe_alpha "$MOE_ALPHA" \
        $([ "$PRETRAIN_BACKBONE" = true ] && echo "--pretrain_backbone")

    cd "$SCRIPT_DIR"

    echo "Finished seed $SEED"
}

# ==========================================================
# Run
# ==========================================================

# Train 1 seed
# train_seed 42

# Train multiple seeds
for seed in 43 44 45 46
do
    train_seed $seed
done

echo ""
echo "========================================"
echo " All Training Completed"
echo "========================================"