#!/bin/bash

# Cluster MoE Training Script
# This script trains the ClusteringMoEModel with specified parameters

set -e
clear
# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# Model and Data Configuration
DATASET_NAME="plantdoc"
TYPE_MODEL="non_pretrain_models"
BACKBONE_NAME="mobilenetv3small_torchvision"
MODEL_NAME="kmeans"

# Training Hyperparameters
SEED=42
NUM_EXPERTS=4
TOP_K=2
METRIC="cosine"  # choices: ["cosine", "euclidean"]
TEMPERATURE=0.5
PRETRAIN_BACKBONE=false

# Optimizer Configuration
LR=1e-3
WEIGHT_DECAY=1e-3

# Training Settings
NUM_EPOCHS=300
BATCH_SIZE=32

# ─────────────────────────────────────────────
# Get Script Directory
# ─────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ─────────────────────────────────────────────
# Activate Virtual Environment
# ─────────────────────────────────────────────
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# ─────────────────────────────────────────────
# Run Training
# ─────────────────────────────────────────────
echo "========================================"
echo "  Starting ClusteringMoE Training"
echo "========================================"
echo "Dataset:        $DATASET_NAME"
echo "Type Model:     $TYPE_MODEL"
echo "Backbone:       $BACKBONE_NAME"
echo "Model:          $MODEL_NAME"
echo "Seed:           $SEED"
echo "Num Experts:    $NUM_EXPERTS"
echo "Top K:          $TOP_K"
echo "Metric:         $METRIC"
echo "Temperature:    $TEMPERATURE"
echo "Learning Rate:  $LR"
echo "Batch Size:     $BATCH_SIZE"
echo "Epochs:         $NUM_EPOCHS"
echo "========================================"

cd src
python -m training.clustering_moe \
    --seed "$SEED" \
    --num_experts "$NUM_EXPERTS" \
    --top_k "$TOP_K" \
    --metric "$METRIC" \
    --temperature "$TEMPERATURE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --dataset_name "$DATASET_NAME" \
    --type_model "$TYPE_MODEL" \
    --backbone_name "$BACKBONE_NAME" \
    --model_name "$MODEL_NAME" \
    $([ "$PRETRAIN_BACKBONE" = true ] && echo "--pretrain_backbone")

echo "========================================"
echo "  Training Completed"
echo "========================================"
