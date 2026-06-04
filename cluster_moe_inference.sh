#!/bin/bash

# Cluster MoE Inference Script
# This script runs inference on the ClusteringMoEModel with specified parameters

set -e

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# Model and Data Configuration
DATASET_NAME="plantdoc"
TYPE_MODEL="non_pretrain_models"
BACKBONE_NAME="mobilenetv3small_torchvision"
MODEL_NAME="kmeans"

# Model Hyperparameters
SEED=42
NUM_EXPERTS=4
TOP_K=2
METRIC="cosine"  # choices: ["cosine", "euclidean"]
TEMPERATURE=0.5
PRETRAIN_BACKBONE=false

# Inference Settings
BATCH_SIZE=64
CHECKPOINT="best_checkpoint.pth"  # choices: ["best_checkpoint.pth", "last_checkpoint.pth"]
RUN_TIME="run_20260604-153316"

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
# Run Inference
# ─────────────────────────────────────────────
echo "========================================"
echo "  Starting ClusteringMoE Inference"
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
echo "Batch Size:     $BATCH_SIZE"
echo "Checkpoint:     $CHECKPOINT"
echo "========================================"

cd src
python -m inference.cluster_moe_models.inference \
    --seed "$SEED" \
    --num_experts "$NUM_EXPERTS" \
    --top_k "$TOP_K" \
    --metric "$METRIC" \
    --temperature "$TEMPERATURE" \
    --batch_size "$BATCH_SIZE" \
    --dataset_name "$DATASET_NAME" \
    --type_model "$TYPE_MODEL" \
    --backbone_name "$BACKBONE_NAME" \
    --model_name "$MODEL_NAME" \
    --checkpoint "$CHECKPOINT" \
    --run_time "$RUN_TIME" \
    $([ "$PRETRAIN_BACKBONE" = true ] && echo "--pretrain_backbone")

echo "========================================"
echo "  Inference Completed"
echo "========================================"
