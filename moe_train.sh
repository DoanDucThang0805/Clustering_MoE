#!/bin/bash

# MoE Training Script (full CLI)
# Usage: ./moe_train.sh [--option value] [--flag]

set -euo pipefail

# Defaults (match src/training/moe.py)
SEED=42
TYPE_MODEL="moe_temperature_0.5"
NUM_EXPERTS=4
TOP_K=2
NUM_EPOCHS=400
BATCH_SIZE=32
LR=0.001
WEIGHT_DECAY=0.001
MOE_ALPHA=0.05
TEMPERATURE=0.5
ROUTER_MODE="context_aware"
CONTEXT_DIM=6
USE_CONTEXT=true
PRETRAIN_BACKBONE=false
BACKBONE_NAME="mobilenetv3small_torchvision"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --seed N                        Random seed (default: ${SEED})
  --type_model NAME               Type model (default: ${TYPE_MODEL})
  --num_experts N                 Number of experts (default: ${NUM_EXPERTS})
  --top_k N                       Top-k experts (default: ${TOP_K})
  --num_epochs N                  Number of epochs (default: ${NUM_EPOCHS})
  --batch_size N                  Batch size (default: ${BATCH_SIZE})
  --lr F                          Learning rate (default: ${LR})
  --weight_decay F                Weight decay (default: ${WEIGHT_DECAY})
  --moe_alpha F                   MoE auxiliary loss weight (default: ${MOE_ALPHA})
  --temperature F                 Temperature for gating (default: ${TEMPERATURE})
  --router_mode MODE              Router mode: noisy|context_aware (default: ${ROUTER_MODE})
  --context_dim N                 Context feature dim (default: ${CONTEXT_DIM})
  --use_context                   Use context features (default: ${USE_CONTEXT})
  --no_context                    Disable context features
  --backbone_name NAME            Backbone name (default: ${BACKBONE_NAME})
  --pretrain_backbone             (flag) set pretrain_backbone to false in python args
  --no-pretrain_backbone          keep pretrain_backbone true (default behavior)
  -h, --help                      Show this help

Examples:
  $0 --num_experts 8 --backbone_name mobilenetv3small_timm --no_context --num_epochs 100
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2;;
        --type_model) TYPE_MODEL="$2"; shift 2;;
        --num_experts) NUM_EXPERTS="$2"; shift 2;;
        --top_k|--top-k) TOP_K="$2"; shift 2;;
        --num_epochs) NUM_EPOCHS="$2"; shift 2;;
        --batch_size) BATCH_SIZE="$2"; shift 2;;
        --lr) LR="$2"; shift 2;;
        --weight_decay) WEIGHT_DECAY="$2"; shift 2;;
        --moe_alpha) MOE_ALPHA="$2"; shift 2;;
        --temperature) TEMPERATURE="$2"; shift 2;;
        --router_mode) ROUTER_MODE="$2"; shift 2;;
        --context_dim) CONTEXT_DIM="$2"; shift 2;;
        --use_context) USE_CONTEXT=true; shift;;
        --no_context) USE_CONTEXT=false; shift;;
        --backbone_name) BACKBONE_NAME="$2"; shift 2;;
        --pretrain_backbone) PRETRAIN_BACKBONE=false; shift;;
        --no-pretrain_backbone) PRETRAIN_BACKBONE=true; shift;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage; exit 1;;
    esac
done

# Activate venv if available
if [ -d "venv" ]; then
    # shellcheck source=/dev/null
    source venv/bin/activate
fi

echo "========================================"
echo "  Starting MoE Training"
echo "========================================"
echo "seed:               $SEED"
echo "type_model:         $TYPE_MODEL"
echo "num_experts:        $NUM_EXPERTS"
echo "top_k:              $TOP_K"
echo "num_epochs:         $NUM_EPOCHS"
echo "batch_size:         $BATCH_SIZE"
echo "lr:                 $LR"
echo "weight_decay:       $WEIGHT_DECAY"
echo "moe_alpha:          $MOE_ALPHA"
echo "temperature:        $TEMPERATURE"
echo "router_mode:        $ROUTER_MODE"
echo "context_dim:        $CONTEXT_DIM"
echo "use_context:        $USE_CONTEXT"
echo "pretrain_backbone:  $PRETRAIN_BACKBONE"
echo "backbone_name:      $BACKBONE_NAME"
echo "========================================"

cd src

python -m training.moe \
    --seed "$SEED" \
    --type_model "$TYPE_MODEL" \
    --num_experts "$NUM_EXPERTS" \
    --top_k "$TOP_K" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --moe_alpha "$MOE_ALPHA" \
    --temperature "$TEMPERATURE" \
    --router_mode "$ROUTER_MODE" \
    --context_dim "$CONTEXT_DIM" \
    $( [ "$USE_CONTEXT" = false ] && echo "--no_context" ) \
    --backbone_name "$BACKBONE_NAME" \
    $( [ "$PRETRAIN_BACKBONE" = false ] && echo "--pretrain_backbone" )

echo "========================================"
echo "  Training script finished (exit code $? )"
echo "========================================="
