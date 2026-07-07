#!/bin/bash

# MoE Training Script (full CLI)
# CP1: learned-gate MoE với ImageNet-pretrained MobileNetV3-Small, chạy 10 seed.
# Usage: ./moe_train.sh [--option value] [--flag]

set -euo pipefail

# Defaults (match src/training/moe.py)
SEEDS=(42 43 44 45 46 47 48 49 50 51)
TYPE_MODEL="moe_temperature_0.5_pretrain_backbone"
NUM_EXPERTS=4
TOP_K=2
NUM_EPOCHS=400
BATCH_SIZE=64
LR=0.001
WEIGHT_DECAY=0.001
MOE_ALPHA=0.05
TEMPERATURE=0.5
ROUTER_MODE="context_aware"
CONTEXT_DIM=6
USE_CONTEXT=true
PRETRAIN_BACKBONE=true
BACKBONE_NAME="mobilenetv3small_torchvision"
FORCE_RETRAIN=false

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

usage() {
    cat <<EOF
Usage: $0 [options]

CP1: learned-gate MoE với ImageNet-pretrained backbone, G=4, top-k=2, tau=0.5.
Mặc định chạy 10 seed: ${SEEDS[*]}

Options:
  --seed N                        Chỉ chạy 1 seed (ghi đè danh sách mặc định).
  --seeds "A B C"                 Danh sách seed tùy chỉnh.
  --force                         Train lại kể cả khi checkpoint đã tồn tại.
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
  $0                              # CP1: 10 seed pretrained learned-gate MoE
  $0 --seed 47                    # chỉ seed 47
  $0 --seeds "42 43 44 45 46"     # pilot 5 seed
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEEDS=("$2"); shift 2;;
        --seeds) read -r -a SEEDS <<< "$2"; shift 2;;
        --force) FORCE_RETRAIN=true; shift;;
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

# Đã có checkpoint hoàn chỉnh cho seed này chưa?
is_done() {
    local seed=$1
    local seed_dir="$SCRIPT_DIR/checkpoints/plantdoc/$TYPE_MODEL/${BACKBONE_NAME}_moe/${NUM_EXPERTS}_experts/top_${TOP_K}/seed_${seed}"
    local run_dir

    [[ "$FORCE_RETRAIN" == true ]] && return 1

    for run_dir in "$seed_dir"/run_*; do
        [ -f "$run_dir/best_checkpoint.pth" ] && return 0
    done
    return 1
}

train_one() {
    local seed=$1

    echo "========================================"
    echo "  Starting MoE Training"
    echo "========================================"
    echo "seed:               $seed"
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

    ( cd src && python -m training.moe \
        --seed "$seed" \
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
        $( [ "$PRETRAIN_BACKBONE" = false ] && echo "--pretrain_backbone" ) )
}

# ─────────────────────────────────────────────
# Chạy toàn bộ seed
# ─────────────────────────────────────────────
total=${#SEEDS[@]}
num_done=0
current=0

echo "========================================"
echo "  CP1 — Learned-gate MoE (pretrained)"
echo "========================================"
echo "  Seeds        : ${SEEDS[*]}"
echo "  Type model   : $TYPE_MODEL"
echo "  Force retrain: $FORCE_RETRAIN"
echo "  Total seeds  : $total"
echo "========================================"

for seed in "${SEEDS[@]}"; do
    current=$(( current + 1 ))
    if is_done "$seed"; then
        echo ""
        echo "  [SKIP $current/$total] seed=$seed đã có checkpoint."
        continue
    fi

    echo ""
    echo "  [RUN $current/$total] seed=$seed"
    train_one "$seed"
    num_done=$(( num_done + 1 ))
    echo "  ✓ Done seed=$seed"
done

echo ""
echo "========================================"
echo "  All seeds finished"
echo "  Ran     : $num_done"
echo "  Skipped : $(( total - num_done ))"
echo "========================================="
