#!/bin/bash
# =============================================================================
# Extract embeddings with src/embedding/pretrain_backbone/image_embedding.py.
#
# Examples:
#   bash extract_embedding.sh
#   bash extract_embedding.sh --split train --seed 42
#   bash extract_embedding.sh --seed 43 --run_time run_20260629-103117
#   bash extract_embedding.sh --weights_source imagenet --split train --seed 42
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python"

# ─────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────
DATASET_NAME="plantdoc"
MODEL_NAME="mobilenetv3small_torchvision"
TYPE_MODEL="pretrain_baseline"
TYPE_BACKBONE="pretrain_backbone"
WEIGHTS_SOURCE="checkpoint"
RUN_TIME="run_20260701-151815"
SPLIT="all"
SEED=43
BATCH_SIZE=64
NUM_WORKERS=4

usage() {
    sed -n '2,9p' "$0"
}

require_value() {
    local option=$1
    local value=${2:-}

    if [[ -z "$value" || "$value" == --* ]]; then
        echo "[ERROR] $option requires a value."
        exit 1
    fi
}

# ─────────────────────────────────────────────
# Parse CLI overrides
# ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset_name)
            require_value "$1" "${2:-}"
            DATASET_NAME="$2"
            shift 2
            ;;
        --model_name)
            require_value "$1" "${2:-}"
            MODEL_NAME="$2"
            shift 2
            ;;
        --type_model)
            require_value "$1" "${2:-}"
            TYPE_MODEL="$2"
            shift 2
            ;;
        --type_backbone)
            require_value "$1" "${2:-}"
            TYPE_BACKBONE="$2"
            shift 2
            ;;
        --weights_source)
            require_value "$1" "${2:-}"
            WEIGHTS_SOURCE="$2"
            shift 2
            ;;
        --run_time)
            require_value "$1" "${2:-}"
            RUN_TIME="$2"
            shift 2
            ;;
        --split)
            require_value "$1" "${2:-}"
            SPLIT="$2"
            shift 2
            ;;
        --seed)
            require_value "$1" "${2:-}"
            SEED="$2"
            shift 2
            ;;
        --batch_size)
            require_value "$1" "${2:-}"
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_workers)
            require_value "$1" "${2:-}"
            NUM_WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

# Keep embeddings from different weight sources in separate namespaces unless
# the caller explicitly supplies --type_backbone.
if [[ -z "$TYPE_BACKBONE" ]]; then
    if [[ "$WEIGHTS_SOURCE" == "checkpoint" ]]; then
        TYPE_BACKBONE="pretrain_backbone"
    else
        TYPE_BACKBONE="imagenet_initialization_backbone"
    fi
fi

case "$WEIGHTS_SOURCE" in
    checkpoint|imagenet) ;;
    *)
        echo "[ERROR] weights_source must be 'checkpoint' or 'imagenet'."
        exit 1
        ;;
esac

case "$SPLIT" in
    train|validation|test|all) ;;
    *)
        echo "[ERROR] split must be train, validation, test, or all."
        exit 1
        ;;
esac

if [[ ! -x "$PYTHON" ]]; then
    echo "[ERROR] Virtualenv Python not found: $PYTHON"
    exit 1
fi

EXTRA_ARGS=()
if [[ "$WEIGHTS_SOURCE" == "checkpoint" ]]; then
    if [[ -z "$TYPE_MODEL" ]]; then
        echo "[ERROR] type_model is required for checkpoint weights."
        exit 1
    fi

    if [[ -z "$RUN_TIME" ]]; then
        for candidate in \
            "$SCRIPT_DIR/checkpoints/$DATASET_NAME/$TYPE_MODEL/$MODEL_NAME/seed_$SEED"/run_*/best_checkpoint.pth; do
            if [[ -f "$candidate" ]]; then
                RUN_TIME="$(basename "$(dirname "$candidate")")"
            fi
        done
    fi

    if [[ -z "$RUN_TIME" ]]; then
        echo "[ERROR] No checkpoint run found for seed $SEED."
        exit 1
    fi

    CHECKPOINT_PATH="$SCRIPT_DIR/checkpoints/$DATASET_NAME/$TYPE_MODEL/$MODEL_NAME/seed_$SEED/$RUN_TIME/best_checkpoint.pth"
    if [[ ! -f "$CHECKPOINT_PATH" ]]; then
        echo "[ERROR] Checkpoint not found: $CHECKPOINT_PATH"
        exit 1
    fi

    EXTRA_ARGS=(
        --type_model "$TYPE_MODEL"
        --run_time "$RUN_TIME"
    )
fi

# ─────────────────────────────────────────────
# Print config
# ─────────────────────────────────────────────
echo "============================================="
echo "  Extract Embeddings"
echo "============================================="
echo "  dataset_name   : $DATASET_NAME"
echo "  model_name     : $MODEL_NAME"
echo "  weights_source : $WEIGHTS_SOURCE"
if [[ "$WEIGHTS_SOURCE" == "checkpoint" ]]; then
    echo "  type_model     : $TYPE_MODEL"
    echo "  run_time       : $RUN_TIME"
else
    echo "  type_model     : N/A"
    echo "  run_time       : N/A"
fi
echo "  type_backbone  : $TYPE_BACKBONE"
echo "  split          : $SPLIT"
echo "  seed           : $SEED"
echo "  batch_size     : $BATCH_SIZE"
echo "  num_workers    : $NUM_WORKERS"
echo "============================================="

# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
cd "$SCRIPT_DIR/src"
"$PYTHON" -m embedding.pretrain_backbone.image_embedding \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --type_backbone "$TYPE_BACKBONE" \
    --weights_source "$WEIGHTS_SOURCE" \
    --split "$SPLIT" \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    "${EXTRA_ARGS[@]}"

echo
echo "[Done] Embeddings saved."
