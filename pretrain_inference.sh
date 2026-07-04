#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python"

# ─────────────────────────────────────────────
# Configuration — edit these values if needed
# ─────────────────────────────────────────────
DATASET_NAME="plantdoc"
TYPE_MODEL="pretrain_baseline"
MODEL_NAME="mobilenetv3small_torchvision"
SEED=49

# Example: RUN_TIME="run_20260701-140028"
# Leave empty to automatically use the newest run containing best_checkpoint.pth.
RUN_TIME="run_20260705-032101"

usage() {
    cat <<EOF
Usage: bash pretrain_inference.sh [options]

Options:
  --dataset_name NAME   Dataset name (default: $DATASET_NAME)
  --type_model TYPE     Checkpoint model namespace (default: $TYPE_MODEL)
  --model_name NAME     Model name (default: $MODEL_NAME)
  --seed N              Seed to evaluate (default: $SEED)
  --run_time RUN        Run directory (default: newest complete run)
  -h, --help            Show this help

Example:
  bash pretrain_inference.sh --seed 46 --run_time run_20260701-140028
EOF
}

require_value() {
    local option=$1
    local value=${2:-}

    if [[ -z "$value" || "$value" == --* ]]; then
        echo "[ERROR] $option requires a value." >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset_name|--dataset-name)
            require_value "$1" "${2:-}"
            DATASET_NAME="$2"
            shift 2
            ;;
        --type_model|--type-model)
            require_value "$1" "${2:-}"
            TYPE_MODEL="$2"
            shift 2
            ;;
        --model_name|--model-name)
            require_value "$1" "${2:-}"
            MODEL_NAME="$2"
            shift 2
            ;;
        --seed)
            require_value "$1" "${2:-}"
            SEED="$2"
            shift 2
            ;;
        --run_time|--run-time|--runtime)
            require_value "$1" "${2:-}"
            RUN_TIME="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ! -x "$PYTHON" ]]; then
    echo "[ERROR] Virtualenv Python not found: $PYTHON" >&2
    exit 1
fi

if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] seed must be a non-negative integer: $SEED" >&2
    exit 1
fi

SEED_DIR="$SCRIPT_DIR/checkpoints/$DATASET_NAME/$TYPE_MODEL/$MODEL_NAME/seed_$SEED"

if [[ -z "$RUN_TIME" ]]; then
    for candidate in "$SEED_DIR"/run_*/best_checkpoint.pth; do
        if [[ -f "$candidate" ]]; then
            RUN_TIME="$(basename "$(dirname "$candidate")")"
        fi
    done
fi

if [[ -z "$RUN_TIME" ]]; then
    echo "[ERROR] No best checkpoint found under: $SEED_DIR" >&2
    exit 1
fi

CHECKPOINT_PATH="$SEED_DIR/$RUN_TIME/best_checkpoint.pth"
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
fi

echo "========================================"
echo "  Pretrained Baseline Inference"
echo "========================================"
echo "  Dataset    : $DATASET_NAME"
echo "  Type model : $TYPE_MODEL"
echo "  Model name : $MODEL_NAME"
echo "  Seed       : $SEED"
echo "  Run time   : $RUN_TIME"
echo "  Checkpoint : $CHECKPOINT_PATH"
echo "========================================"

cd "$SCRIPT_DIR/src"
"$PYTHON" -m inference.pretrain_baseline.inference \
    --dataset_name "$DATASET_NAME" \
    --type_model "$TYPE_MODEL" \
    --model_name "$MODEL_NAME" \
    --seed "$SEED" \
    --run_time "$RUN_TIME"

echo "========================================"
echo "  Inference completed"
echo "========================================"
