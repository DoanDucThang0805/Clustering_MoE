#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─────────────────────────────────────────────
# Fixed Configuration
# ─────────────────────────────────────────────
DATASET_NAME="plantdoc"
BACKBONE_TYPE="dense_aligned_pretrain_backbone"
CENTROID_BACKBONE_TYPE="pretrain_backbone"
BACKBONE_NAME="mobilenetv3small_torchvision"
MODEL_CLUSTERING_NAME="kmeans"
METRIC="cosine"
TEMPERATURE=0.5
PRETRAIN_BACKBONE=true
LR=3e-4
WEIGHT_DECAY=1e-2
LABEL_SMOOTHING=0
NUM_EPOCHS=400
BATCH_SIZE=32
FORCE_RETRAIN=false
BACKBONE_CHECKPOINT="/media/data/minhht/clustering_moe/checkpoints/plantdoc/pretrain_baseline/mobilenetv3small_torchvision/seed_48/run_20260704-205832/best_checkpoint.pth"

# ─────────────────────────────────────────────
# Search Space
# ─────────────────────────────────────────────
SEEDS=(48)

# Format: "num_experts top_k"
CONFIGS=(
    # "2 1"
    # "2 2"
    # "3 1"
    # "3 2"
    # "4 1"
    "4 2"
    # "5 1"
    # "5 2"
    # "5 3"
    # "6 1"
    # "6 2"
    # "6 3"
    # "8 1"
    # "8 2"
    # "8 3"
    # "8 4"
)

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
usage() {
    echo "Usage: bash cluster_moe_train.sh [options]"
    echo "  --seed N  Run only the selected seed."
    echo "  --backbone_type TYPE           Checkpoint output namespace."
    echo "  --centroid_backbone_type TYPE  K-Means centroid namespace."
    echo "  --backbone_checkpoint PATH     Exact dense checkpoint used to initialize the backbone."
    echo "  --force                        Retrain even if complete checkpoints exist."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "[ERROR] --seed requires an integer value."
                exit 1
            fi
            SEEDS=("$2")
            shift 2
            ;;
        --force)
            FORCE_RETRAIN=true
            shift
            ;;
        --backbone_type|--backbone-type)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "[ERROR] $1 requires a value."
                exit 1
            fi
            BACKBONE_TYPE="$2"
            shift 2
            ;;
        --centroid_backbone_type|--centroid-backbone-type)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "[ERROR] $1 requires a value."
                exit 1
            fi
            CENTROID_BACKBONE_TYPE="$2"
            shift 2
            ;;
        --backbone_checkpoint|--backbone-checkpoint)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "[ERROR] $1 requires a path."
                exit 1
            fi
            BACKBONE_CHECKPOINT="$2"
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

# ─────────────────────────────────────────────
# Functions
# ─────────────────────────────────────────────

is_done() {
    local num_experts=$1
    local top_k=$2
    local seed=$3
    local seed_dir="$SCRIPT_DIR/checkpoints/$DATASET_NAME/clustering_moe/$BACKBONE_TYPE/${BACKBONE_NAME}_backbone/$MODEL_CLUSTERING_NAME/temperature_$TEMPERATURE/G${num_experts}_${METRIC}_top${top_k}/seed_${seed}"
    local run_dir

    if [[ "$FORCE_RETRAIN" == true ]]; then
        return 1
    fi

    for run_dir in "$seed_dir"/run_*; do
        if [ -f "$run_dir/best_checkpoint.pth" ] &&
           [ -f "$run_dir/last_checkpoint.pth" ]; then
            return 0
        fi
    done
    return 1
}

centroid_path() {
    local num_experts=$1
    local seed=$2

    echo "$SCRIPT_DIR/clustering_results/$DATASET_NAME/$CENTROID_BACKBONE_TYPE/${BACKBONE_NAME}_backbone/$MODEL_CLUSTERING_NAME/$METRIC/seed_${seed}/clusters_kmeans_G${num_experts}_seed${seed}.npz"
}

dense_checkpoint_path() {
    local seed=$1
    local seed_dir="$SCRIPT_DIR/checkpoints/$DATASET_NAME/pretrain_baseline/$BACKBONE_NAME/seed_${seed}"
    local candidate
    local selected=""

    if [[ -n "$BACKBONE_CHECKPOINT" ]]; then
        echo "$BACKBONE_CHECKPOINT"
        return
    fi

    for candidate in "$seed_dir"/run_*/best_checkpoint.pth; do
        if [[ -f "$candidate" ]]; then
            selected="$candidate"
        fi
    done

    echo "$selected"
}

train_one() {
    local num_experts=$1
    local top_k=$2
    local seed=$3
    local pretrained_args=()
    local dense_checkpoint

    echo ""
    echo "  seed=$seed  |  G=$num_experts  |  top_k=$top_k"
    echo "----------------------------------------"

    if [ "$PRETRAIN_BACKBONE" = true ]; then
        pretrained_args+=(--pretrain_backbone)
    fi

    dense_checkpoint=$(dense_checkpoint_path "$seed")

    python -m training.clustering_moe \
        --seed              "$seed"         \
        --num_experts       "$num_experts"  \
        --top_k             "$top_k"        \
        --distance_metric   "$METRIC"       \
        --temperature       "$TEMPERATURE"  \
        --backbone_checkpoint "$dense_checkpoint" \
        --lr                "$LR"           \
        --weight_decay      "$WEIGHT_DECAY" \
        --label_smoothing   "$LABEL_SMOOTHING" \
        --num_epochs        "$NUM_EPOCHS"   \
        --batch_size        "$BATCH_SIZE"   \
        --dataset_name      "$DATASET_NAME" \
        --backbone_type     "$BACKBONE_TYPE" \
        --centroid_backbone_type "$CENTROID_BACKBONE_TYPE" \
        --backbone_name     "$BACKBONE_NAME" \
        --model_clustering_name "$MODEL_CLUSTERING_NAME" \
        "${pretrained_args[@]}"

    echo "  ✓ Done: G=$num_experts top_k=$top_k seed=$seed"
}

preflight() {
    local config
    local num_experts
    local top_k
    local seed
    local centroid
    local dense_checkpoint

    if [ ! -f "$SCRIPT_DIR/venv/bin/activate" ]; then
        echo "[ERROR] Virtual environment not found: $SCRIPT_DIR/venv"
        exit 1
    fi

    source "$SCRIPT_DIR/venv/bin/activate"

    if ! python -c \
        'import sys, torch; print(f"CUDA: {torch.cuda.is_available()}"); sys.exit(0 if torch.cuda.is_available() else 1)'; then
        echo "[ERROR] CUDA is required for the 10-seed training batch."
        exit 1
    fi

    for config in "${CONFIGS[@]}"; do
        read -r num_experts top_k <<< "$config"
        for seed in "${SEEDS[@]}"; do
            centroid=$(centroid_path "$num_experts" "$seed")
            if [ ! -f "$centroid" ]; then
                echo "[ERROR] Missing centroid: $centroid"
                exit 1
            fi

            dense_checkpoint=$(dense_checkpoint_path "$seed")
            if [[ -z "$dense_checkpoint" || ! -f "$dense_checkpoint" ]]; then
                echo "[ERROR] Missing dense checkpoint for seed $seed"
                exit 1
            fi
        done
    done
}


run_all() {
    local total=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))
    local num_skipped=0
    local num_done=0

    # Precount skipped
    for config in "${CONFIGS[@]}"; do
        read -r num_experts top_k <<< "$config"
        for seed in "${SEEDS[@]}"; do
            if is_done "$num_experts" "$top_k" "$seed"; then
                num_skipped=$(( num_skipped + 1 ))
            fi
        done
    done

    local num_remaining=$(( total - num_skipped ))

    echo "========================================"
    echo "  ClusteringMoE — Sweep Training"
    echo "========================================"
    echo "  Total configs : ${#CONFIGS[@]}"
    echo "  Seeds         : ${SEEDS[*]}"
    echo "  Output type   : $BACKBONE_TYPE"
    echo "  Centroid type : $CENTROID_BACKBONE_TYPE"
    echo "  Backbone ckpt : ${BACKBONE_CHECKPOINT:-latest complete baseline run per seed}"
    echo "  LR (all model): $LR"
    echo "  Weight decay  : $WEIGHT_DECAY"
    echo "  Label smooth  : $LABEL_SMOOTHING"
    echo "  Total runs    : $total"
    echo "  Force retrain : $FORCE_RETRAIN"
    echo "  Already done  : $num_skipped"
    echo "  Remaining     : $num_remaining"
    echo "========================================"

    local current=0

    for config in "${CONFIGS[@]}"; do
        read -r num_experts top_k <<< "$config"
        for seed in "${SEEDS[@]}"; do

            if is_done "$num_experts" "$top_k" "$seed"; then
                echo ""
                echo "  [SKIP] G=$num_experts top_k=$top_k seed=$seed"
                continue
            fi

            current=$(( current + 1 ))
            echo ""
            echo "========================================"
            echo "  Run [$current / $num_remaining]"
            echo "========================================"

            train_one "$num_experts" "$top_k" "$seed"

        done
    done

    echo ""
    echo "========================================"
    echo "  All runs completed"
    echo "  Skipped : $num_skipped"
    echo "  Ran     : $current"
    echo "========================================"
}

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
cd "$SCRIPT_DIR"
preflight
cd src
run_all
