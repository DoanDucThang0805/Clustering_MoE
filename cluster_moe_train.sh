#!/bin/bash
set -e
clear

# ─────────────────────────────────────────────
# Fixed Configuration
# ─────────────────────────────────────────────
DATASET_NAME="plantdoc"
BACKBONE_TYPE="non_pretrain_backbone"
BACKBONE_NAME="mobilenetv3small_torchvision"
MODEL_CLUSTERING_NAME="kmeans"
METRIC="euclidean"
TEMPERATURE=0.5
PRETRAIN_BACKBONE=false
LR=1e-3
WEIGHT_DECAY=1e-3
NUM_EPOCHS=400
BATCH_SIZE=32

# ─────────────────────────────────────────────
# Search Space
# ─────────────────────────────────────────────
SEEDS=(42 43 44 45)

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
# Already completed runs — skip these
# Format: "num_experts top_k seed"
# ─────────────────────────────────────────────
DONE_RUNS=(
    # "4 2 42"
    # "4 2 43"
    # "4 2 44"
    # "4 2 45"
)

# ─────────────────────────────────────────────
# Functions
# ─────────────────────────────────────────────

is_done() {
    local num_experts=$1
    local top_k=$2
    local seed=$3
    for done in "${DONE_RUNS[@]}"; do
        if [ "$done" = "$num_experts $top_k $seed" ]; then
            return 0   # already done
        fi
    done
    return 1   # not done yet
}


train_one() {
    local num_experts=$1
    local top_k=$2
    local seed=$3

    echo ""
    echo "  seed=$seed  |  G=$num_experts  |  top_k=$top_k"
    echo "----------------------------------------"

    python -m training.clustering_moe \
        --seed              "$seed"         \
        --num_experts       "$num_experts"  \
        --top_k             "$top_k"        \
        --distance_metric   "$METRIC"       \
        --temperature       "$TEMPERATURE"  \
        --lr                "$LR"           \
        --weight_decay      "$WEIGHT_DECAY" \
        --num_epochs        "$NUM_EPOCHS"   \
        --batch_size        "$BATCH_SIZE"   \
        --dataset_name      "$DATASET_NAME" \
        --backbone_type     "$BACKBONE_TYPE" \
        --backbone_name     "$BACKBONE_NAME" \
        --model_clustering_name "$MODEL_CLUSTERING_NAME" \
        $([ "$PRETRAIN_BACKBONE" = true ] && echo "--pretrain_backbone")

    echo "  ✓ Done: G=$num_experts top_k=$top_k seed=$seed"
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
    echo "  Total runs    : $total"
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
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

cd src
run_all