#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_NAME="plantdoc"
BACKBONE_TYPE="imagenet_initialization_backbone"
BACKBONE_NAME="mobilenetv3small_torchvision"
MODEL_CLUSTERING_NAME="kmeans"

SEEDS=(42 46 50)
NUM_EXPERTS=4
TOP_K=2
METRIC="cosine"
TEMPERATURE=0.5

LR=1e-3
WEIGHT_DECAY=1e-3
NUM_EPOCHS=400
BATCH_SIZE=32

embedding_path() {
    local seed=$1
    echo "$SCRIPT_DIR/feature_embeddings/$DATASET_NAME/$BACKBONE_TYPE/${BACKBONE_NAME}_backbone/seed_${seed}/features_train_seed${seed}.npz"
}

centroid_path() {
    local seed=$1
    echo "$SCRIPT_DIR/clustering_results/$DATASET_NAME/$BACKBONE_TYPE/${BACKBONE_NAME}_backbone/$MODEL_CLUSTERING_NAME/$METRIC/seed_${seed}/clusters_kmeans_G${NUM_EXPERTS}_seed${seed}.npz"
}

training_is_done() {
    local seed=$1
    local seed_dir="$SCRIPT_DIR/checkpoints/$DATASET_NAME/clustering_moe/$BACKBONE_TYPE/${BACKBONE_NAME}_backbone/$MODEL_CLUSTERING_NAME/temperature_$TEMPERATURE/G${NUM_EXPERTS}_${METRIC}_top${TOP_K}/seed_${seed}"
    local run_dir

    for run_dir in "$seed_dir"/run_*; do
        if [ -f "$run_dir/best_checkpoint.pth" ] &&
           [ -f "$run_dir/last_checkpoint.pth" ]; then
            return 0
        fi
    done
    return 1
}

if [ ! -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found: $SCRIPT_DIR/venv"
    exit 1
fi

source "$SCRIPT_DIR/venv/bin/activate"

if ! python -c \
    'import sys, torch; print(f"CUDA: {torch.cuda.is_available()}"); sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "[ERROR] CUDA is required for this pilot."
    exit 1
fi

cd "$SCRIPT_DIR/src"

echo "========================================"
echo "  ImageNet-Aligned Cluster-MoE Pilot"
echo "========================================"
echo "  Seeds       : ${SEEDS[*]}"
echo "  LR          : $LR"
echo "  Config      : G=$NUM_EXPERTS top_k=$TOP_K $METRIC tau=$TEMPERATURE"
echo "========================================"

for seed in "${SEEDS[@]}"; do
    embedding=$(embedding_path "$seed")
    if [ -f "$embedding" ]; then
        echo "[SKIP] Embedding exists for seed $seed"
    else
        echo "[EMBEDDING] seed=$seed weights=ImageNet"
        python -m embedding.pretrain_backbone.image_embedding \
            --dataset_name "$DATASET_NAME" \
            --model_name "$BACKBONE_NAME" \
            --type_backbone "$BACKBONE_TYPE" \
            --weights_source imagenet \
            --split train \
            --seed "$seed" \
            --batch_size 64 \
            --num_workers 4
    fi

    centroid=$(centroid_path "$seed")
    if [ -f "$centroid" ]; then
        echo "[SKIP] Centroid exists for seed $seed"
    else
        echo "[KMEANS] seed=$seed G=$NUM_EXPERTS metric=$METRIC"
        python -m training.kmean \
            --dataset_name "$DATASET_NAME" \
            --backbone_type "$BACKBONE_TYPE" \
            --backbone_name "$BACKBONE_NAME" \
            --seed "$seed" \
            --num_clusters "$NUM_EXPERTS" \
            --metric "$METRIC"
    fi
done

for seed in "${SEEDS[@]}"; do
    if training_is_done "$seed"; then
        echo "[SKIP] Complete Cluster-MoE run exists for seed $seed"
        continue
    fi

    echo "[TRAIN] seed=$seed"
    python -m training.clustering_moe \
        --seed "$seed" \
        --num_experts "$NUM_EXPERTS" \
        --top_k "$TOP_K" \
        --distance_metric "$METRIC" \
        --temperature "$TEMPERATURE" \
        --pretrain_backbone \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --num_epochs "$NUM_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --dataset_name "$DATASET_NAME" \
        --backbone_type "$BACKBONE_TYPE" \
        --backbone_name "$BACKBONE_NAME" \
        --model_clustering_name "$MODEL_CLUSTERING_NAME"
done

echo "Completed ImageNet-aligned pilot."
