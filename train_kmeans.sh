#!/bin/bash
set -euo pipefail

source venv/bin/activate
cd src

DATASET_NAME="plantdoc"
BACKBONE_TYPE="pretrain_backbone"
BACKBONE_NAME="mobilenetv3small_torchvision"

SEEDS=(46)
METRICS=("cosine" "euclidean")
NUM_CLUSTERS=(2 3 4 5 6 8)

total_runs=$(( ${#SEEDS[@]} * ${#METRICS[@]} ))
current_run=0

echo "========================================"
echo "  Pretrained Backbone K-Means Grid"
echo "========================================"
echo "  Seeds      : ${SEEDS[*]}"
echo "  Metrics    : ${METRICS[*]}"
echo "  Clusters   : ${NUM_CLUSTERS[*]}"
echo "  Total runs : $total_runs"
echo "========================================"

for seed in "${SEEDS[@]}"; do
    for metric in "${METRICS[@]}"; do
        current_run=$((current_run + 1))

        echo ""
        echo "[$current_run/$total_runs] seed=$seed metric=$metric"

        python -m training.kmean \
            --dataset_name "$DATASET_NAME" \
            --backbone_type "$BACKBONE_TYPE" \
            --backbone_name "$BACKBONE_NAME" \
            --seed "$seed" \
            --num_clusters "${NUM_CLUSTERS[@]}" \
            --metric "$metric"
    done
done

echo ""
echo "Completed all $total_runs K-Means runs."
