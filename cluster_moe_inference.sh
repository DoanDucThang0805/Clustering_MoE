#!/bin/bash

# Cluster MoE Inference Script (full CLI)
# Usage: ./cluster_moe_inference.sh [--option value] [--flag]

set -euo pipefail

# Default configuration
DATASET_NAME="plantdoc"
BACKBONE_TYPE="dense_aligned_pretrain_backbone"
CENTROID_BACKBONE_TYPE="pretrain_backbone"
BACKBONE_NAME="mobilenetv3small_torchvision"
MODEL_CLUSTERING_NAME="kmeans"

SEED=49
NUM_EXPERTS=4
TOP_K=2
METRIC="cosine"
TEMPERATURE=0.5
PRETRAIN_BACKBONE=true

BATCH_SIZE=32
CHECKPOINT="best_checkpoint.pth"
RUN_TIME="run_20260705-110126"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --dataset_name NAME                 Dataset name (default: ${DATASET_NAME})
  --backbone_type TYPE                Backbone type / folder (default: ${BACKBONE_TYPE})
  --centroid_backbone_type TYPE       Namespace containing K-Means centroids
  --backbone_name NAME                Backbone name (default: ${BACKBONE_NAME})
  --model_clustering_name NAME        Clustering model name (default: ${MODEL_CLUSTERING_NAME})
  --checkpoint FILE                   Checkpoint file (default: ${CHECKPOINT})
  --runtime NAME                      Runtime name / folder (default: latest compatible run)
  --seed N                            Random seed (default: ${SEED})
  --num_experts N                     Number of experts (default: ${NUM_EXPERTS})
  --top_k N                           Top-k gating (default: ${TOP_K})
  --metric METRIC                     Metric: cosine|euclidean (default: ${METRIC})
  --temperature F                     Temperature (default: ${TEMPERATURE})
  --pretrain_backbone                  Flag: use pretraining for backbone
  --batch_size N                      Batch size (default: ${BATCH_SIZE})
  -h, --help                          Show this help

Example:
  $0 --dataset_name plantdoc --backbone_type non_pretrain_models --backbone_name mobilenetv3small_torchvision \
     --model_clustering_name kmeans --checkpoint best_checkpoint.pth --runtime run_20260604-153316
EOF
}

# Parse CLI args (long options)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset_name|--dataset-name)
            DATASET_NAME="$2"; shift 2;;
        --backbone_type|--backbone-type)
            BACKBONE_TYPE="$2"; shift 2;;
        --centroid_backbone_type|--centroid-backbone-type)
            CENTROID_BACKBONE_TYPE="$2"; shift 2;;
        --backbone_name|--backbone-name)
            BACKBONE_NAME="$2"; shift 2;;
        --model_clustering_name|--model-clustering-name)
            MODEL_CLUSTERING_NAME="$2"; shift 2;;
        --checkpoint)
            CHECKPOINT="$2"; shift 2;;
        --runtime)
            RUN_TIME="$2"; shift 2;;
        --seed)
            SEED="$2"; shift 2;;
        --num_experts)
            NUM_EXPERTS="$2"; shift 2;;
        --top_k|--top-k)
            TOP_K="$2"; shift 2;;
        --metric)
            METRIC="$2"; shift 2;;
        --temperature)
            TEMPERATURE="$2"; shift 2;;
        --pretrain_backbone|--pretrain-backbone)
            PRETRAIN_BACKBONE=true; shift;;
        --batch_size|--batch-size)
            BATCH_SIZE="$2"; shift 2;;
        -h|--help)
            usage; exit 0;;
        *)
            echo "Unknown option: $1" >&2; usage; exit 1;;
    esac
done

# Activate venv if present
if [ -d "venv" ]; then
    # shellcheck source=/dev/null
    source venv/bin/activate
fi

if [[ -z "$CENTROID_BACKBONE_TYPE" ]]; then
    CENTROID_BACKBONE_TYPE="$BACKBONE_TYPE"
fi

# Resolve the newest run containing the requested checkpoint when --runtime is
# omitted. The run_* directory names contain sortable timestamps.
CHECKPOINT_SEED_DIR="$SCRIPT_DIR/checkpoints/$DATASET_NAME/clustering_moe/$BACKBONE_TYPE/${BACKBONE_NAME}_backbone/$MODEL_CLUSTERING_NAME/temperature_$TEMPERATURE/G${NUM_EXPERTS}_${METRIC}_top${TOP_K}/seed_${SEED}"

if [[ -z "$RUN_TIME" ]]; then
    for candidate in "$CHECKPOINT_SEED_DIR"/run_*/"$CHECKPOINT"; do
        if [[ -f "$candidate" ]]; then
            RUN_TIME="$(basename "$(dirname "$candidate")")"
        fi
    done
fi

if [[ -z "$RUN_TIME" ]]; then
    echo "[ERROR] No $CHECKPOINT found under: $CHECKPOINT_SEED_DIR" >&2
    exit 1
fi

CHECKPOINT_PATH="$CHECKPOINT_SEED_DIR/$RUN_TIME/$CHECKPOINT"
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
fi

PRETRAINED_ARGS=()
if [[ "$PRETRAIN_BACKBONE" == true ]]; then
    PRETRAINED_ARGS+=(--pretrain_backbone)
fi

echo "========================================"
echo "  Starting ClusteringMoE Inference"
echo "========================================"
echo "Dataset:                $DATASET_NAME"
echo "Backbone type:          $BACKBONE_TYPE"
echo "Centroid backbone type: $CENTROID_BACKBONE_TYPE"
echo "Backbone name:          $BACKBONE_NAME"
echo "Clustering model name:  $MODEL_CLUSTERING_NAME"
echo "Seed:                   $SEED"
echo "Num Experts:            $NUM_EXPERTS"
echo "Top K:                  $TOP_K"
echo "Metric:                 $METRIC"
echo "Temperature:            $TEMPERATURE"
echo "Batch Size:             $BATCH_SIZE"
echo "Checkpoint:             $CHECKPOINT"
echo "Runtime:                $RUN_TIME"
echo "Pretrain backbone:      $PRETRAIN_BACKBONE"
echo "========================================"

cd src

python -m inference.cluster_moe_model.inference \
    --seed "$SEED" \
    --num_experts "$NUM_EXPERTS" \
    --top_k "$TOP_K" \
    --metric "$METRIC" \
    --temperature "$TEMPERATURE" \
    "${PRETRAINED_ARGS[@]}" \
    --batch_size "$BATCH_SIZE" \
    --dataset_name "$DATASET_NAME" \
    --backbone_type "$BACKBONE_TYPE" \
    --centroid_backbone_type "$CENTROID_BACKBONE_TYPE" \
    --backbone_name "$BACKBONE_NAME" \
    --model_clustering_name "$MODEL_CLUSTERING_NAME" \
    --checkpoint "$CHECKPOINT" \
    --runtime "$RUN_TIME"

echo "========================================"
echo "  Inference Completed"
echo "========================================"
