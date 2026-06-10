#!/bin/bash

# Hybrid MoE Inference Script (full CLI)
# Usage: ./hybrid_inference.sh [--option value] [--flag]

set -euo pipefail

# Defaults
SEED=46
NUM_EXPERTS=4
TOP_K=2
METRIC="cosine"
LAMBDA=0.5
TEMPERATURE=0.5
PRETRAIN_BACKBONE=false
BATCH_SIZE=64
DATASET_NAME="plantdoc"
BACKBONE_TYPE="non_pretrain_backbone"
BACKBONE_NAME="mobilenetv3small_torchvision"
MODEL_CLUSTERING_NAME="kmeans"
CHECKPOINT="best_checkpoint.pth"
RUN_TIME="run_20260610-031329"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --seed N                    Random seed (default: ${SEED})
  --num_experts N             Number of experts (default: ${NUM_EXPERTS})
  --top_k N                   Top-k (default: ${TOP_K})
  --metric METRIC             Metric: cosine|euclidean (default: ${METRIC})
  --lambda_ F                 Lambda weight (default: ${LAMBDA})
  --temperature F             Temperature (default: ${TEMPERATURE})
  --pretrain_backbone         Flag: use pretrain backbone (default: ${PRETRAIN_BACKBONE})
  --batch_size N              Batch size (default: ${BATCH_SIZE})
  --dataset_name NAME         Dataset name (default: ${DATASET_NAME})
  --backbone_type TYPE        Backbone type/folder (default: ${BACKBONE_TYPE})
  --backbone_name NAME        Backbone name (default: ${BACKBONE_NAME})
  --model_clustering_name N   Clustering model name (default: ${MODEL_CLUSTERING_NAME})
  --checkpoint FILE           Checkpoint file (default: ${CHECKPOINT})
  --runtime NAME              Runtime folder/name (default: ${RUN_TIME})
  -h, --help                  Show this help

Example:
  $0 --dataset_name plantdoc --backbone_type non_pretrain_models --backbone_name mobilenetv3small_torchvision \
     --model_clustering_name kmeans --num_experts 4 --lambda_ 0.5 --checkpoint best_checkpoint.pth --runtime run_20260604-153316
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2;;
        --num_experts) NUM_EXPERTS="$2"; shift 2;;
        --top_k|--top-k) TOP_K="$2"; shift 2;;
        --metric) METRIC="$2"; shift 2;;
        --lambda_|--lambda) LAMBDA="$2"; shift 2;;
        --temperature) TEMPERATURE="$2"; shift 2;;
        --pretrain_backbone) PRETRAIN_BACKBONE=true; shift;;
        --batch_size) BATCH_SIZE="$2"; shift 2;;
        --dataset_name) DATASET_NAME="$2"; shift 2;;
        --backbone_type) BACKBONE_TYPE="$2"; shift 2;;
        --backbone_name) BACKBONE_NAME="$2"; shift 2;;
        --model_clustering_name) MODEL_CLUSTERING_NAME="$2"; shift 2;;
        --checkpoint) CHECKPOINT="$2"; shift 2;;
        --runtime) RUN_TIME="$2"; shift 2;;
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
echo "  Starting HybridMoE Inference"
echo "========================================"
echo "seed:               $SEED"
echo "num_experts:        $NUM_EXPERTS"
echo "top_k:              $TOP_K"
echo "metric:             $METRIC"
echo "lambda_:            $LAMBDA"
echo "temperature:        $TEMPERATURE"
echo "pretrain_backbone:  $PRETRAIN_BACKBONE"
echo "batch_size:         $BATCH_SIZE"
echo "dataset_name:       $DATASET_NAME"
echo "backbone_type:      $BACKBONE_TYPE"
echo "backbone_name:      $BACKBONE_NAME"
echo "model_clustering_name: $MODEL_CLUSTERING_NAME"
echo "checkpoint:         $CHECKPOINT"
echo "runtime:            $RUN_TIME"
echo "========================================"

cd src

python -m inference.hybrid_moe_model.inference \
    --seed "$SEED" \
    --num_experts "$NUM_EXPERTS" \
    --top_k "$TOP_K" \
    --metric "$METRIC" \
    --lambda_ "$LAMBDA" \
    --temperature "$TEMPERATURE" \
    $( [ "$PRETRAIN_BACKBONE" = true ] && echo "--pretrain_backbone" ) \
    --batch_size "$BATCH_SIZE" \
    --dataset_name "$DATASET_NAME" \
    --backbone_type "$BACKBONE_TYPE" \
    --backbone_name "$BACKBONE_NAME" \
    --model_clustering_name "$MODEL_CLUSTERING_NAME" \
    --checkpoint "$CHECKPOINT" \
    --runtime "$RUN_TIME"

EXIT_CODE=$?

echo "========================================"
echo "  Inference finished (exit code: $EXIT_CODE)"
echo "========================================"
