#!/bin/bash

# MoE Baseline Routing Diagnostics Script
# Usage: ./moe_baseline_routing_diagnostics.sh [--option value]

set -euo pipefail
clear

# Defaults
CHECKPOINT="/media/data/minhht/clustering_moe/checkpoints/plantdoc/moe_temperature_0.5/mobilenetv3small_torchvision_moe/4_experts/top_2/seed_45/run_20260609-151705/best_checkpoint.pth"
OUTPUT_DIR="reports/moe_routing_diagnostics"
SPLIT="test"
BATCH_SIZE=32
CSV_NAME="moe_baseline_expert_usage.csv"
PLOT_NAME="moe_baseline_expert_usage.png"
BACKBONE_NAME="mobilenetv3small_torchvision"
PRETRAIN_BACKBONE=false

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --checkpoint FILE         Path to MoE checkpoint (default: ${CHECKPOINT})
  --output_dir DIR          Directory to save diagnostic CSV and plots (default: ${OUTPUT_DIR})
  --split NAME              Dataset split for diagnostics: train, validation, test (default: ${SPLIT})
  --batch_size N            Batch size (default: ${BATCH_SIZE})
  --csv_name FILE           Output CSV file name (default: ${CSV_NAME})
  --plot_name FILE          Output plot file name (default: ${PLOT_NAME})
  --backbone_name NAME      Backbone name (default: ${BACKBONE_NAME})
  --pretrain_backbone       Flag: use pretrain backbone (default: ${PRETRAIN_BACKBONE})
  -h, --help                Show this help

Example:
  $0 --checkpoint checkpoints/moe_checkpoint.pth --output_dir reports/diagnostics
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2;;
        --output_dir) OUTPUT_DIR="$2"; shift 2;;
        --split) SPLIT="$2"; shift 2;;
        --batch_size) BATCH_SIZE="$2"; shift 2;;
        --csv_name) CSV_NAME="$2"; shift 2;;
        --plot_name) PLOT_NAME="$2"; shift 2;;
        --backbone_name) BACKBONE_NAME="$2"; shift 2;;
        --pretrain_backbone) PRETRAIN_BACKBONE=true; shift;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage; exit 1;;
    esac
done

# Convert paths to absolute to avoid issues after cd src
if [[ "$CHECKPOINT" != /* ]]; then
    CHECKPOINT_ABS="$SCRIPT_DIR/$CHECKPOINT"
else
    CHECKPOINT_ABS="$CHECKPOINT"
fi

if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR_ABS="$SCRIPT_DIR/$OUTPUT_DIR"
else
    OUTPUT_DIR_ABS="$OUTPUT_DIR"
fi

# Activate venv if available
if [ -d "venv" ]; then
    # shellcheck source=/dev/null
    source venv/bin/activate
fi

echo "========================================"
echo "  Starting MoE Routing Diagnostics"
echo "========================================"
echo "checkpoint:         $CHECKPOINT"
echo "output_dir:         $OUTPUT_DIR"
echo "split:              $SPLIT"
echo "batch_size:         $BATCH_SIZE"
echo "csv_name:           $CSV_NAME"
echo "plot_name:          $PLOT_NAME"
echo "backbone_name:      $BACKBONE_NAME"
echo "pretrain_backbone:  $PRETRAIN_BACKBONE"
echo "========================================"

cd src

python -m diagnostics.moe_baseline_routing_diagnostics \
    --checkpoint "$CHECKPOINT_ABS" \
    --output_dir "$OUTPUT_DIR_ABS" \
    --split "$SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --csv_name "$CSV_NAME" \
    --plot_name "$PLOT_NAME" \
    --backbone_name "$BACKBONE_NAME" \
    $( [ "$PRETRAIN_BACKBONE" = true ] && echo "--pretrain_backbone" )

EXIT_CODE=$?

echo "========================================"
echo "  Diagnostics finished (exit code: $EXIT_CODE)"
echo "========================================"
