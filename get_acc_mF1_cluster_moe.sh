#!/bin/bash

# Script to evaluate Clustering MoE models on test dataset
# Computes accuracy and macro-F1 scores for all discovered checkpoints

set -e

# ==================== CONFIGURATION ====================
# Edit these variables to customize the evaluation
DATASET_NAME="plantdoc"
TYPE_MODEL="clustering_moe"
BACKBONE_TYPE="non_pretrain_backbone"
BACKBONE_NAME="mobilenetv3small_torchvision"
MODEL_CLUSTERING_NAME="kmeans"
TEMPERATURE="0.5"
CSV_STORE_DIR="./mean_acc_mF1_results/cluster_moe"
EXPORT_TO_CSV="--export_to_csv"  # Use "" to disable export
CSV_FILENAME="cluster_moe_${BACKBONE_TYPE}_${BACKBONE_NAME}_${MODEL_CLUSTERING_NAME}_temp${TEMPERATURE}.csv"
# ======================================================

# Print configuration
echo "=========================================="
echo "Clustering MoE Model Evaluation"
echo "=========================================="
echo "Dataset:              $DATASET_NAME"
echo "Model Type:           $TYPE_MODEL"
echo "Backbone Type:        $BACKBONE_TYPE"
echo "Backbone Name:        $BACKBONE_NAME"
echo "Clustering Method:    $MODEL_CLUSTERING_NAME"
echo "Temperature:          $TEMPERATURE"
echo "CSV Output Dir:       $CSV_STORE_DIR"
echo "CSV Filename:         $CSV_FILENAME"
echo "Export to CSV:        $EXPORT_TO_CSV"
echo "=========================================="
echo ""

# Create output directory if it doesn't exist
mkdir -p "$CSV_STORE_DIR"

# Build and run Python command
echo "Running evaluation..."
python src/benchmark/get_acc_mF1_cluster_moe.py \
  --dataset_name "$DATASET_NAME" \
  --type_model "$TYPE_MODEL" \
  --backbone_type "$BACKBONE_TYPE" \
  --backbone_name "$BACKBONE_NAME" \
  --model_clustering_name "$MODEL_CLUSTERING_NAME" \
  --temperature "$TEMPERATURE" \
  --csv_store_dir "$CSV_STORE_DIR" \
  --csv_filename "$CSV_FILENAME" \
  $EXPORT_TO_CSV

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    if [ ! -z "$EXPORT_TO_CSV" ]; then
        echo "Results saved to: $CSV_STORE_DIR/$CSV_FILENAME"
    fi
    echo "=========================================="
else
    echo ""
    echo "Evaluation failed!"
    exit 1
fi
