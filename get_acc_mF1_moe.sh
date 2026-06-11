#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_NAME="mobilenetv3small_torchvision_moe"
TYPE_MODEL="moe_temperature_0.5"
DATASET_NAME="plantdoc"
CSV_STORE_DIR="/media/data/minhht/clustering_moe/mean_acc_mF1_results/moe"
EXPORT_TO_CSV=true
CSV_FILENAME="linear_moe.csv"

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --model_name VALUE      Model name              (default: $MODEL_NAME)
  --type_model VALUE      Model type/config       (default: $TYPE_MODEL)
  --dataset_name VALUE    Dataset name            (default: $DATASET_NAME)
  --csv_store_dir VALUE   Directory to store CSV  (default: $CSV_STORE_DIR)
  --export_to_csv         Export results to CSV
  --csv_filename VALUE    Output CSV filename      (default: $CSV_FILENAME)
  -h, --help              Show this help message

Example:
  $0 --type_model moe_contextaware_temp1.0 --export_to_csv --csv_filename out.csv
EOF
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name)    MODEL_NAME="$2";    shift 2 ;;
    --type_model)    TYPE_MODEL="$2";    shift 2 ;;
    --dataset_name)  DATASET_NAME="$2";  shift 2 ;;
    --csv_store_dir) CSV_STORE_DIR="$2"; shift 2 ;;
    --csv_filename)  CSV_FILENAME="$2";  shift 2 ;;
    --export_to_csv) EXPORT_TO_CSV=true; shift   ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Activate virtualenv if present
if [[ -f "$REPO_ROOT/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/venv/bin/activate"
fi

# Locate Python
if command -v python3 &>/dev/null; then
  PYTHON=python3
elif command -v python &>/dev/null; then
  PYTHON=python
else
  echo "Error: Python not found in PATH" >&2
  exit 1
fi

# Verify entry-point exists
SCRIPT="$REPO_ROOT/src/benchmark/get_acc_mF1_moe.py"
if [[ ! -f "$SCRIPT" ]]; then
  echo "Error: script not found: $SCRIPT" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Build and run command
# ---------------------------------------------------------------------------
CMD=(
  "$PYTHON" -m benchmark.get_acc_mF1_moe
  --model_name    "$MODEL_NAME"
  --type_model    "$TYPE_MODEL"
  --dataset_name  "$DATASET_NAME"
  --csv_store_dir "$CSV_STORE_DIR"
)

if [[ "$EXPORT_TO_CSV" == true ]]; then
  CMD+=(--export_to_csv --csv_filename "$CSV_FILENAME")
fi

echo "Running: ${CMD[*]}"
cd "$REPO_ROOT/src"
exec "${CMD[@]}"
