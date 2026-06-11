#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<EOF
Usage: $0 --type_model TYPE_MODEL [options]

Required:
  --type_model VALUE        Model type/config (e.g. moe_contextaware_temp1.0)

Options:
  --model_name VALUE        Model name (default: mobilenetv3small_moe)
  --dataset_name VALUE      Dataset name (default: plantdoc)
  --csv_store_dir VALUE     Directory to store CSV (default: ./results)
  --export_to_csv           Export aggregated results to CSV
  --csv_filename VALUE      CSV filename when exporting (default: results_moe.csv)
  -h, --help                Show this help message

Example:
  $0 --type_model moe_contextaware_temp1.0 --export_to_csv --csv_filename out.csv
EOF
}

# Defaults
MODEL_NAME="mobilenetv3small_torchvision_moe"
TYPE_MODEL="moe_temperature_0.5"
DATASET_NAME="plantdoc"
CSV_STORE_DIR="/media/data/minhht/clustering_moe/mean_acc_mF1_results"
EXPORT_TO_CSV=true
CSV_FILENAME="results_moe.csv"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name)
      MODEL_NAME="$2"; shift 2;;
    --model_name=*)
      MODEL_NAME="${1#*=}"; shift;;
    --type_model)
      TYPE_MODEL="$2"; shift 2;;
    --type_model=*)
      TYPE_MODEL="${1#*=}"; shift;;
    --dataset_name)
      DATASET_NAME="$2"; shift 2;;
    --dataset_name=*)
      DATASET_NAME="${1#*=}"; shift;;
    --csv_store_dir)
      CSV_STORE_DIR="$2"; shift 2;;
    --csv_store_dir=*)
      CSV_STORE_DIR="${1#*=}"; shift;;
    --export_to_csv)
      EXPORT_TO_CSV=true; shift;;
    --csv_filename)
      CSV_FILENAME="$2"; shift 2;;
    --csv_filename=*)
      CSV_FILENAME="${1#*=}"; shift;;
    -h|--help)
      print_usage; exit 0;;
    --)
      shift; break;;
    *)
      echo "Unknown option: $1" >&2; print_usage; exit 1;;
  esac
done

if [[ -z "$TYPE_MODEL" ]]; then
  echo "Error: --type_model is required" >&2
  print_usage
  exit 2
fi

# Activate virtualenv if present
if [[ -f "./venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ./venv/bin/activate
fi

# Find python
PYTHON=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "Python not found in PATH" >&2
  exit 3
fi

# Ensure script exists
SCRIPT_PATH="src/benchmark/get_acc_mF1_moe.py"
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Cannot find $SCRIPT_PATH from repository root" >&2
  exit 4
fi

# Build command: run as module from src/
cd src
python -m benchmark.get_acc_mF1_moe \
  --model_name "$MODEL_NAME" \
  --type_model "$TYPE_MODEL" \
  --dataset_name "$DATASET_NAME" \
  --csv_store_dir "$CSV_STORE_DIR"


if [[ "$EXPORT_TO_CSV" == true ]]; then
  CMD+=(--export_to_csv --csv_filename "$CSV_FILENAME")
fi

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
