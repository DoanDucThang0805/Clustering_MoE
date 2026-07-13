#!/usr/bin/env bash
# CP7 — Train Soft MoE baseline (classifier-side, all-expert), 5 seed, tuần tự, idempotent.
# KHÔNG đụng moe_train.sh / training/moe.py.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults — training budget Y HỆT moe_train.sh (yêu cầu "cùng training budget" của PDF)
SEEDS=(42 43 44 45 46)
DATASET_NAME="plantdoc"
NUM_EXPERTS=4
BACKBONE_NAME="mobilenetv3small_torchvision"
CONTEXT_DIM=6
TEMPERATURE=0.5
NUM_EPOCHS=400
BATCH_SIZE=64
LR=0.001
WEIGHT_DECAY=0.001
PRETRAIN_BACKBONE=true
FORCE_RETRAIN=false

usage() {
    cat <<EOF
CP7 Soft MoE training (all-expert, no top-k, no balance loss)

  --seed N              Chỉ chạy 1 seed
  --seeds "A B C"       Danh sách seed (default: ${SEEDS[*]})
  --dataset_name NAME   plantdoc | plantvillage (default: $DATASET_NAME)
  --num_experts N       (default: $NUM_EXPERTS)
  --backbone_name NAME  (default: $BACKBONE_NAME)
  --temperature F       (default: $TEMPERATURE — giữ 0.5 như MoE/Cluster-MoE)
  --num_epochs N        (default: $NUM_EPOCHS)
  --batch_size N        (default: $BATCH_SIZE)
  --lr F                (default: $LR)
  --weight_decay F      (default: $WEIGHT_DECAY)
  --no_pretrain         Không dùng backbone ImageNet-pretrained
  --force               Train lại kể cả đã có checkpoint
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEEDS=("$2"); shift 2;;
        --seeds) read -r -a SEEDS <<< "$2"; shift 2;;
        --dataset_name) DATASET_NAME="$2"; shift 2;;
        --num_experts) NUM_EXPERTS="$2"; shift 2;;
        --backbone_name) BACKBONE_NAME="$2"; shift 2;;
        --context_dim) CONTEXT_DIM="$2"; shift 2;;
        --temperature) TEMPERATURE="$2"; shift 2;;
        --num_epochs) NUM_EPOCHS="$2"; shift 2;;
        --batch_size) BATCH_SIZE="$2"; shift 2;;
        --lr) LR="$2"; shift 2;;
        --weight_decay) WEIGHT_DECAY="$2"; shift 2;;
        --no_pretrain) PRETRAIN_BACKBONE=false; shift;;
        --force) FORCE_RETRAIN=true; shift;;
        -h|--help) usage; exit 0;;
        *) echo "[ERROR] Unknown arg: $1"; usage; exit 1;;
    esac
done

[ -d "venv" ] && source venv/bin/activate

# Idempotent: skip nếu seed đã có best_checkpoint.
# LƯU Ý: path dùng $DATASET_NAME (moe_train.sh từng hardcode 'plantdoc' -> skip nhầm).
is_done() {
    local seed=$1
    local seed_dir="$SCRIPT_DIR/checkpoints/$DATASET_NAME/soft_moe/${BACKBONE_NAME}_softmoe/${NUM_EXPERTS}_experts/seed_${seed}"
    [[ "$FORCE_RETRAIN" == true ]] && return 1
    for run_dir in "$seed_dir"/run_*; do
        [ -f "$run_dir/best_checkpoint.pth" ] && return 0
    done
    return 1
}

echo "========================================"
echo "  CP7 — Soft MoE (all-expert, no top-k)"
echo "========================================"
echo "  dataset      : $DATASET_NAME"
echo "  seeds        : ${SEEDS[*]}"
echo "  num_experts  : $NUM_EXPERTS  (ALL active — không top-k)"
echo "  backbone     : $BACKBONE_NAME"
echo "  temperature  : $TEMPERATURE"
echo "  budget       : ${NUM_EPOCHS}ep / bs ${BATCH_SIZE} / lr ${LR} / wd ${WEIGHT_DECAY}"
echo "  balance loss : KHÔNG (CrossEntropy thuần)"
echo "========================================"

total=${#SEEDS[@]}
current=0
ran=0
skipped=0

for seed in "${SEEDS[@]}"; do
    current=$((current + 1))
    if is_done "$seed"; then
        echo "  [SKIP $current/$total] seed=$seed đã có checkpoint."
        skipped=$((skipped + 1))
        continue
    fi

    echo "  [RUN $current/$total] seed=$seed"
    ( cd src && python -m training.soft_moe \
        --seed "$seed" \
        --dataset_name "$DATASET_NAME" \
        --num_experts "$NUM_EXPERTS" \
        --backbone_name "$BACKBONE_NAME" \
        --context_dim "$CONTEXT_DIM" \
        --temperature "$TEMPERATURE" \
        --num_epochs "$NUM_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        $( [ "$PRETRAIN_BACKBONE" = true ] && echo "--pretrain_backbone" ) )
    ran=$((ran + 1))
    echo "  ✓ Done seed=$seed"
done

echo "========================================"
echo "  All seeds finished   Ran: $ran   Skipped: $skipped"
echo "========================================"
