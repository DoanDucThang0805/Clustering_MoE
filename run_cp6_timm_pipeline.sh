#!/usr/bin/env bash
# CP6 pipeline — EfficientNet-B0 trên PlantDoc, 5 seed, tuần tự, idempotent.
# Mỗi seed: dense B0 + MoE B0 + extract + kmeans + Cluster-MoE B0 (dense-aligned).
# Cả 3 model dùng backbone efficientnetb0_torchvision. Bỏ qua bước đã có checkpoint.
set -u
ROOT="/media/data/minhht/clustering_moe"
cd "$ROOT"
DS="plantdoc"
BB="efficientnetb0_timm"
SEEDS=(42 43 44 45 46)

have() { ls $1 >/dev/null 2>&1; }
latest() { ls -td $1 2>/dev/null | head -1; }

for S in "${SEEDS[@]}"; do
  echo "########## CP6-TIMM SEED $S START $(date +%H:%M:%S) ##########"

  # 1) Dense B0 (baseline + backbone align)
  DENSE="checkpoints/$DS/pretrain_baseline/$BB/seed_$S"
  if have "$DENSE/run_*/best_checkpoint.pth"; then echo "[skip] dense B0 seed $S"; else
    echo ">> dense B0 seed $S"
    ( cd src && python -m training.efficientnetb0_timm --seed "$S" --dataset_name "$DS" )
  fi
  DENSE_BEST=$(latest "$DENSE/run_*")/best_checkpoint.pth
  DENSE_RUN=$(basename "$(dirname "$DENSE_BEST")")
  echo "   dense_run=$DENSE_RUN"

  # 2) Learned-gate MoE B0
  MOE="checkpoints/$DS/moe_temperature_0.5_pretrain_backbone/${BB}_moe/4_experts/top_2/seed_$S"
  if have "$MOE/run_*/best_checkpoint.pth"; then echo "[skip] MoE B0 seed $S"; else
    echo ">> MoE B0 seed $S"
    bash moe_train.sh --seed "$S" --dataset_name "$DS" --backbone_name "$BB" \
        --type_model moe_temperature_0.5_pretrain_backbone
  fi

  # 3-4) Extract embedding + KMeans (fit train). Bỏ qua nếu centroid có.
  CENT="clustering_results/$DS/pretrain_backbone/${BB}_backbone/kmeans/cosine/seed_$S/clusters_kmeans_G4_seed$S.npz"
  if [ -f "$CENT" ]; then echo "[skip] centroid B0 seed $S"; else
    echo ">> extract embedding B0 (all splits) seed $S từ $DENSE_RUN"
    bash extract_embedding.sh --dataset_name "$DS" --model_name "$BB" --type_model pretrain_baseline \
        --type_backbone pretrain_backbone --seed "$S" --run_time "$DENSE_RUN" --split all
    echo ">> kmeans G4 cosine B0 seed $S (fit train)"
    ( cd src && python -m training.kmean --dataset_name "$DS" --backbone_type pretrain_backbone \
        --backbone_name "$BB" --seed "$S" --num_clusters 4 --metric cosine --feature_split train )
  fi

  # 5) Cluster-MoE B0 (dense-aligned)
  CLU="checkpoints/$DS/clustering_moe/dense_aligned_pretrain_backbone/${BB}_backbone/kmeans/temperature_0.5/G4_cosine_top2/seed_$S"
  if have "$CLU/run_*/best_checkpoint.pth"; then echo "[skip] Cluster-MoE B0 seed $S"; else
    echo ">> Cluster-MoE B0 seed $S (backbone=$DENSE_BEST)"
    ( cd src && python -m training.clustering_moe --seed "$S" --dataset_name "$DS" \
        --num_experts 4 --top_k 2 --distance_metric cosine --temperature 0.5 \
        --lr 3e-4 --weight_decay 1e-2 --label_smoothing 0.05 \
        --num_epochs 400 --batch_size 32 \
        --backbone_type dense_aligned_pretrain_backbone \
        --centroid_backbone_type pretrain_backbone \
        --backbone_name "$BB" --model_clustering_name kmeans \
        --backbone_checkpoint "$ROOT/$DENSE_BEST" --pretrain_backbone )
  fi

  echo "########## CP6-TIMM SEED $S DONE $(date +%H:%M:%S) ##########"
done
echo "########## CP6-TIMM PIPELINE ALL DONE ##########"
