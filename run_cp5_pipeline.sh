#!/usr/bin/env bash
# CP5 pipeline — PlantVillage tomato, 5 seed, tuần tự, idempotent.
# Mỗi seed: dense lamb1k (baseline) + dense torchvision (cho cluster) + MoE +
# extract embedding + kmeans + Cluster-MoE. Bỏ qua bước đã có best_checkpoint.
set -u
ROOT="/media/data/minhht/clustering_moe"
cd "$ROOT"
DS="plantvillage"
SEEDS=(42 43 44 45 46)

have() { ls $1 >/dev/null 2>&1; }               # glob tồn tại?
latest() { ls -td $1 2>/dev/null | head -1; }   # dir mới nhất

for S in "${SEEDS[@]}"; do
  echo "########## CP5 SEED $S START $(date +%H:%M:%S) ##########"

  # 1) Dense lamb1k (baseline báo cáo)
  LAMB="checkpoints/$DS/pretrain_baseline/mobilenetv3small_timm_lamb1k/seed_$S"
  if have "$LAMB/run_*/best_checkpoint.pth"; then echo "[skip] dense lamb1k seed $S"; else
    echo ">> dense lamb1k seed $S"
    ( cd src && python -m training.mobilenetv3smallv2 --seed "$S" --dataset_name "$DS" )
  fi

  # 2) Dense torchvision (cho cluster alignment)
  TV="checkpoints/$DS/pretrain_baseline/mobilenetv3small_torchvision/seed_$S"
  if have "$TV/run_*/best_checkpoint.pth"; then echo "[skip] dense torchvision seed $S"; else
    echo ">> dense torchvision seed $S"
    ( cd src && python -m training.mobilenetv3small --seed "$S" --dataset_name "$DS" )
  fi
  TV_BEST=$(latest "$TV/run_*")/best_checkpoint.pth
  TV_RUN=$(basename "$(dirname "$TV_BEST")")
  echo "   tv_run=$TV_RUN"

  # 3) Learned-gate MoE
  MOE="checkpoints/$DS/moe_temperature_0.5_pretrain_backbone/mobilenetv3small_torchvision_moe/4_experts/top_2/seed_$S"
  if have "$MOE/run_*/best_checkpoint.pth"; then echo "[skip] MoE seed $S"; else
    echo ">> MoE seed $S"
    bash moe_train.sh --seed "$S" --dataset_name "$DS" --type_model moe_temperature_0.5_pretrain_backbone
  fi

  # 4-5) Extract embedding + KMeans (centroid fit train-only). Bỏ qua nếu centroid có.
  CENT="clustering_results/$DS/pretrain_backbone/mobilenetv3small_torchvision_backbone/kmeans/cosine/seed_$S/clusters_kmeans_G4_seed$S.npz"
  if [ -f "$CENT" ]; then echo "[skip] centroid seed $S"; else
    echo ">> extract embedding (all splits) seed $S từ $TV_RUN"
    bash extract_embedding.sh --dataset_name "$DS" --type_model pretrain_baseline \
        --type_backbone pretrain_backbone --seed "$S" --run_time "$TV_RUN" --split all
    echo ">> kmeans G4 cosine seed $S (fit train)"
    ( cd src && python -m training.kmean --dataset_name "$DS" --backbone_type pretrain_backbone \
        --backbone_name mobilenetv3small_torchvision --seed "$S" --num_clusters 4 \
        --metric cosine --feature_split train )
  fi

  # 6) Cluster-MoE (dense-aligned)
  CLU="checkpoints/$DS/clustering_moe/dense_aligned_pretrain_backbone/mobilenetv3small_torchvision_backbone/kmeans/temperature_0.5/G4_cosine_top2/seed_$S"
  if have "$CLU/run_*/best_checkpoint.pth"; then echo "[skip] Cluster-MoE seed $S"; else
    echo ">> Cluster-MoE seed $S (backbone=$TV_BEST)"
    ( cd src && python -m training.clustering_moe --seed "$S" --dataset_name "$DS" \
        --num_experts 4 --top_k 2 --distance_metric cosine --temperature 0.5 \
        --lr 3e-4 --weight_decay 1e-2 --label_smoothing 0.05 \
        --num_epochs 400 --batch_size 32 \
        --backbone_type dense_aligned_pretrain_backbone \
        --centroid_backbone_type pretrain_backbone \
        --backbone_name mobilenetv3small_torchvision --model_clustering_name kmeans \
        --backbone_checkpoint "$ROOT/$TV_BEST" --pretrain_backbone )
  fi

  echo "########## CP5 SEED $S DONE $(date +%H:%M:%S) ##########"
done
echo "########## CP5 PIPELINE ALL DONE ##########"
