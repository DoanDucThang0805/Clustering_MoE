#!/usr/bin/env bash
# Cluster-MoE B0-timm RETRAIN2: dùng dense efficientnetb0_timm RESTART-2 làm backbone align.
# Namespace riêng (*_retrain2) — KHÔNG đè bộ CP6 gốc. Tuần tự 5 seed, idempotent.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[ -d venv ] && source venv/bin/activate

DS="plantdoc"
BB="efficientnetb0_timm"
EMB_NS="pretrain_backbone_retrain2"                    # embeddings + centroids
CLU_NS="dense_aligned_pretrain_backbone_retrain2"      # checkpoints cluster

# Dense B0-timm RESTART-2 (bộ tốt nhất: 87.44 ± 1.48) — ghim tường minh từng seed
declare -A DENSE_RUN=(
  [42]=run_20260714-044024
  [43]=run_20260714-050934
  [44]=run_20260714-054708
  [45]=run_20260714-061449
  [46]=run_20260714-065921
)

have() { ls $1 >/dev/null 2>&1; }

for S in 42 43 44 45 46; do
  RUN="${DENSE_RUN[$S]}"
  DENSE_BEST="checkpoints/$DS/pretrain_baseline/$BB/seed_$S/$RUN/best_checkpoint.pth"
  echo "########## CLUSTER-RETRAIN2 SEED $S START $(date +%H:%M:%S) (dense=$RUN) ##########"
  [ -f "$DENSE_BEST" ] || { echo "[ERROR] thiếu $DENSE_BEST"; exit 1; }

  # 1-2) Extract embedding + KMeans vào namespace retrain2
  CENT="clustering_results/$DS/$EMB_NS/${BB}_backbone/kmeans/cosine/seed_$S/clusters_kmeans_G4_seed$S.npz"
  if [ -f "$CENT" ]; then echo "[skip] centroid retrain2 seed $S"; else
    echo ">> extract embedding (all splits) seed $S từ $RUN"
    bash extract_embedding.sh --dataset_name "$DS" --model_name "$BB" --type_model pretrain_baseline \
        --type_backbone "$EMB_NS" --seed "$S" --run_time "$RUN" --split all
    echo ">> kmeans G4 cosine seed $S (fit train)"
    ( cd src && python -m training.kmean --dataset_name "$DS" --backbone_type "$EMB_NS" \
        --backbone_name "$BB" --seed "$S" --num_clusters 4 --metric cosine --feature_split train )
  fi

  # 3) Cluster-MoE (dense-aligned, recipe CP6: lr 3e-4, wd 1e-2, ls 0.05)
  CLU="checkpoints/$DS/clustering_moe/$CLU_NS/${BB}_backbone/kmeans/temperature_0.5/G4_cosine_top2/seed_$S"
  if have "$CLU/run_*/best_checkpoint.pth"; then echo "[skip] Cluster-MoE retrain2 seed $S"; else
    echo ">> Cluster-MoE retrain2 seed $S"
    ( cd src && python -m training.clustering_moe --seed "$S" --dataset_name "$DS" \
        --num_experts 4 --top_k 2 --distance_metric cosine --temperature 0.5 \
        --lr 3e-4 --weight_decay 1e-2 --label_smoothing 0.05 \
        --num_epochs 400 --batch_size 32 \
        --backbone_type "$CLU_NS" \
        --centroid_backbone_type "$EMB_NS" \
        --backbone_name "$BB" --model_clustering_name kmeans \
        --backbone_checkpoint "$ROOT/$DENSE_BEST" --pretrain_backbone )
  fi
  echo "########## CLUSTER-RETRAIN2 SEED $S DONE $(date +%H:%M:%S) ##########"
done
echo "########## CLUSTER-RETRAIN2 ALL DONE ##########"
