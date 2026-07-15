#!/usr/bin/env bash
# B0-torchvision FROM SCRATCH (non-pretrain): dense + learned MoE + Cluster-MoE, 5 seed tuần tự.
# Mirror pipeline CP6 pretrained, chỉ khác initialization. Idempotent.
# Namespace: non_pretrain_baseline / moe_temperature_0.5 / dense_aligned_non_pretrain_backbone
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[ -d venv ] && source venv/bin/activate

DS="plantdoc"
BB="efficientnetb0_torchvision"
EMB_NS="non_pretrain_backbone"                     # embeddings + centroids (from-scratch dense)
CLU_NS="dense_aligned_non_pretrain_backbone"       # checkpoints cluster

have() { ls $1 >/dev/null 2>&1; }
latest() { ls -td $1 2>/dev/null | head -1; }

for S in 42 43 44 45 46; do
  echo "########## B0TV-SCRATCH SEED $S START $(date +%H:%M:%S) ##########"

  # 1) Dense B0-tv from scratch
  DENSE_DIR="checkpoints/$DS/non_pretrain_baseline/$BB/seed_$S"
  if have "$DENSE_DIR/run_*/best_checkpoint.pth"; then echo "[skip] dense scratch seed $S"; else
    echo ">> dense B0-tv scratch seed $S"
    ( cd src && python -m training.efficientnetb0_scratch --seed "$S" --dataset_name "$DS" )
  fi
  DENSE_BEST=$(latest "$DENSE_DIR/run_*")/best_checkpoint.pth
  DENSE_RUN=$(basename "$(dirname "$DENSE_BEST")")
  echo "   dense_run=$DENSE_RUN"

  # 2) Learned-gate MoE from scratch (--pretrain_backbone là store_false => TRUYỀN flag = TẮT pretrained)
  MOE_DIR="checkpoints/$DS/moe_temperature_0.5/${BB}_moe/4_experts/top_2/seed_$S"
  if have "$MOE_DIR/run_*/best_checkpoint.pth"; then echo "[skip] MoE scratch seed $S"; else
    echo ">> MoE B0-tv scratch seed $S"
    ( cd src && python -m training.moe --seed "$S" --dataset_name "$DS" \
        --type_model moe_temperature_0.5 --num_experts 4 --top_k 2 \
        --num_epochs 400 --batch_size 64 --lr 0.001 --weight_decay 0.001 --moe_alpha 0.05 \
        --temperature 0.5 --router_mode context_aware --context_dim 6 \
        --backbone_name "$BB" --pretrain_backbone )
  fi

  # 3-4) Extract embedding (từ dense scratch) + KMeans fit train
  CENT="clustering_results/$DS/$EMB_NS/${BB}_backbone/kmeans/cosine/seed_$S/clusters_kmeans_G4_seed$S.npz"
  if [ -f "$CENT" ]; then echo "[skip] centroid scratch seed $S"; else
    echo ">> extract embedding (all splits) seed $S từ $DENSE_RUN"
    bash extract_embedding.sh --dataset_name "$DS" --model_name "$BB" --type_model non_pretrain_baseline \
        --type_backbone "$EMB_NS" --seed "$S" --run_time "$DENSE_RUN" --split all
    echo ">> kmeans G4 cosine seed $S (fit train)"
    ( cd src && python -m training.kmean --dataset_name "$DS" --backbone_type "$EMB_NS" \
        --backbone_name "$BB" --seed "$S" --num_clusters 4 --metric cosine --feature_split train )
  fi

  # 5) Cluster-MoE dense-aligned from scratch (KHÔNG --pretrain_backbone: store_true => bỏ = False)
  CLU="checkpoints/$DS/clustering_moe/$CLU_NS/${BB}_backbone/kmeans/temperature_0.5/G4_cosine_top2/seed_$S"
  if have "$CLU/run_*/best_checkpoint.pth"; then echo "[skip] Cluster scratch seed $S"; else
    echo ">> Cluster-MoE B0-tv scratch seed $S (backbone=$DENSE_BEST)"
    ( cd src && python -m training.clustering_moe --seed "$S" --dataset_name "$DS" \
        --num_experts 4 --top_k 2 --distance_metric cosine --temperature 0.5 \
        --lr 3e-4 --weight_decay 1e-2 --label_smoothing 0.05 \
        --num_epochs 400 --batch_size 32 \
        --backbone_type "$CLU_NS" \
        --centroid_backbone_type "$EMB_NS" \
        --backbone_name "$BB" --model_clustering_name kmeans \
        --backbone_checkpoint "$ROOT/$DENSE_BEST" )
  fi

  echo "########## B0TV-SCRATCH SEED $S DONE $(date +%H:%M:%S) ##########"
done
echo "########## B0TV-SCRATCH ALL DONE ##########"
