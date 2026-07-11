"""
CP3 — Expert masking ablation + conditional confusion (Cluster-MoE).

Can thiệp tại inference (KHÔNG retrain, KHÔNG sửa file model):
tắt từng expert g, renormalize trọng số top-k còn lại theo PDF eq. 4:
    alpha_tilde_{i,j} = alpha_{i,j} * 1(j != g) / (sum_{l in K_i, l != g} alpha_{i,l} + eps)
Đo suy giảm Accuracy / Macro-F1 / Weighted-F1 + per-class F1.
Kèm conditional confusion theo primary expert (= top_indices[:,0]).

Tái dựng đúng forward của ClusteringMoEModel:
    embedding -> moe_layer(gating + expert mix) -> residual + norm -> classifier.

Xuất 2 CSV đúng schema PDF mục VI.B:
    expert_masking_ablation.csv, conditional_confusion_summary.csv

Chạy từ src/:
    python -m diagnostics.expert_masking --seeds "42 43 44 45 46 47 48 49 50 51" \
        --out_masking ../paper_results/tables/expert_masking_ablation.csv \
        --out_confusion ../paper_results/tables/conditional_confusion_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, recall_score, confusion_matrix)

ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-9
NUM_CLASSES = 8
LABELS = list(range(NUM_CLASSES))


def resolve_cluster_ckpt(seed: int) -> Path:
    seed_dir = (ROOT / "checkpoints/plantdoc/clustering_moe/dense_aligned_pretrain_backbone"
                / "mobilenetv3small_torchvision_backbone/kmeans/temperature_0.5"
                / "G4_cosine_top2" / f"seed_{seed}")
    cands = sorted(glob.glob(str(seed_dir / "run_*" / "best_checkpoint.pth")))
    if not cands:
        raise FileNotFoundError(f"No best_checkpoint.pth under {seed_dir}")
    return Path(cands[-1])


def load_model(seed: int):
    from models.clustering_moe.model import ClusteringMoEModel
    ckpt = torch.load(resolve_cluster_ckpt(seed), map_location=DEVICE)
    sd = ckpt["model_state_dict"]
    model = ClusteringMoEModel(
        num_classes=ckpt["num_classes"], centroids=sd["moe_layer.gating.centroids"],
        top_k=ckpt["top_k"], backbone_name="mobilenetv3small_torchvision",
        metric=ckpt["metric"], pretrain_backbone=True, temperature=ckpt["temperature"],
    )
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    return model, ckpt


@torch.inference_mode()
def cache_embeddings(model):
    """Chạy backbone 1 lần trên test set -> embeddings [N,D], labels [N]."""
    from datasets.plantdoc_dataset import test_dataset
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    embs, labs = [], []
    for images, labels in loader:
        embs.append(model.backbone(images.to(DEVICE)))
        labs.append(labels)
    return torch.cat(embs), torch.cat(labs).numpy()


@torch.inference_mode()
def masked_expert_mix(model, emb, weights, top_indices):
    """Tái dựng MoELayer.forward với weights (đã áp mask)."""
    layer = model.moe_layer
    output = torch.zeros_like(emb)
    for k_idx in range(layer.top_k):
        expert_ids = top_indices[:, k_idx]
        expert_w = weights[:, k_idx].unsqueeze(1)
        expert_out = torch.zeros_like(emb)
        for e in range(layer.num_experts):
            m = expert_ids == e
            if m.any():
                expert_out[m] = layer.experts[e](emb[m])
        output += expert_w * expert_out
    return output


@torch.inference_mode()
def predict(model, emb, weights, top_indices):
    moe_out = masked_expert_mix(model, emb, weights, top_indices)
    normalized = model.norm(emb + moe_out)
    logits = model.classifier(normalized)
    return torch.argmax(logits, dim=1).cpu().numpy()


def apply_mask(weights: torch.Tensor, top_indices: torch.Tensor, g: int | None):
    """eq. 4: zero expert g trong top-k rồi renormalize theo hàng."""
    if g is None:
        return weights
    w = weights.clone()
    w[top_indices == g] = 0.0
    w = w / (w.sum(dim=1, keepdim=True) + EPS)
    return w


# ── metrics ──────────────────────────────────────────────────
def masking_metrics(labels, preds) -> dict:
    per_f1 = f1_score(labels, preds, average=None, labels=LABELS, zero_division=0)
    row = dict(
        accuracy=float(accuracy_score(labels, preds)),
        macro_f1=float(f1_score(labels, preds, average="macro", labels=LABELS, zero_division=0)),
        weighted_f1=float(f1_score(labels, preds, average="weighted", labels=LABELS, zero_division=0)),
    )
    for c in LABELS:
        row[f"class_{c}_f1"] = float(per_f1[c])
    return row


def confusion_group_metrics(labels, preds) -> dict:
    per_rec = recall_score(labels, preds, average=None, labels=LABELS, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=LABELS)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    idx = int(off.argmax())
    ti, pj = divmod(idx, NUM_CLASSES)
    cnt = int(off[ti, pj])
    row = dict(
        num_samples=len(labels),
        accuracy=float(accuracy_score(labels, preds)),
        macro_f1=float(f1_score(labels, preds, average="macro", labels=LABELS, zero_division=0)),
        most_confused_pair=(f"{ti}->{pj}" if cnt > 0 else "none"),
        most_confused_count=cnt,
    )
    for c in LABELS:
        row[f"class_{c}_recall"] = float(per_rec[c])
    return row


MASK_FIELDS = (["seed", "model", "routing", "G", "top_k", "tau", "mask_expert",
                "accuracy", "macro_f1", "weighted_f1",
                "delta_accuracy", "delta_macro_f1", "delta_weighted_f1"]
               + [f"class_{c}_f1" for c in LABELS])

CONF_FIELDS = (["seed", "model", "routing", "G", "top_k", "tau", "primary_expert",
                "num_samples", "accuracy", "macro_f1",
                "most_confused_pair", "most_confused_count"]
               + [f"class_{c}_recall" for c in LABELS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42 43 44 45 46 47 48 49 50 51")
    ap.add_argument("--out_masking", type=Path, required=True)
    ap.add_argument("--out_confusion", type=Path, required=True)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split()]

    mask_rows, conf_rows = [], []
    for seed in seeds:
        model, ckpt = load_model(seed)
        G, top_k, tau = ckpt["num_experts"], ckpt["top_k"], ckpt["temperature"]
        emb, labels = cache_embeddings(model)
        weights, top_indices, _ = model.moe_layer.gating(emb)

        # ── masking ablation ──
        base = masking_metrics(labels, predict(model, emb, weights, top_indices))
        for g in [None, 0, 1, 2, 3]:
            w = apply_mask(weights, top_indices, g)
            m = masking_metrics(labels, predict(model, emb, w, top_indices))
            row = {"seed": seed, "model": "cluster_moe", "routing": "cosine",
                   "G": G, "top_k": top_k, "tau": tau,
                   "mask_expert": "none" if g is None else g,
                   "delta_accuracy": round(m["accuracy"] - base["accuracy"], 4),
                   "delta_macro_f1": round(m["macro_f1"] - base["macro_f1"], 4),
                   "delta_weighted_f1": round(m["weighted_f1"] - base["weighted_f1"], 4)}
            for k in ["accuracy", "macro_f1", "weighted_f1"] + [f"class_{c}_f1" for c in LABELS]:
                row[k] = round(m[k], 4)
            mask_rows.append(row)
        print(f"[seed {seed}] base acc={base['accuracy']:.4f} mF1={base['macro_f1']:.4f}  "
              f"| Δacc mask0..3 = "
              + ", ".join(f"{mask_rows[-4+g]['delta_accuracy']:+.3f}" for g in range(4)))

        # ── conditional confusion theo primary expert (unmasked) ──
        primary = top_indices[:, 0].cpu().numpy()
        preds_base = predict(model, emb, weights, top_indices)
        for e in range(G):
            sel = primary == e
            if sel.sum() == 0:
                continue
            cm = confusion_group_metrics(labels[sel], preds_base[sel])
            crow = {"seed": seed, "model": "cluster_moe", "routing": "cosine",
                    "G": G, "top_k": top_k, "tau": tau, "primary_expert": e}
            crow.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in cm.items()})
            conf_rows.append(crow)

    for path, fields, rows in [(args.out_masking, MASK_FIELDS, mask_rows),
                               (args.out_confusion, CONF_FIELDS, conf_rows)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
