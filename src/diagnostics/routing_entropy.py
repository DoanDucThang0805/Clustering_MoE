"""
CP2 — Routing entropy diagnostics.

Tính routing entropy (H_i), mean/std/normalized entropy, expert usage và usage CV
trên test split cho learned-gate MoE và Cluster-MoE, từ checkpoint có sẵn
(KHÔNG retrain). Xuất routing_entropy.csv theo schema tài liệu bổ sung (CP2).

Công thức (PDF eq. 2-3):
    H_i      = -sum_{g in K_i} alpha_{i,g} * log(alpha_{i,g} + eps)      eps = 1e-9
    H_bar    = mean(H_i)
    H_norm   = H_bar / log(top_k)
Trong đó alpha_{i,g} là trọng số softmax trên top-k expert (sum = 1 mỗi mẫu).

Cách chạy (từ src/):
    python -m diagnostics.routing_entropy --model_type cluster_moe \
        --seeds "42 43 44 45 46 47 48 49 50 51" --split test \
        --output_csv ../mean_acc_mF1_results/routing_entropy.csv
    python -m diagnostics.routing_entropy --model_type moe \
        --seeds "42 43 44 45 46 47 48 49 50 51" --split test --pretrain_backbone \
        --output_csv ../mean_acc_mF1_results/routing_entropy.csv   # append
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

ROOT_DIR = Path(__file__).resolve().parents[2]
EPS = 1e-9


# ─────────────────────────────────────────────────────────────
# Checkpoint path resolution
# ─────────────────────────────────────────────────────────────
def _fmt_tau(tau: float) -> str:
    """0.5 -> '0.5', 1.0 -> '1.0' (khớp tên thư mục checkpoint)."""
    return f"{tau:.1f}"


def resolve_checkpoint(model_type: str, seed: int, tau: float = 0.5,
                       moe_type_model: str | None = None) -> Path:
    """Trả về best_checkpoint.pth mới nhất cho seed (bỏ qua file lạ như CSV)."""
    tau_s = _fmt_tau(tau)
    if model_type == "cluster_moe":
        seed_dir = (
            ROOT_DIR
            / "checkpoints/plantdoc/clustering_moe/dense_aligned_pretrain_backbone"
            / "mobilenetv3small_torchvision_backbone/kmeans" / f"temperature_{tau_s}"
            / "G4_cosine_top2" / f"seed_{seed}"
        )
    elif model_type == "moe":
        if moe_type_model:
            type_model = moe_type_model.replace("{tau}", tau_s)
        elif tau_s == "0.5":
            type_model = "moe_temperature_0.5_pretrain_backbone"
        else:
            type_model = f"moe_temperature_{tau_s}_pretrain_backbone"
        seed_dir = (
            ROOT_DIR
            / "checkpoints/plantdoc" / type_model
            / "mobilenetv3small_torchvision_moe/4_experts/top_2" / f"seed_{seed}"
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    candidates = sorted(glob.glob(str(seed_dir / "run_*" / "best_checkpoint.pth")))
    if not candidates:
        raise FileNotFoundError(f"No best_checkpoint.pth under {seed_dir}")
    return Path(candidates[-1])


# ─────────────────────────────────────────────────────────────
# Entropy / usage helpers
# ─────────────────────────────────────────────────────────────
def entropy_from_weights(weights: torch.Tensor) -> torch.Tensor:
    """H_i cho từng mẫu từ alpha top-k [B, top_k]  ->  [B]."""
    w = weights.clamp_min(0.0)
    return -(w * torch.log(w + EPS)).sum(dim=1)


def summarize(
    all_entropy: list[float],
    expert_counts: np.ndarray,
    num_samples: int,
    top_k: int,
    num_experts: int,
    labels: list[int],
    preds: list[int],
) -> dict:
    ent = np.asarray(all_entropy, dtype=np.float64)
    mean_entropy = float(ent.mean())
    std_entropy = float(ent.std())
    normalized_entropy = mean_entropy / math.log(top_k)

    # usage_i = count_i / N  ->  sum = top_k (theo sanity check PDF)
    usage = expert_counts.astype(np.float64) / max(num_samples, 1)
    usage_cv = float(usage.std() / usage.mean()) if usage.mean() > 0 else 0.0

    row = {
        "mean_entropy": mean_entropy,
        "std_entropy": std_entropy,
        "normalized_entropy": normalized_entropy,
        "usage_cv": usage_cv,
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }
    for i in range(4):
        row[f"expert_usage_{i + 1}"] = float(usage[i]) if i < num_experts else 0.0
    return row


# ─────────────────────────────────────────────────────────────
# Cluster-MoE
# ─────────────────────────────────────────────────────────────
def run_cluster_moe(seed: int, split: str, backbone_name: str,
                    pretrain_backbone: bool, batch_size: int, device: torch.device,
                    tau: float = 0.5) -> dict:
    from models.clustering_moe.model import ClusteringMoEModel
    from datasets.plantdoc_dataset import train_dataset, validation_dataset, test_dataset

    ckpt_path = resolve_checkpoint("cluster_moe", seed, tau)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state_dict"]
    centroids = sd["moe_layer.gating.centroids"]  # (G, D) — lấy trực tiếp từ state_dict

    model = ClusteringMoEModel(
        num_classes=ckpt["num_classes"],
        centroids=centroids,
        top_k=ckpt["top_k"],
        backbone_name=backbone_name,
        metric=ckpt["metric"],
        pretrain_backbone=pretrain_backbone,
        temperature=ckpt["temperature"],
    )
    model.load_state_dict(sd)
    model.to(device).eval()

    top_k = ckpt["top_k"]
    num_experts = ckpt["num_experts"]
    dataset = {"train": train_dataset, "validation": validation_dataset, "test": test_dataset}[split]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_entropy: list[float] = []
    expert_counts = np.zeros(num_experts, dtype=np.int64)
    labels_all: list[int] = []
    preds_all: list[int] = []
    n = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            logits, weights, top_indices, _ = model(images)   # weights [B, top_k] = alpha
            all_entropy.extend(entropy_from_weights(weights.cpu()).tolist())
            expert_counts += np.bincount(
                top_indices.cpu().reshape(-1).numpy(), minlength=num_experts
            )
            preds_all.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels_all.extend(labels.tolist())
            n += images.size(0)

    row = summarize(all_entropy, expert_counts, n, top_k, num_experts, labels_all, preds_all)
    row.update(seed=seed, model="cluster_moe", routing="cosine", G=num_experts,
               top_k=top_k, tau=ckpt["temperature"], split=split)
    print(f"[cluster_moe seed={seed}] Hbar={row['mean_entropy']:.4f} "
          f"Hnorm={row['normalized_entropy']:.4f} acc={row['accuracy']:.4f} "
          f"mf1={row['macro_f1']:.4f}  (ckpt {ckpt_path.parent.name})")
    return row


# ─────────────────────────────────────────────────────────────
# Learned-gate MoE
# ─────────────────────────────────────────────────────────────
def run_moe(seed: int, split: str, backbone_name: str,
            pretrain_backbone: bool, batch_size: int, device: torch.device,
            tau: float = 0.5, moe_type_model: str | None = None) -> dict:
    from models.moe.model import MoEModel
    from models.moe.gating import ContextAwareLinearGating
    from datasets.plantdoc_dataset_moe import build_datasets

    ckpt_path = resolve_checkpoint("moe", seed, tau, moe_type_model)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state_dict"]

    model = MoEModel(
        context_dim=ckpt["context_dim"],
        num_classes=ckpt["num_classes"],
        num_experts=ckpt["num_experts"],
        top_k=ckpt["top_k"],
        router_mode=ckpt["router_mode"],
        backbone_name=backbone_name,
        pretrain_backbone=pretrain_backbone,
        temperature=ckpt["temperature"],
    )
    # Gating-type fix: checkpoint dùng ContextAwareLinearGating (gate_projector là Linear đơn)
    uses_linear = (
        "moe_layer.gating.gate_projector.weight" in sd
        and "moe_layer.gating.gate_projector.0.weight" not in sd
    )
    if uses_linear:
        model.moe_layer.gating = ContextAwareLinearGating(
            model_dim=model.feature_extractor.output_dim,
            context_dim=ckpt["context_dim"],
            num_experts=ckpt["num_experts"],
            top_k=ckpt["top_k"],
            temperature=ckpt["temperature"],
        )
    model.load_state_dict(sd)
    model.to(device).eval()  # eval → tắt noise trong gating

    top_k = ckpt["top_k"]
    num_experts = ckpt["num_experts"]
    router_mode = ckpt["router_mode"]
    train_ds, val_ds, test_ds = build_datasets(use_context=True)
    dataset = {"train": train_ds, "validation": val_ds, "test": test_ds}[split]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_entropy: list[float] = []
    expert_counts = np.zeros(num_experts, dtype=np.int64)
    labels_all: list[int] = []
    preds_all: list[int] = []
    n = 0

    with torch.inference_mode():
        for images, labels, context in loader:
            images = images.to(device)
            context = context.to(device)

            # α phải lấy bằng cách chạy lại gating (forward không trả α)
            feat = model.feature_extractor(images)
            feat_norm = model.pre_moe_norm(feat)
            if router_mode == "context_aware":
                weights, top_indices, _ = model.moe_layer.gating(feat_norm, context)
                logits = model(images, context)[0]
            else:
                weights, top_indices, _ = model.moe_layer.gating(feat_norm)
                logits = model(images)[0]

            all_entropy.extend(entropy_from_weights(weights.cpu()).tolist())
            expert_counts += np.bincount(
                top_indices.cpu().reshape(-1).numpy(), minlength=num_experts
            )
            preds_all.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels_all.extend(labels.tolist())
            n += images.size(0)

    row = summarize(all_entropy, expert_counts, n, top_k, num_experts, labels_all, preds_all)
    row.update(seed=seed, model="moe", routing="learned", G=num_experts,
               top_k=top_k, tau=ckpt["temperature"], split=split)
    print(f"[moe seed={seed}] Hbar={row['mean_entropy']:.4f} "
          f"Hnorm={row['normalized_entropy']:.4f} acc={row['accuracy']:.4f} "
          f"mf1={row['macro_f1']:.4f}  (ckpt {ckpt_path.parent.name})")
    return row


# ─────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "seed", "model", "routing", "G", "top_k", "tau", "split",
    "mean_entropy", "std_entropy", "normalized_entropy",
    "expert_usage_1", "expert_usage_2", "expert_usage_3", "expert_usage_4",
    "usage_cv", "accuracy", "macro_f1",
]


def append_rows(output_csv: Path, rows: list[dict]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists()
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CSV_FIELDS})


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="CP2 routing entropy diagnostics")
    p.add_argument("--model_type", required=True, choices=["cluster_moe", "moe"])
    p.add_argument("--seeds", type=str, default="42 43 44 45 46 47 48 49 50 51")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "validation", "test"])
    p.add_argument("--backbone_name", type=str, default="mobilenetv3small_torchvision")
    p.add_argument("--pretrain_backbone", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--tau", type=float, default=0.5,
                   help="Temperature namespace của checkpoint (0.3/0.5/0.7/1.0)")
    p.add_argument("--moe_type_model", type=str, default=None,
                   help="Ghi đè namespace learned-gate; dùng '{tau}' làm placeholder, "
                        "vd 'moe_taucurve_temperature_{tau}_pretrain_backbone'")
    p.add_argument("--output_csv", type=Path, required=True)
    return p.parse_args()


def main():
    args = get_args()
    seeds = [int(s) for s in args.seeds.split()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = run_cluster_moe if args.model_type == "cluster_moe" else run_moe

    rows = []
    for seed in seeds:
        kwargs = dict(
            seed=seed, split=args.split, backbone_name=args.backbone_name,
            pretrain_backbone=args.pretrain_backbone,
            batch_size=args.batch_size, device=device, tau=args.tau,
        )
        if args.model_type == "moe":
            kwargs["moe_type_model"] = args.moe_type_model
        rows.append(runner(**kwargs))
    append_rows(args.output_csv, rows)
    print(f"\nSaved {len(rows)} rows ({args.model_type}) → {args.output_csv}")


if __name__ == "__main__":
    main()
