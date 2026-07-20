"""CP4 — đo lại paired statistics bằng inference trực tiếp trên 3 checkpoint
chỉ định (KHÔNG đọc lại từ pretrained_backbone_results.csv của CP1):

    dense            : checkpoints/plantdoc/pretrain_baseline/
                        mobilenetv3small_torchvision_retrain2
    learned_gate_moe : checkpoints/plantdoc/moe_temperature_0.5_pretrain_backbone/
                        mobilenetv3small_torchvision_moe
    cluster_moe      : checkpoints/plantdoc/clustering_moe/
                        dense_aligned_pretrain_backbone/
                        mobilenetv3small_torchvision_backbone/kmeans/
                        temperature_0.5/G4_cosine_top2

10 seed (42-51), inference thật trên PlantDoc test set. Mỗi seed dùng file
``best_checkpoint.pth`` trong đúng checkpoint root đã khóa ở trên. Holm và BH
được áp dụng trên toàn bộ sáu phép kiểm định (3 cặp model x 2 metric).

Chỉ xuất một bảng reviewer-facing:
    paper_results/tables/cp4_paired_tests_summary.csv

Chạy từ src/:
    python -m statistical_test.cp4_direct_inference
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from torchvision.models import mobilenet_v3_small

from statistical_test.paired_checkpoint_test import (
    holm_bonferroni, benjamini_hochberg, paired_tests_free,
)

ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = list(range(42, 52))
NUM_CLASSES = 8

DENSE_ROOT = ROOT / "checkpoints/plantdoc/pretrain_baseline/mobilenetv3small_torchvision_retrain2"
MOE_ROOT = (ROOT / "checkpoints/plantdoc/moe_temperature_0.5_pretrain_backbone"
            / "mobilenetv3small_torchvision_moe" / "4_experts" / "top_2")
CLUSTER_ROOT = (ROOT / "checkpoints/plantdoc/clustering_moe/dense_aligned_pretrain_backbone"
                / "mobilenetv3small_torchvision_backbone" / "kmeans" / "temperature_0.5"
                / "G4_cosine_top2")

DISPLAY_NAME = {
    "cluster_moe": "Cluster-MoE",
    "learned_gate_moe": "Learned-gate MoE",
    "dense": "MobileNetV3-Small baseline",
}
METRIC_LABEL = {"accuracy": "Accuracy", "macro_f1": "Macro-F1"}
CP4_PAIRS = (("cluster_moe", "dense"), ("cluster_moe", "learned_gate_moe"),
             ("learned_gate_moe", "dense"))
CP4_METRICS = ("accuracy", "macro_f1")


def _best_checkpoint(seed_dir: Path) -> Path:
    cands = sorted(seed_dir.glob("run_*/best_checkpoint.pth"))
    if not cands:
        raise FileNotFoundError(f"No checkpoint under {seed_dir}")
    if len(cands) != 1:
        raise RuntimeError(
            f"Expected one locked best checkpoint under {seed_dir}, found {len(cands)}"
        )
    return cands[0]


def _metrics(labels, preds) -> dict:
    return dict(
        accuracy=float(accuracy_score(labels, preds)),
        macro_f1=float(f1_score(labels, preds, average="macro", zero_division=0)),
    )


def eval_dense(seed: int) -> dict:
    from datasets.plantdoc_dataset import test_dataset
    ckpt = torch.load(_best_checkpoint(DENSE_ROOT / f"seed_{seed}"), map_location=DEVICE)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    labs, preds = [], []
    with torch.inference_mode():
        for images, labels in loader:
            preds.extend(torch.argmax(model(images.to(DEVICE)), 1).cpu().tolist())
            labs.extend(labels.tolist())
    return _metrics(labs, preds)


def eval_moe(seed: int) -> dict:
    from models.moe.model import MoEModel
    from models.moe.gating import ContextAwareLinearGating
    from datasets.plantdoc_dataset_moe import build_datasets
    ckpt = torch.load(_best_checkpoint(MOE_ROOT / f"seed_{seed}"), map_location=DEVICE)
    sd = ckpt["model_state_dict"]
    model = MoEModel(
        context_dim=ckpt["context_dim"], num_classes=ckpt["num_classes"],
        num_experts=ckpt["num_experts"], top_k=ckpt["top_k"],
        router_mode=ckpt["router_mode"], backbone_name="mobilenetv3small_torchvision",
        pretrain_backbone=True, temperature=ckpt["temperature"],
    )
    if ("moe_layer.gating.gate_projector.weight" in sd
            and "moe_layer.gating.gate_projector.0.weight" not in sd):
        model.moe_layer.gating = ContextAwareLinearGating(
            model_dim=model.feature_extractor.output_dim, context_dim=ckpt["context_dim"],
            num_experts=ckpt["num_experts"], top_k=ckpt["top_k"], temperature=ckpt["temperature"],
        )
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    _, _, test_ds = build_datasets(use_context=True)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    rm = ckpt["router_mode"]
    labs, preds = [], []
    with torch.inference_mode():
        for images, labels, context in loader:
            images, context = images.to(DEVICE), context.to(DEVICE)
            logits = model(images, context)[0] if rm == "context_aware" else model(images)[0]
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
            labs.extend(labels.tolist())
    return _metrics(labs, preds)


def eval_cluster(seed: int) -> dict:
    from models.clustering_moe.model import ClusteringMoEModel
    from datasets.plantdoc_dataset import test_dataset
    ckpt = torch.load(_best_checkpoint(CLUSTER_ROOT / f"seed_{seed}"), map_location=DEVICE)
    sd = ckpt["model_state_dict"]
    model = ClusteringMoEModel(
        num_classes=ckpt["num_classes"], centroids=sd["moe_layer.gating.centroids"],
        top_k=ckpt["top_k"], backbone_name="mobilenetv3small_torchvision",
        metric=ckpt["metric"], pretrain_backbone=True, temperature=ckpt["temperature"],
    )
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    labs, preds = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits, _, _, _ = model(images.to(DEVICE))
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
            labs.extend(labels.tolist())
    return _metrics(labs, preds)


EVAL_FN = {"dense": eval_dense, "learned_gate_moe": eval_moe, "cluster_moe": eval_cluster}


def main() -> None:
    data: dict[str, dict[int, dict[str, float]]] = {m: {} for m in EVAL_FN}
    for model_name, fn in EVAL_FN.items():
        print(f"\n[{model_name}]")
        for seed in SEEDS:
            m = fn(seed)
            data[model_name][seed] = m
            print(f"  seed {seed}: accuracy={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")

    out_dir = ROOT / "paper_results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Paired stats; Holm/BH trên toàn bộ 6 test ──

    recs = []
    for metric in CP4_METRICS:
        for a, b in CP4_PAIRS:
            xa = np.array([data[a][s][metric] for s in SEEDS])
            xb = np.array([data[b][s][metric] for s in SEEDS])
            mean_delta, t_p, w_p = paired_tests_free(xa, xb)
            recs.append(dict(
                metric=metric, a=a, b=b, mean_diff=mean_delta,
                t_p=t_p, w_p=w_p,
            ))

    holm = holm_bonferroni([r["t_p"] for r in recs])
    bh = benjamini_hochberg([r["t_p"] for r in recs])

    table_fields = [
        "Dataset", "Comparison", "Metric", "Mean Δ",
        "p_t", "p_w", "p_Holm", "p_BH",
    ]
    table_rows = []
    for r, hp, bp in zip(recs, holm, bh):
        table_rows.append({
            "Dataset": "PlantDoc",
            "Comparison": f"{DISPLAY_NAME[r['a']]} vs {DISPLAY_NAME[r['b']]}",
            "Metric": METRIC_LABEL[r["metric"]],
            "Mean Δ": f"{r['mean_diff']:.4f}",
            "p_t": f"{r['t_p']:.4f}", "p_w": f"{r['w_p']:.4f}",
            "p_Holm": f"{hp:.4f}", "p_BH": f"{bp:.4f}",
        })

    table_path = out_dir / "cp4_paired_tests_summary.csv"
    with open(table_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=table_fields)
        w.writeheader()
        w.writerows(table_rows)
    print(f"Saved -> {table_path}")


if __name__ == "__main__":
    main()
