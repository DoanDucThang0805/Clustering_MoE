"""Cập nhật khối `efficientnetb0_timm` trong backbone_generalization.csv (CP6),
theo đúng lựa chọn checkpoint được chỉ định thủ công (không phải best-of-N theo
VAL như `cp6_results.py` mặc định):

  - dense            : LẦN TRAIN THỨ NHẤT của mỗi seed
                        (checkpoints/plantdoc/pretrain_baseline/efficientnetb0_timm)
  - learned_gate_moe : LẦN TRAIN THỨ HAI của mỗi seed
                        (checkpoints/plantdoc/moe_temperature_0.5_pretrain_backbone/
                         efficientnetb0_timm_moe)
  - cluster_moe      : bộ đã khoá "retrain2"
                        (checkpoints/plantdoc/clustering_moe/
                         dense_aligned_pretrain_backbone_retrain2/efficientnetb0_timm_backbone)

Chỉ thay các hàng backbone="efficientnetb0_timm"; hai khối backbone khác
(efficientnetb0 torchvision, mobilenetv3small) trong file giữ nguyên.

Chạy từ src/:
    python -m benchmark.cp6_update_efficientnetb0_timm
"""
from __future__ import annotations

import csv
import glob
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = "plantdoc"
SEEDS = [42, 43, 44, 45, 46]
BACKBONE_LABEL = "efficientnetb0_timm"
OUT = ROOT / "paper_results" / "tables" / "backbone_generalization.csv"
SUMMARY_OUT = (
    ROOT / "paper_results" / "tables"
    / "cp6_efficientnetb0_timm_summary.csv"
)

DENSE_ROOT = ROOT / "checkpoints/plantdoc/pretrain_baseline/efficientnetb0_timm"
MOE_ROOT = (ROOT / "checkpoints/plantdoc/moe_temperature_0.5_pretrain_backbone"
            / "efficientnetb0_timm_moe" / "4_experts" / "top_2")
CLUSTER_ROOT = (ROOT / "checkpoints/plantdoc/clustering_moe/dense_aligned_pretrain_backbone_retrain2"
                / "efficientnetb0_timm_backbone" / "kmeans" / "temperature_0.5" / "G4_cosine_top2")

# params_m giữ nguyên giá trị đã có trong file hiện tại (kiến trúc không đổi theo seed/run)
PARAMS_M = {"dense": 4.0178, "learned_gate_moe": 14.8955, "cluster_moe": 14.8759}
ROUTING = {"dense": "none", "learned_gate_moe": "learned", "cluster_moe": "cosine"}


def _nth_run(seed_dir: Path, index: int) -> Path:
    cands = sorted(glob.glob(str(seed_dir / "run_*" / "best_checkpoint.pth")))
    if len(cands) <= index:
        raise FileNotFoundError(f"Không đủ {index + 1} run dưới {seed_dir} (có {len(cands)})")
    return Path(cands[index])


def _latest_run(seed_dir: Path) -> Path:
    cands = sorted(glob.glob(str(seed_dir / "run_*" / "best_checkpoint.pth")))
    if not cands:
        raise FileNotFoundError(f"No checkpoint under {seed_dir}")
    return Path(cands[-1])


def _metrics(labels, preds) -> dict:
    return dict(
        accuracy=round(float(accuracy_score(labels, preds)), 4),
        macro_f1=round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
        weighted_f1=round(float(f1_score(labels, preds, average="weighted", zero_division=0)), 4),
    )


def eval_dense(seed: int) -> dict:
    from models.pretrain_baseline.model_registry import MODEL_REGISTRY
    from datasets.plantdoc_dataset import test_dataset
    ckpt = torch.load(_nth_run(DENSE_ROOT / f"seed_{seed}", 0), map_location=DEVICE)
    model = MODEL_REGISTRY["efficientnetb0_timm"]
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
    ckpt = torch.load(_nth_run(MOE_ROOT / f"seed_{seed}", 1), map_location=DEVICE)
    sd = ckpt["model_state_dict"]
    model = MoEModel(
        context_dim=ckpt["context_dim"], num_classes=ckpt["num_classes"],
        num_experts=ckpt["num_experts"], top_k=ckpt["top_k"],
        router_mode=ckpt["router_mode"], backbone_name="efficientnetb0_timm",
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
    ckpt = torch.load(_latest_run(CLUSTER_ROOT / f"seed_{seed}"), map_location=DEVICE)
    sd = ckpt["model_state_dict"]
    model = ClusteringMoEModel(
        num_classes=ckpt["num_classes"], centroids=sd["moe_layer.gating.centroids"],
        top_k=ckpt["top_k"], backbone_name="efficientnetb0_timm",
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
FIELDS = ["seed", "dataset", "backbone", "initialization", "model", "routing",
          "G", "top_k", "tau", "accuracy", "macro_f1", "weighted_f1",
          "params_m", "flops_g", "model_size_mb", "peak_cpu_memory_mb", "cpu_latency_ms"]


def main() -> None:
    existing_rows = list(csv.DictReader(open(OUT))) if OUT.exists() else []
    kept_rows = [r for r in existing_rows if r["backbone"] != BACKBONE_LABEL]

    new_rows = []
    for mdl, fn in EVAL_FN.items():
        for seed in SEEDS:
            m = fn(seed)
            new_rows.append({
                "seed": seed, "dataset": DATASET, "backbone": BACKBONE_LABEL,
                "initialization": "imagenet_pretrained", "model": mdl, "routing": ROUTING[mdl],
                "G": 4 if mdl != "dense" else "", "top_k": 2 if mdl != "dense" else "",
                "tau": 0.5 if mdl != "dense" else "", **m,
                "params_m": PARAMS_M[mdl], "flops_g": "",
                "model_size_mb": "", "peak_cpu_memory_mb": "", "cpu_latency_ms": "",
            })
            print(f"[{mdl:16s} seed {seed}] acc={m['accuracy']} mF1={m['macro_f1']} wF1={m['weighted_f1']}")

    # Giữ đúng vị trí khối efficientnetb0_timm trong file gốc (chèn sau efficientnetb0,
    # trước mobilenetv3small) để không đảo lộn bố cục bảng.
    insert_at = next((i for i, r in enumerate(kept_rows) if r["backbone"] == "mobilenetv3small"), len(kept_rows))
    rows = kept_rows[:insert_at] + new_rows + kept_rows[insert_at:]

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {OUT}")

    summary_fields = [
        "dataset", "backbone", "initialization", "model", "routing",
        "G", "top_k", "tau", "checkpoint_protocol", "n_seeds",
        "accuracy_mean_std", "macro_f1_mean_std", "weighted_f1_mean_std",
        "params_m",
    ]
    protocol = {
        "dense": "train_1",
        "learned_gate_moe": "train_2",
        "cluster_moe": "dense_aligned_retrain2",
    }
    summary_rows = []
    for mdl in EVAL_FN:
        model_rows = [r for r in new_rows if r["model"] == mdl]
        summary = {
            "dataset": DATASET,
            "backbone": BACKBONE_LABEL,
            "initialization": "imagenet_pretrained",
            "model": mdl,
            "routing": ROUTING[mdl],
            "G": 4 if mdl != "dense" else "",
            "top_k": 2 if mdl != "dense" else "",
            "tau": 0.5 if mdl != "dense" else "",
            "checkpoint_protocol": protocol[mdl],
            "n_seeds": len(model_rows),
            "params_m": PARAMS_M[mdl],
        }
        for metric in ("accuracy", "macro_f1", "weighted_f1"):
            values = [float(r[metric]) for r in model_rows]
            summary[f"{metric}_mean_std"] = (
                f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"
            )
        summary_rows.append(summary)

    with open(SUMMARY_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields, lineterminator="\n")
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Saved {len(summary_rows)} rows -> {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
