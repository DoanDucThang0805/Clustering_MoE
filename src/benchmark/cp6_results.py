"""
CP6 — Backbone generalization (EfficientNet-B0 trên PlantDoc) → backbone_generalization.csv.

Tính per-seed accuracy/macro_f1/weighted_f1 trên PlantDoc test cho 3 model
(dense · learned-gate MoE · Cluster-MoE) với các backbone B0:
  - efficientnetb0_torchvision  (label "efficientnetb0")
  - efficientnetb0_timm         (label "efficientnetb0_timm")
Gộp thêm hàng MobileNetV3-Small (từ CP1) để so Bảng III.

Schema (PDF IX.C):
  seed,dataset,backbone,initialization,model,routing,G,top_k,tau,
  accuracy,macro_f1,weighted_f1,params_m,flops_g,
  model_size_mb,peak_cpu_memory_mb,cpu_latency_ms

Chạy từ src/:
  python -m benchmark.cp6_results --out ../paper_results/tables/backbone_generalization.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = "plantdoc"
ROUTING = {"dense": "none", "learned_gate_moe": "learned", "cluster_moe": "cosine"}
# (registry backbone key, nhãn CSV)
B0_BACKBONES = [("efficientnetb0_torchvision", "efficientnetb0"),
                ("efficientnetb0_timm", "efficientnetb0_timm")]
MNV3 = {"dense": (1.5261, 0.1229), "learned_gate_moe": (3.4845, 0.1269),
        "cluster_moe": (3.4772, 0.1268)}


def _latest(seed_dir: Path) -> Path:
    """Chọn run có VALIDATION accuracy cao nhất (best-of-N restarts).

    Seed chỉ có 1 run -> trả đúng run đó (hành vi cũ không đổi). TUYỆT ĐỐI không
    chọn theo test (285 ảnh, 1 ảnh = 0.35 điểm -> chọn theo test là chọn nhiễu).
    """
    c = sorted(glob.glob(str(seed_dir / "run_*" / "best_checkpoint.pth")))
    if not c:
        raise FileNotFoundError(f"No checkpoint under {seed_dir}")
    if len(c) == 1:
        return Path(c[0])
    best_path, best_val = None, -float("inf")
    for p in c:
        ck = torch.load(p, map_location="cpu")
        val = max(ck.get("val_acc_history", [-float("inf")]))
        if val > best_val:
            best_path, best_val = Path(p), val
    print(f"    [best-of-{len(c)} theo VAL={best_val:.2f}%] {best_path.parent.name}")
    return best_path


def _params_m(model) -> float:
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 4)


def _metrics(labels, preds) -> dict:
    return dict(
        accuracy=round(float(accuracy_score(labels, preds)), 4),
        macro_f1=round(float(f1_score(labels, preds, average="macro")), 4),
        weighted_f1=round(float(f1_score(labels, preds, average="weighted")), 4),
    )


def eval_dense(seed: int, bb: str):
    from models.pretrain_baseline.model_registry import MODEL_REGISTRY
    from datasets.registry import get_test
    ckpt = torch.load(_latest(
        ROOT / f"checkpoints/{DATASET}/pretrain_baseline/{bb}" / f"seed_{seed}"), map_location=DEVICE)
    sd = ckpt["model_state_dict"]
    model = MODEL_REGISTRY[bb]
    if "timm" in bb:
        ncls = sd["classifier.weight"].shape[0]
        model.reset_classifier(ncls)
    else:
        ncls = sd["classifier.1.weight"].shape[0]
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, ncls)
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    loader = DataLoader(get_test(DATASET), batch_size=64, shuffle=False)
    labs, preds = [], []
    with torch.inference_mode():
        for images, labels in loader:
            preds.extend(torch.argmax(model(images.to(DEVICE)), 1).cpu().tolist())
            labs.extend(labels.tolist())
    return _metrics(labs, preds), _params_m(model)


def eval_moe(seed: int, bb: str):
    from models.moe.model import MoEModel
    from models.moe.gating import ContextAwareLinearGating
    from datasets.registry import get_moe_build
    ckpt = torch.load(_latest(
        ROOT / f"checkpoints/{DATASET}/moe_temperature_0.5_pretrain_backbone/{bb}_moe/4_experts/top_2"
        / f"seed_{seed}"), map_location=DEVICE)
    sd = ckpt["model_state_dict"]
    model = MoEModel(context_dim=ckpt["context_dim"], num_classes=ckpt["num_classes"],
                     num_experts=ckpt["num_experts"], top_k=ckpt["top_k"],
                     router_mode=ckpt["router_mode"], backbone_name=bb,
                     pretrain_backbone=True, temperature=ckpt["temperature"])
    if ("moe_layer.gating.gate_projector.weight" in sd
            and "moe_layer.gating.gate_projector.0.weight" not in sd):
        model.moe_layer.gating = ContextAwareLinearGating(
            model_dim=model.feature_extractor.output_dim, context_dim=ckpt["context_dim"],
            num_experts=ckpt["num_experts"], top_k=ckpt["top_k"], temperature=ckpt["temperature"])
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    _, _, test_ds = get_moe_build(DATASET)(use_context=True)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    rm = ckpt["router_mode"]
    labs, preds = [], []
    with torch.inference_mode():
        for images, labels, context in loader:
            images, context = images.to(DEVICE), context.to(DEVICE)
            logits = model(images, context)[0] if rm == "context_aware" else model(images)[0]
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
            labs.extend(labels.tolist())
    return _metrics(labs, preds), _params_m(model)


def eval_cluster(seed: int, bb: str):
    from models.clustering_moe.model import ClusteringMoEModel
    from datasets.registry import get_test
    ckpt = torch.load(_latest(
        ROOT / f"checkpoints/{DATASET}/clustering_moe/dense_aligned_pretrain_backbone/{bb}_backbone"
        / "kmeans/temperature_0.5/G4_cosine_top2" / f"seed_{seed}"), map_location=DEVICE)
    sd = ckpt["model_state_dict"]
    model = ClusteringMoEModel(num_classes=ckpt["num_classes"], centroids=sd["moe_layer.gating.centroids"],
                               top_k=ckpt["top_k"], backbone_name=bb,
                               metric=ckpt["metric"], pretrain_backbone=True, temperature=ckpt["temperature"])
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    loader = DataLoader(get_test(DATASET), batch_size=64, shuffle=False)
    labs, preds = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits, _, _, _ = model(images.to(DEVICE))
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
            labs.extend(labels.tolist())
    return _metrics(labs, preds), _params_m(model)


FIELDS = ["seed", "dataset", "backbone", "initialization", "model", "routing",
          "G", "top_k", "tau", "accuracy", "macro_f1", "weighted_f1",
          "params_m", "flops_g", "model_size_mb", "peak_cpu_memory_mb", "cpu_latency_ms"]
MODELS = {"dense": eval_dense, "learned_gate_moe": eval_moe, "cluster_moe": eval_cluster}


def _row(seed, backbone, mdl, m, params, flops):
    return {"seed": seed, "dataset": DATASET, "backbone": backbone,
            "initialization": "imagenet_pretrained", "model": mdl, "routing": ROUTING[mdl],
            "G": 4 if mdl != "dense" else "", "top_k": 2 if mdl != "dense" else "",
            "tau": 0.5 if mdl != "dense" else "", **m,
            "params_m": params, "flops_g": flops,
            "model_size_mb": "", "peak_cpu_memory_mb": "", "cpu_latency_ms": ""}


def mobilenetv3_rows(cp1_csv: Path):
    rows = []
    if not cp1_csv.exists():
        return rows
    for r in csv.DictReader(open(cp1_csv)):
        if r["initialization"] != "imagenet_pretrained":
            continue
        mdl = r["model"]
        rows.append(_row(r["seed"], "mobilenetv3small", mdl,
                         {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "weighted_f1": r["weighted_f1"]},
                         MNV3[mdl][0], MNV3[mdl][1]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "paper_results/tables/backbone_generalization.csv")
    ap.add_argument("--cp1_csv", type=Path, default=ROOT / "paper_results/tables/pretrained_backbone_results.csv")
    ap.add_argument("--seeds", type=str, default="42 43 44 45 46")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split()]

    rows = []
    for bb, label in B0_BACKBONES:
        for mdl, fn in MODELS.items():
            for s in seeds:
                try:
                    m, params = fn(s, bb)
                except FileNotFoundError as e:
                    print(f"[SKIP] {label}/{mdl} seed {s}: chưa có checkpoint")
                    continue
                rows.append(_row(s, label, mdl, m, params, ""))
                print(f"[{label:20s} {mdl:16s} seed {s}] acc={m['accuracy']} mF1={m['macro_f1']} params={params}M")

    rows += mobilenetv3_rows(args.cp1_csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
