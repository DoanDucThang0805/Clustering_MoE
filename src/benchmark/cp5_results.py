"""
CP5 — Cross-dataset results (PlantVillage) → cross_dataset_results.csv (PDF VIII.C).

Tính per-seed accuracy/macro_f1/weighted_f1 trên PlantVillage test cho 3 model:
  dense (lamb1k) · learned-gate MoE (torchvision) · Cluster-MoE cosine (torchvision).
Gộp thêm hàng PlantDoc (từ CP1 pretrained_backbone_results.csv) để so sánh 2 dataset.

Schema (PDF VIII.C):
  seed,dataset,num_classes,num_train,num_val,num_test,
  backbone,initialization,model,routing,G,top_k,tau,
  accuracy,macro_f1,weighted_f1,params_m,flops_g,latency_ms

Chạy từ src/:
  python -m benchmark.cp5_results --out ../paper_results/tables/cross_dataset_results.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 43, 44, 45, 46]

# PlantVillage split counts (stratified 80/10/10) + num_classes
PV = dict(dataset="plantvillage", num_classes=10, num_train=12808, num_val=1601, num_test=1602)
# PlantDoc (CP1) — để gộp hàng so sánh
PD = dict(dataset="plantdoc", num_classes=8, num_train=2274, num_val=284, num_test=285)

# flops_g/latency_ms ~ arch-dependent (backbone giống PlantDoc); params_m tính từ model.
ARCH = {
    "dense":            dict(flops_g=0.1229, latency_ms=6.2622),   # mobilenetv3-small
    "learned_gate_moe": dict(flops_g=0.1269, latency_ms=8.2498),
    "cluster_moe":      dict(flops_g=0.1268, latency_ms=7.9588),
}
ROUTING = {"dense": "none", "learned_gate_moe": "learned", "cluster_moe": "cosine"}


def _latest(seed_dir: Path) -> Path:
    c = sorted(glob.glob(str(seed_dir / "run_*" / "best_checkpoint.pth")))
    if not c:
        raise FileNotFoundError(f"No checkpoint under {seed_dir}")
    return Path(c[-1])


def _params_m(model) -> float:
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 4)


def _metrics(labels, preds) -> dict:
    return dict(
        accuracy=round(float(accuracy_score(labels, preds)), 4),
        macro_f1=round(float(f1_score(labels, preds, average="macro")), 4),
        weighted_f1=round(float(f1_score(labels, preds, average="weighted")), 4),
    )


# ── dense lamb1k ──────────────────────────────────────────────
def eval_dense(seed: int):
    from models.pretrain_baseline.model_registry import MODEL_REGISTRY
    from datasets.registry import get_test
    ckpt = torch.load(_latest(
        ROOT / "checkpoints/plantvillage/pretrain_baseline/mobilenetv3small_timm_lamb1k" / f"seed_{seed}"),
        map_location=DEVICE)
    sd = ckpt["model_state_dict"]
    model = MODEL_REGISTRY["mobilenetv3small_timm_lamb1k"]
    model.reset_classifier(PV["num_classes"])
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    loader = DataLoader(get_test("plantvillage"), batch_size=64, shuffle=False)
    labs, preds = [], []
    with torch.inference_mode():
        for images, labels in loader:
            preds.extend(torch.argmax(model(images.to(DEVICE)), 1).cpu().tolist())
            labs.extend(labels.tolist())
    return _metrics(labs, preds), _params_m(model)


# ── learned-gate MoE (torchvision) ────────────────────────────
def eval_moe(seed: int):
    from models.moe.model import MoEModel
    from models.moe.gating import ContextAwareLinearGating
    from datasets.registry import get_moe_build
    ckpt = torch.load(_latest(
        ROOT / "checkpoints/plantvillage/moe_temperature_0.5_pretrain_backbone"
        / "mobilenetv3small_torchvision_moe/4_experts/top_2" / f"seed_{seed}"),
        map_location=DEVICE)
    sd = ckpt["model_state_dict"]
    model = MoEModel(context_dim=ckpt["context_dim"], num_classes=ckpt["num_classes"],
                     num_experts=ckpt["num_experts"], top_k=ckpt["top_k"],
                     router_mode=ckpt["router_mode"], backbone_name="mobilenetv3small_torchvision",
                     pretrain_backbone=True, temperature=ckpt["temperature"])
    if ("moe_layer.gating.gate_projector.weight" in sd
            and "moe_layer.gating.gate_projector.0.weight" not in sd):
        model.moe_layer.gating = ContextAwareLinearGating(
            model_dim=model.feature_extractor.output_dim, context_dim=ckpt["context_dim"],
            num_experts=ckpt["num_experts"], top_k=ckpt["top_k"], temperature=ckpt["temperature"])
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    _, _, test_ds = get_moe_build("plantvillage")(use_context=True)
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


# ── Cluster-MoE (torchvision) ─────────────────────────────────
def eval_cluster(seed: int):
    from models.clustering_moe.model import ClusteringMoEModel
    from datasets.registry import get_test
    ckpt = torch.load(_latest(
        ROOT / "checkpoints/plantvillage/clustering_moe/dense_aligned_pretrain_backbone"
        / "mobilenetv3small_torchvision_backbone/kmeans/temperature_0.5/G4_cosine_top2" / f"seed_{seed}"),
        map_location=DEVICE)
    sd = ckpt["model_state_dict"]
    model = ClusteringMoEModel(num_classes=ckpt["num_classes"], centroids=sd["moe_layer.gating.centroids"],
                               top_k=ckpt["top_k"], backbone_name="mobilenetv3small_torchvision",
                               metric=ckpt["metric"], pretrain_backbone=True, temperature=ckpt["temperature"])
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    loader = DataLoader(get_test("plantvillage"), batch_size=64, shuffle=False)
    labs, preds = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits, _, _, _ = model(images.to(DEVICE))
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
            labs.extend(labels.tolist())
    return _metrics(labs, preds), _params_m(model)


FIELDS = ["seed", "dataset", "num_classes", "num_train", "num_val", "num_test",
          "backbone", "initialization", "model", "routing", "G", "top_k", "tau",
          "accuracy", "macro_f1", "weighted_f1", "params_m", "flops_g", "latency_ms"]

MODELS = {"dense": eval_dense, "learned_gate_moe": eval_moe, "cluster_moe": eval_cluster}
BACKBONE = {"dense": "mobilenetv3small_lamb1k", "learned_gate_moe": "mobilenetv3small_torchvision",
            "cluster_moe": "mobilenetv3small_torchvision"}


def plantdoc_rows(cp1_csv: Path):
    """Hàng PlantDoc từ CP1 (pretrained) để so sánh cạnh nhau."""
    rows = []
    if not cp1_csv.exists():
        return rows
    for r in csv.DictReader(open(cp1_csv)):
        if r["initialization"] != "imagenet_pretrained":
            continue
        m = r["model"]
        rows.append({
            "seed": r["seed"], **{k: PD[k] for k in ["dataset", "num_classes", "num_train", "num_val", "num_test"]},
            "backbone": "mobilenetv3small_torchvision", "initialization": "imagenet_pretrained",
            "model": m, "routing": ROUTING[m],
            "G": 4 if m != "dense" else "", "top_k": 2 if m != "dense" else "", "tau": 0.5 if m != "dense" else "",
            "accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "weighted_f1": r["weighted_f1"],
            "params_m": r["params_m"], "flops_g": r["flops_g"], "latency_ms": ARCH[m]["latency_ms"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "paper_results/tables/cross_dataset_results.csv")
    ap.add_argument("--cp1_csv", type=Path, default=ROOT / "paper_results/tables/pretrained_backbone_results.csv")
    ap.add_argument("--seeds", type=str, default="42 43 44 45 46")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split()]

    rows = []
    for mdl, fn in MODELS.items():
        for s in seeds:
            try:
                m, params = fn(s)
            except FileNotFoundError as e:
                print(f"[SKIP] {mdl} seed {s}: {e}")
                continue
            rows.append({
                "seed": s, **{k: PV[k] for k in ["dataset", "num_classes", "num_train", "num_val", "num_test"]},
                "backbone": BACKBONE[mdl], "initialization": "imagenet_pretrained",
                "model": mdl, "routing": ROUTING[mdl],
                "G": 4 if mdl != "dense" else "", "top_k": 2 if mdl != "dense" else "",
                "tau": 0.5 if mdl != "dense" else "",
                **m, "params_m": params, "flops_g": ARCH[mdl]["flops_g"], "latency_ms": ARCH[mdl]["latency_ms"],
            })
            print(f"[{mdl:16s} seed {s}] acc={m['accuracy']} mF1={m['macro_f1']} wF1={m['weighted_f1']}")

    rows += plantdoc_rows(args.cp1_csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
