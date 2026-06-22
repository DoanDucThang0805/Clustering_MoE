import csv
import os
from pathlib import Path

import torch
from thop import profile

from models.clustering_moe.model import ClusteringMoEModel
from models.non_pretrain_baseline.mobilenetv3small import model as baseline_model
from models.moe.model import MoEModel

# ─── Model definitions ────────────────────────────────────────────────────────

cluster_moe_model = ClusteringMoEModel(
    num_classes=8,
    centroids=torch.randn(4, 576),
    top_k=2,
    backbone_name="mobilenetv3small_torchvision",
    metric="cosine",
    pretrain_backbone=False,
    temperature=0.5,
)

moe_model = MoEModel(
    context_dim=6,
    num_classes=8,
    num_experts=4,
    top_k=2,
    router_mode="context_aware",
    backbone_name="mobilenetv3small_torchvision",
    pretrain_backbone=False,
    temperature=0.5,
)

# ─── Models to benchmark ──────────────────────────────────────────────────────

models_to_benchmark = [
    ("MobileNetV3-Small (Baseline)", baseline_model, (torch.randn(1, 3, 224, 224),)),
    ("MoEModel",                     moe_model,       (torch.randn(1, 3, 224, 224), torch.randn(1, 6))),
    ("ClusteringMoEModel",           cluster_moe_model, (torch.randn(1, 3, 224, 224),)),
]

# ─── Benchmark loop ───────────────────────────────────────────────────────────

results = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_name, model, dummy_inputs in models_to_benchmark:
    model = model.to(device).eval()
    dummy_inputs_gpu = tuple(t.to(device) for t in dummy_inputs)
    with torch.no_grad():
        macs, params = profile(model, inputs=dummy_inputs_gpu, verbose=False)

    gflops = macs * 2 / 1e9          # MACs → FLOPs, then convert to G
    params_m = params / 1e6          # params → M

    results.append({
        "Model":        model_name,
        "Params (M)":   f"{params_m:.4f}",
        "FLOPs (G)":    f"{gflops:.4f}",
    })

    print(f"[{model_name}]  Params: {params_m:.4f} M  |  FLOPs: {gflops:.4f} G")

# ─── Export CSV ───────────────────────────────────────────────────────────────

OUTPUT_CSV = (
    Path(__file__).parents[2]
    / "params_flops_results"
    / "params_flops.csv"
)

if OUTPUT_CSV:
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Params (M)", "FLOPs (G)"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {OUTPUT_CSV}")
else:
    print("\nOUTPUT_CSV is empty — please fill in the path to export CSV.")
