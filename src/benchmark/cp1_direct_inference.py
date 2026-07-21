"""Recompute the CP1 pretrained rows from three locked PlantDoc checkpoints.

The from-scratch rows already present in ``cp1_backbone_init_summary.csv`` are
preserved. The ImageNet-pretrained rows are inferred directly for seeds 42--51
and replaced with mean and sample standard deviation over those ten seeds.

Run from ``src`` with::

    python -m benchmark.cp1_direct_inference
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small


ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = list(range(42, 52))
NUM_CLASSES = 8
OUT = ROOT / "paper_results/tables/cp1_backbone_init_summary.csv"

DENSE_ROOT = (
    ROOT
    / "checkpoints/plantdoc/pretrain_baseline/"
    "mobilenetv3small_torchvision_retrain2"
)
MOE_ROOT = (
    ROOT
    / "checkpoints/plantdoc/moe_temperature_0.5_pretrain_backbone/"
    "mobilenetv3small_torchvision_moe/4_experts/top_2"
)
CLUSTER_ROOT = (
    ROOT
    / "checkpoints/plantdoc/clustering_moe/dense_aligned_pretrain_backbone/"
    "mobilenetv3small_torchvision_backbone/kmeans/temperature_0.5/"
    "G4_cosine_top2"
)

MODEL_ORDER = ("dense", "learned_gate_moe", "cluster_moe")
PARAMS_M = {
    "dense": 1.5261,
    "learned_gate_moe": 3.4845,
    "cluster_moe": 3.4772,
}
FLOPS_G = {
    "dense": 0.1229,
    "learned_gate_moe": 0.1269,
    "cluster_moe": 0.1268,
}
FIELDS = [
    "initialization",
    "model",
    "n_seeds",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "weighted_f1_mean",
    "weighted_f1_std",
    "params_m",
    "flops_g",
]


def _locked_checkpoint(root: Path, seed: int) -> Path:
    candidates = sorted((root / f"seed_{seed}").glob("run_*/best_checkpoint.pth"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one checkpoint for seed {seed} under {root}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
    }


def eval_dense(seed: int) -> dict[str, float]:
    from datasets.plantdoc_dataset import test_dataset

    checkpoint_path = _locked_checkpoint(DENSE_ROOT, seed)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features, NUM_CLASSES
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    labels: list[int] = []
    predictions: list[int] = []
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(DEVICE))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(targets.tolist())
    return _metrics(labels, predictions)


def eval_moe(seed: int) -> dict[str, float]:
    from datasets.plantdoc_dataset_moe import build_datasets
    from models.moe.gating import ContextAwareLinearGating
    from models.moe.model import MoEModel

    checkpoint_path = _locked_checkpoint(MOE_ROOT, seed)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"]
    model = MoEModel(
        context_dim=checkpoint["context_dim"],
        num_classes=checkpoint["num_classes"],
        num_experts=checkpoint["num_experts"],
        top_k=checkpoint["top_k"],
        router_mode=checkpoint["router_mode"],
        backbone_name="mobilenetv3small_torchvision",
        pretrain_backbone=True,
        temperature=checkpoint["temperature"],
    )
    if (
        "moe_layer.gating.gate_projector.weight" in state_dict
        and "moe_layer.gating.gate_projector.0.weight" not in state_dict
    ):
        model.moe_layer.gating = ContextAwareLinearGating(
            model_dim=model.feature_extractor.output_dim,
            context_dim=checkpoint["context_dim"],
            num_experts=checkpoint["num_experts"],
            top_k=checkpoint["top_k"],
            temperature=checkpoint["temperature"],
        )
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()

    _, _, test_dataset = build_datasets(use_context=True)
    labels: list[int] = []
    predictions: list[int] = []
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    router_mode = checkpoint["router_mode"]
    with torch.inference_mode():
        for images, targets, context in loader:
            images = images.to(DEVICE)
            context = context.to(DEVICE)
            logits = (
                model(images, context)[0]
                if router_mode == "context_aware"
                else model(images)[0]
            )
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(targets.tolist())
    return _metrics(labels, predictions)


def eval_cluster(seed: int) -> dict[str, float]:
    from datasets.plantdoc_dataset import test_dataset
    from models.clustering_moe.model import ClusteringMoEModel

    checkpoint_path = _locked_checkpoint(CLUSTER_ROOT, seed)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"]
    model = ClusteringMoEModel(
        num_classes=checkpoint["num_classes"],
        centroids=state_dict["moe_layer.gating.centroids"],
        top_k=checkpoint["top_k"],
        backbone_name="mobilenetv3small_torchvision",
        metric=checkpoint["metric"],
        pretrain_backbone=True,
        temperature=checkpoint["temperature"],
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()

    labels: list[int] = []
    predictions: list[int] = []
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    with torch.inference_mode():
        for images, targets in loader:
            logits, _, _, _ = model(images.to(DEVICE))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(targets.tolist())
    return _metrics(labels, predictions)


EVALUATORS = {
    "dense": eval_dense,
    "learned_gate_moe": eval_moe,
    "cluster_moe": eval_cluster,
}


def _summary_row(model: str, results: list[dict[str, float]]) -> dict[str, object]:
    row: dict[str, object] = {
        "initialization": "imagenet_pretrained",
        "model": model,
        "n_seeds": len(results),
        "params_m": PARAMS_M[model],
        "flops_g": FLOPS_G[model],
    }
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        values = [result[metric] for result in results]
        row[f"{metric}_mean"] = f"{statistics.mean(values):.4f}"
        row[f"{metric}_std"] = f"{statistics.stdev(values):.4f}"
    return row


def main() -> None:
    existing = list(csv.DictReader(OUT.open(encoding="utf-8"))) if OUT.exists() else []
    scratch_rows = [row for row in existing if row["initialization"] == "from_scratch"]

    pretrained_rows = []
    for model in MODEL_ORDER:
        results = []
        print(f"\n[{model}]", flush=True)
        for seed in SEEDS:
            metrics = EVALUATORS[model](seed)
            results.append(metrics)
            print(
                f"seed={seed}: accuracy={metrics['accuracy']:.4f}, "
                f"macro_f1={metrics['macro_f1']:.4f}, "
                f"weighted_f1={metrics['weighted_f1']:.4f}",
                flush=True,
            )
        pretrained_rows.append(_summary_row(model, results))

    with OUT.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(scratch_rows + pretrained_rows)
    print(f"\nSaved {len(scratch_rows) + len(pretrained_rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
