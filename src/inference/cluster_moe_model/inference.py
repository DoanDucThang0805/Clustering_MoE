"""
Inference script for ClusteringMoE Plant Disease Classification.
"""

from pathlib import Path
import random
from argparse import ArgumentParser, Namespace
from typing import Literal

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from models.clustering_moe.model import ClusteringMoEModel


# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────
# Centroids
# ─────────────────────────────────────────────────────────────
def load_centroids(
    root_dir:              Path,
    dataset_name:          str,
    backbone_type:         str,
    backbone_name:         str,
    model_clustering_name: str,
    metric:                Literal["cosine", "euclidean"],
    num_experts:           int,
    seed:                  int,
) -> torch.Tensor:
    path = (
        root_dir
        / "clustering_results"
        / dataset_name
        / backbone_type
        / f"{backbone_name}_backbone"
        / model_clustering_name
        / metric
        / f"seed_{seed}"
        / f"clusters_kmeans_G{num_experts}_seed{seed}.npz"
    )
    if not path.exists():
        raise FileNotFoundError(f"Centroid file not found:\n  {path}")

    data      = np.load(path)
    centroids = data["centroids"]                               # (G, D)
    print(f"Loaded centroids: {centroids.shape}  from {path.name}")
    return torch.tensor(centroids, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# Path builders
# ─────────────────────────────────────────────────────────────
def build_checkpoint_path(root_dir: Path, args: Namespace) -> Path:
    return (
        root_dir
        / "checkpoints"
        / args.dataset_name
        / "clustering_moe"
        / args.backbone_type
        / f"{args.backbone_name}_backbone"
        / args.model_clustering_name
        / f"temperature_{args.temperature}"
        / f"G{args.num_experts}_{args.metric}_top{args.top_k}"
        / f"seed_{args.seed}"
        / args.runtime
        / args.checkpoint
    )


def build_output_dir(root_dir: Path, args: Namespace, ckpt: dict) -> Path:
    return (
        root_dir
        / "reports"
        / args.dataset_name
        / "clustering_moe_model"
        / args.backbone_type
        / args.backbone_name
        / args.model_clustering_name
        / f"G{ckpt['num_experts']}_{ckpt['metric']}_top{ckpt['top_k']}"
        / f"temperature_{ckpt['temperature']}"
        / f"seed_{args.seed}"
        / args.runtime
    )


# ─────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────
def load_model(
    ckpt:      dict,
    centroids: torch.Tensor,
    args:      Namespace,
) -> ClusteringMoEModel:
    model = ClusteringMoEModel(
        num_classes       = ckpt["num_classes"],
        centroids         = centroids,
        top_k             = ckpt["top_k"],
        backbone_name     = args.backbone_name,
        metric            = ckpt["metric"],
        pretrain_backbone = args.pretrain_backbone,
        temperature       = ckpt["temperature"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model


# ─────────────────────────────────────────────────────────────
# Inference engine
# ─────────────────────────────────────────────────────────────
class ClusterMoEInference:

    def __init__(
        self,
        model:       ClusteringMoEModel,
        device:      torch.device,
        output_dir:  Path,
        class_names: list[str] | None = None,
    ):
        self.model       = model.to(device).eval()
        self.device      = device
        self.output_dir  = output_dir
        self.class_names = class_names
        self.output_dir.mkdir(parents=True, exist_ok=True)


    @torch.inference_mode()
    def run(self, loader: DataLoader) -> dict:
        """Run inference over loader, return raw results."""
        all_preds       = []
        all_labels      = []
        all_weights     = []
        all_top_indices = []

        for images, labels in loader:
            images = images.to(self.device)

            logits, weights, top_indices, _ = self.model(images)
            preds = torch.argmax(logits, dim=1).cpu()

            all_preds.append(preds)
            all_labels.append(labels)
            all_weights.append(weights.cpu())
            all_top_indices.append(top_indices.cpu())

        return {
            "preds":       torch.cat(all_preds).numpy(),
            "labels":      torch.cat(all_labels).numpy(),
            "weights":     torch.cat(all_weights).numpy(),
            "top_indices": torch.cat(all_top_indices).numpy(),
        }


    def report(self, results: dict) -> None:
        """Print and save classification report."""
        preds  = results["preds"]
        labels = results["labels"]
        acc    = accuracy_score(labels, preds) * 100

        report_str = classification_report(
            labels, preds,
            target_names = self.class_names,
            digits       = 4,
        )

        print(f"\n{'='*60}")
        print(f"  Test Accuracy: {acc:.2f}%")
        print(f"{'='*60}")
        print(report_str)

        path = self.output_dir / "classification_report.txt"
        path.write_text(f"Test Accuracy: {acc:.2f}%\n\n{report_str}")
        print(f"Saved → {path}")


    def plot_confusion_matrix(self, results: dict) -> None:
        """Save count and normalized confusion matrix side-by-side."""
        cm   = confusion_matrix(results["labels"], results["preds"])
        norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        for ax, data, title, fmt in zip(
            axes,
            [cm,   norm],
            ["Confusion Matrix (Count)", "Confusion Matrix (Normalized)"],
            ["d",  ".2f"],
        ):
            sns.heatmap(
                data,
                annot       = True,
                fmt         = fmt,
                cmap        = "Blues",
                xticklabels = self.class_names or "auto",
                yticklabels = self.class_names or "auto",
                ax          = ax,
            )
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("Predicted", fontsize=11)
            ax.set_ylabel("True",      fontsize=11)
            ax.tick_params(axis="x", rotation=45)
            ax.tick_params(axis="y", rotation=0)

        plt.tight_layout()
        path = self.output_dir / "confusion_matrix.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {path}")


    def run_all(self, loader: DataLoader) -> None:
        """Run inference + generate all reports and plots."""
        print("Running inference on test set...")
        results = self.run(loader)
        self.report(results)
        self.plot_confusion_matrix(results)
        print(f"\nAll outputs saved to: {self.output_dir}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args() -> Namespace:
    parser = ArgumentParser(description="Inference ClusteringMoEModel on test set")
    parser.add_argument("--seed",                  type=int,   default=42)
    parser.add_argument("--num_experts",           type=int,   default=4)
    parser.add_argument("--top_k",                 type=int,   default=2)
    parser.add_argument("--metric",                type=str,   default="cosine",
                        choices=["cosine", "euclidean"])
    parser.add_argument("--temperature",           type=float, default=0.5)
    parser.add_argument("--pretrain_backbone",     action="store_true")
    parser.add_argument("--batch_size",            type=int,   default=64)
    parser.add_argument("--dataset_name",          type=str,   required=True)
    parser.add_argument("--backbone_type",         type=str,   required=True)
    parser.add_argument("--backbone_name",         type=str,   required=True)
    parser.add_argument("--model_clustering_name", type=str,   required=True)
    parser.add_argument("--checkpoint",            type=str,   default="best_checkpoint.pth",
                        choices=["best_checkpoint.pth", "last_checkpoint.pth"])
    parser.add_argument("--runtime",               type=str,   help="e.g. run_20260531-101010")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    args     = parse_args()
    set_seed(args.seed)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = Path(__file__).parents[3]

    print("=" * 60)
    print(f"  seed        : {args.seed}")
    print(f"  num_experts : {args.num_experts}")
    print(f"  top_k       : {args.top_k}")
    print(f"  metric      : {args.metric}")
    print(f"  temperature : {args.temperature}")
    print(f"  checkpoint  : {args.checkpoint}")
    print(f"  device      : {device}")
    print("=" * 60)

    # ── Dataset ──────────────────────────────────────────────
    from datasets.plantdoc_dataset import test_dataset

    test_loader = DataLoader(
        test_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True,
    )
    class_names = (
        list(test_dataset.class_to_idx.keys())
        if hasattr(test_dataset, "class_to_idx") else None
    )

    # ── Centroids ─────────────────────────────────────────────
    centroids = load_centroids(
        root_dir              = root_dir,
        dataset_name          = args.dataset_name,
        backbone_type         = args.backbone_type,
        backbone_name         = args.backbone_name,
        model_clustering_name = args.model_clustering_name,
        metric                = args.metric,
        num_experts           = args.num_experts,
        seed                  = args.seed,
    )

    # ── Checkpoint ────────────────────────────────────────────
    checkpoint_path = build_checkpoint_path(root_dir, args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n  {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint (epoch {checkpoint['epoch']}): {checkpoint_path.name}")

    # ── Model ─────────────────────────────────────────────────
    model = load_model(checkpoint, centroids, args)

    # ── Output dir ────────────────────────────────────────────
    output_dir = build_output_dir(root_dir, args, checkpoint)

    # ── Run inference ─────────────────────────────────────────
    inference = ClusterMoEInference(
        model       = model,
        device      = device,
        output_dir  = output_dir,
        class_names = class_names,
    )
    inference.run_all(test_loader)


if __name__ == "__main__":
    main()
