from pathlib import Path
import random
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import seaborn as sns

from models.clustering_moe.model import ClusteringMoEModel

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────
# Load centroids
# ─────────────────────────────────────────────
def load_centroids(
    root_dir:     Path,
    dataset_name: str,
    type_model:   str,
    backbone_name:str,
    model_name:   str,
    num_experts:  int,
    seed:         int,
) -> torch.Tensor:
    path = (
        root_dir
        / "clustering_results"
        / dataset_name
        / type_model
        / f"{backbone_name}_backbone"
        / model_name
        / f"seed_{seed}"
        / f"clusters_kmeans_G{num_experts}_seed{seed}.npz"
    )
    if not path.exists():
        raise FileNotFoundError(f"Centroid file not found:\n  {path}")

    data      = np.load(path)
    centroids = data["centroids"]                               # (G, D)
    print(f"Loaded centroids: {centroids.shape}  from {path.name}")
    return torch.tensor(centroids, dtype=torch.float32)


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
class ClusterMoEInference:

    def __init__(
        self,
        model:      ClusteringMoEModel,
        device:     torch.device,
        output_dir: Path,
        class_names: list[str] | None = None,
    ):
        self.model       = model.to(device).eval()
        self.device      = device
        self.output_dir  = output_dir
        self.class_names = class_names
        self.output_dir.mkdir(parents=True, exist_ok=True)


    @torch.inference_mode()
    def run(self, loader: DataLoader) -> dict:
        """Run inference, return dict with all raw results."""
        all_preds        = []
        all_labels       = []
        all_weights      = []     # (N, top_k)
        all_top_indices  = []     # (N, top_k)

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
        """Print + save classification report."""
        preds  = results["preds"]
        labels = results["labels"]
        acc    = accuracy_score(labels, preds) * 100

        print(f"\n{'='*60}")
        print(f"  Test Accuracy: {acc:.2f}%")
        print(f"{'='*60}")
        print(classification_report(
            labels, preds,
            target_names=self.class_names,
            digits=4,
        ))

        report_str = classification_report(
            labels, preds,
            target_names=self.class_names,
            digits=4,
        )
        report_path = self.output_dir / "classification_report.txt"
        report_path.write_text(
            f"Test Accuracy: {acc:.2f}%\n\n{report_str}"
        )
        print(f"Saved → {report_path}")


    def plot_confusion_matrix(self, results: dict) -> None:
        """Save confusion matrix heatmap."""
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
                annot      = True,
                fmt        = fmt,
                cmap       = "Blues",
                xticklabels= self.class_names or "auto",
                yticklabels= self.class_names or "auto",
                ax         = ax,
            )
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("Predicted", fontsize=11)
            ax.set_ylabel("True", fontsize=11)
            ax.tick_params(axis="x", rotation=45)
            ax.tick_params(axis="y", rotation=0)

        plt.tight_layout()
        path = self.output_dir / "confusion_matrix.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {path}")


    def plot_expert_usage(self, results: dict) -> None:
        """Bar chart: how many times each expert was selected."""
        top_indices  = results["top_indices"]              # (N, top_k)
        num_experts  = self.model.num_experts

        counts = np.zeros(num_experts, dtype=int)
        for e in range(num_experts):
            counts[e] = (top_indices == e).sum()

        plt.figure(figsize=(8, 4))
        bars = plt.bar(range(num_experts), counts, color="#4C72B0", edgecolor="white")
        plt.bar_label(bars, padding=3, fontsize=10)
        plt.title("Expert Usage on Test Set", fontsize=13)
        plt.xlabel("Expert ID")
        plt.ylabel("Selection Count")
        plt.xticks(range(num_experts), [f"E{i}" for i in range(num_experts)])
        plt.tight_layout()
        path = self.output_dir / "expert_usage.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {path}")


    def plot_expert_class_heatmap(self, results: dict) -> None:
        """Heatmap: which expert handles which class."""
        top_indices  = results["top_indices"]              # (N, top_k)
        labels       = results["labels"]                   # (N,)
        num_experts  = self.model.num_experts
        num_classes  = self.model.classifier[-1].out_features

        matrix = np.zeros((num_experts, num_classes), dtype=float)
        for sample_idx in range(len(labels)):
            c = labels[sample_idx]
            for e in top_indices[sample_idx]:
                matrix[e, c] += 1

        # Normalize per expert
        norm = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-9)

        plt.figure(figsize=(12, 5))
        sns.heatmap(
            norm,
            annot       = True,
            fmt         = ".2f",
            cmap        = "Blues",
            xticklabels = self.class_names or [f"C{i}" for i in range(num_classes)],
            yticklabels = [f"E{i}" for i in range(num_experts)],
        )
        plt.title("Expert–Class Routing Distribution (Normalized)", fontsize=13)
        plt.xlabel("Class"); plt.ylabel("Expert")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path = self.output_dir / "expert_class_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {path}")


    def run_all(self, loader: DataLoader) -> None:
        """Run inference + generate all reports and plots."""
        print("Running inference on test set...")
        results = self.run(loader)
        self.report(results)
        self.plot_confusion_matrix(results)
        self.plot_expert_usage(results)
        self.plot_expert_class_heatmap(results)
        print(f"\nAll outputs saved to: {self.output_dir}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
parser = ArgumentParser(description="Inference ClusteringMoEModel on test set")
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--num_experts",   type=int,   default=4)
parser.add_argument("--top_k",         type=int,   default=2)
parser.add_argument("--metric",        type=str,   default="cosine",
                    choices=["cosine", "euclidean"])
parser.add_argument("--temperature",   type=float, default=0.5)
parser.add_argument("--pretrain_backbone", action="store_true")
parser.add_argument("--batch_size",    type=int,   default=64)
parser.add_argument("--dataset_name",  type=str,   required=True)
parser.add_argument("--type_model",    type=str,   required=True)
parser.add_argument("--backbone_name", type=str,   required=True)
parser.add_argument("--model_name",    type=str,   required=True)
parser.add_argument("--checkpoint",    type=str,   default="best_checkpoint.pth",
                    choices=["best_checkpoint.pth", "last_checkpoint.pth"])
parser.add_argument("--run_time", type=str, required=True,
                    help="Run timestamp folder, e.g. run_20240601-120000")
args = parser.parse_args()

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
set_seed(args.seed)

from datasets.plantdoc_dataset import test_dataset

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = Path.cwd().parents[0]

# ─────────────────────────────────────────────
# DataLoader
# ─────────────────────────────────────────────
test_loader = DataLoader(
    test_dataset,
    batch_size  = args.batch_size,
    shuffle     = False,
    num_workers = 4,
    pin_memory  = True,
)

# ─────────────────────────────────────────────
# Centroids + Model
# ─────────────────────────────────────────────
centroids = load_centroids(
    root_dir      = root_dir,
    dataset_name  = args.dataset_name,
    type_model    = args.type_model,
    backbone_name = args.backbone_name,
    model_name    = args.model_name,
    num_experts   = args.num_experts,
    seed          = args.seed,
)

num_classes = len(set(test_dataset.labels))
class_names = list(test_dataset.class_to_idx.keys()) if hasattr(test_dataset, "class_to_idx") else None

model = ClusteringMoEModel(
    num_classes       = num_classes,
    centroids         = centroids,
    top_k             = args.top_k,
    metric            = args.metric,
    pretrain_backbone = args.pretrain_backbone,
    temperature       = args.temperature,
)

# ─────────────────────────────────────────────
# Load checkpoint
# ─────────────────────────────────────────────
checkpoint_dir = (
    root_dir
    / "checkpoints"
    / args.dataset_name
    / args.type_model
    / "cluster_moe"
    / f"G{args.num_experts}_{args.metric}_top{args.top_k}"
    / f"temperature_{args.temperature}"
    / f"seed_{args.seed}"
    / args.run_time
)
checkpoint_path = checkpoint_dir / args.checkpoint

if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found:\n  {checkpoint_path}")

ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
print(f"Loaded checkpoint (epoch {ckpt['epoch']}): {checkpoint_path.name}")

# ─────────────────────────────────────────────
# Output dir
# ─────────────────────────────────────────────
output_dir = checkpoint_dir / "inference_test"

# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"  seed        : {args.seed}")
    print(f"  num_experts : {args.num_experts}")
    print(f"  top_k       : {args.top_k}")
    print(f"  metric      : {args.metric}")
    print(f"  temperature : {args.temperature}")
    print(f"  checkpoint  : {args.checkpoint}")
    print(f"  device      : {device}")
    print("=" * 60)

    inference = ClusterMoEInference(
        model       = model,
        device      = device,
        output_dir  = output_dir,
        class_names = class_names,
    )
    inference.run_all(test_loader)
