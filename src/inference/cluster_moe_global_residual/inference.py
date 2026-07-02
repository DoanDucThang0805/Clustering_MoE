from argparse import ArgumentParser
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from models.clustering_moe.model_global_residual import (
    GlobalResidualClusteringMoEModel,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = ArgumentParser(
        description="Inference for dense-global residual Cluster-MoE"
    )
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--pretrain_backbone", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset_name", default="plantdoc")
    parser.add_argument("--backbone_type", required=True)
    parser.add_argument("--centroid_backbone_type", required=True)
    parser.add_argument("--backbone_name", default="mobilenetv3small_torchvision")
    parser.add_argument("--model_clustering_name", default="kmeans")
    parser.add_argument("--checkpoint", default="best_checkpoint.pth")
    parser.add_argument("--runtime", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    from datasets.plantdoc_dataset import test_dataset

    root_dir = Path(__file__).parents[3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    centroid_path = (
        root_dir
        / "clustering_results"
        / args.dataset_name
        / args.centroid_backbone_type
        / f"{args.backbone_name}_backbone"
        / args.model_clustering_name
        / args.metric
        / f"seed_{args.seed}"
        / f"clusters_kmeans_G{args.num_experts}_seed{args.seed}.npz"
    )
    checkpoint_path = (
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
    if not centroid_path.is_file():
        raise FileNotFoundError(f"Centroid file not found: {centroid_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with np.load(centroid_path) as data:
        centroids = torch.tensor(data["centroids"], dtype=torch.float32)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = GlobalResidualClusteringMoEModel(
        num_classes=checkpoint["num_classes"],
        centroids=centroids,
        top_k=checkpoint["top_k"],
        backbone_name=args.backbone_name,
        metric=checkpoint["metric"],
        pretrain_backbone=args.pretrain_backbone,
        temperature=checkpoint["temperature"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    predictions = []
    labels = []
    with torch.inference_mode():
        for images, batch_labels in loader:
            logits, _, _, _ = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.tolist())

    class_names = list(test_dataset.class_to_idx.keys())
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4,
    )
    print(f"Loaded checkpoint epoch: {checkpoint['epoch']}")
    print(f"Learned residual scale: {model.residual_scale.item():.6f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(report)

    output_dir = (
        root_dir
        / "reports"
        / args.dataset_name
        / "clustering_moe_global_residual"
        / args.backbone_type
        / args.backbone_name
        / args.model_clustering_name
        / f"G{checkpoint['num_experts']}_{checkpoint['metric']}_top{checkpoint['top_k']}"
        / f"temperature_{checkpoint['temperature']}"
        / f"seed_{args.seed}"
        / args.runtime
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classification_report.txt").write_text(
        f"Test Accuracy: {accuracy * 100:.2f}%\n"
        f"Residual Scale: {model.residual_scale.item():.6f}\n\n"
        f"{report}"
    )

    matrix = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
