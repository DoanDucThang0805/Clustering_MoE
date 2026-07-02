from argparse import ArgumentParser
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from models.clustering_moe.model_global_residual import (
    GlobalResidualClusteringMoEModel,
)
from utils.cluster_moe_trainer import ClusterMoETrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_centroids(
    root_dir: Path,
    dataset_name: str,
    backbone_type: str,
    backbone_name: str,
    clustering_name: str,
    metric: str,
    num_experts: int,
    seed: int,
) -> torch.Tensor:
    path = (
        root_dir
        / "clustering_results"
        / dataset_name
        / backbone_type
        / f"{backbone_name}_backbone"
        / clustering_name
        / metric
        / f"seed_{seed}"
        / f"clusters_kmeans_G{num_experts}_seed{seed}.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Centroid file not found:\n  {path}")

    with np.load(path) as data:
        centroids = data["centroids"]

    if centroids.shape[0] != num_experts:
        raise ValueError(
            f"Expected {num_experts} centroids, got {centroids.shape}"
        )

    print(f"Loaded centroids: {centroids.shape} from {path}")
    return torch.tensor(centroids, dtype=torch.float32)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="Train dense-global residual Cluster-MoE"
    )
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument(
        "--distance_metric",
        choices=["cosine", "euclidean"],
        default="cosine",
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--pretrain_backbone", action="store_true")
    parser.add_argument(
        "--freeze_dense_branch",
        action="store_true",
        help="Freeze the loaded backbone and global dense classifier.",
    )
    parser.add_argument("--backbone_checkpoint", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--num_epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset_name", type=str, default="plantdoc")
    parser.add_argument(
        "--backbone_type",
        type=str,
        default="dense_global_residual_pretrain_backbone",
        help="Checkpoint output namespace.",
    )
    parser.add_argument(
        "--centroid_backbone_type",
        type=str,
        default="pretrain_backbone",
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="mobilenetv3small_torchvision",
    )
    parser.add_argument(
        "--model_clustering_name",
        type=str,
        default="kmeans",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()
    set_seed(args.seed)

    from datasets.plantdoc_dataset import train_dataset, validation_dataset

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    root_dir = Path(__file__).parents[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    centroids = load_centroids(
        root_dir=root_dir,
        dataset_name=args.dataset_name,
        backbone_type=args.centroid_backbone_type,
        backbone_name=args.backbone_name,
        clustering_name=args.model_clustering_name,
        metric=args.distance_metric,
        num_experts=args.num_experts,
        seed=args.seed,
    )

    labels = train_dataset.labels
    num_classes = len(set(labels))
    model = GlobalResidualClusteringMoEModel(
        num_classes=num_classes,
        centroids=centroids,
        top_k=args.top_k,
        backbone_name=args.backbone_name,
        metric=args.distance_metric,
        pretrain_backbone=args.pretrain_backbone,
        temperature=args.temperature,
    )
    feature_count, classifier_count = model.load_dense_checkpoint(
        args.backbone_checkpoint
    )
    model.centroid_backbone_type = args.centroid_backbone_type
    if args.freeze_dense_branch:
        model.freeze_dense_branch()

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=labels,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        ),
        label_smoothing=args.label_smoothing,
    )
    optimizer = optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    checkpoint_dir = (
        root_dir
        / "checkpoints"
        / args.dataset_name
        / "clustering_moe"
        / args.backbone_type
        / f"{args.backbone_name}_backbone"
        / args.model_clustering_name
        / f"temperature_{args.temperature}"
        / f"G{args.num_experts}_{args.distance_metric}_top{args.top_k}"
        / f"seed_{args.seed}"
    )

    print("=" * 64)
    print("  Dense-global residual Cluster-MoE")
    print(f"  seed             : {args.seed}")
    print(f"  experts/top-k    : {args.num_experts}/{args.top_k}")
    print(f"  metric/temp      : {args.distance_metric}/{args.temperature}")
    print(f"  backbone ckpt    : {model.backbone_checkpoint_path}")
    print(f"  loaded tensors   : features={feature_count}, classifier={classifier_count}")
    print(f"  initial residual : {model.residual_scale.item():.6f}")
    print(f"  freeze dense     : {args.freeze_dense_branch}")
    print(
        f"  trainable params : "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(f"  centroid source  : {args.centroid_backbone_type}")
    print(f"  output           : {checkpoint_dir}")
    print(f"  device           : {device}")
    print("=" * 64)

    trainer = ClusterMoETrainer(
        num_epochs=args.num_epochs,
        device=device,
        batch_size=args.batch_size,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_dir=str(checkpoint_dir),
        warmup_epochs=10,
        min_lr=1e-6,
        val_acc_threshold=1e-5,
        early_stopping_patience=50,
        max_grad_norm=1.0,
        save_best=True,
    )
    trainer.train()


if __name__ == "__main__":
    main()
