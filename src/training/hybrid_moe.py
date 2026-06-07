from pathlib import Path
import random
from argparse import ArgumentParser
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

from loss.loss_fn import MoELoss
from utils.cluster_moe_trainer import ClusterMoETrainer
from models.hybrid_clustering_moe.model import HybridMoEModel

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────
# Load centroids
# ─────────────────────────────────────────────
def load_centroids(
    root_dir:    Path,
    dataset_name:str,
    backbone_type: str,
    backbone_name: str,
    model_clustering_name:  str,
    distance_metric: Literal["cosine", "euclidean"],
    num_experts: int,
    seed:        int,
) -> torch.Tensor:

    path = (
        root_dir
        / "clustering_results"
        / dataset_name
        / backbone_type
        / f"{backbone_name}_backbone"
        / model_clustering_name
        / distance_metric
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
# CLI
# ─────────────────────────────────────────────
parser = ArgumentParser(description="Train ClusteringMoEModel")
parser.add_argument("--seed",        type=int,   default=42)
parser.add_argument("--num_experts", type=int,   default=4,
                    help="G — number of experts / K-means clusters")
parser.add_argument("--top_k",       type=int,   default=2,
                    help="Number of experts activated per sample")
parser.add_argument("--distance_metric",      type=str,   default="cosine",
                    choices=["cosine", "euclidean"])
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--pretrain_backbone", action="store_true",
                    help="Whether to use a pretrained backbone (default: False)")
parser.add_argument("--lr",          type=float, default=1e-3)
parser.add_argument("--weight_decay",type=float, default=1e-3)
parser.add_argument("--num_epochs",  type=int,   default=200)
parser.add_argument("--batch_size",  type=int,   default=64)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--backbone_type",   type=str)
parser.add_argument("--backbone_name",type=str)
parser.add_argument("--model_clustering_name",   type=str)
parser.add_argument("--lambda_", type=float, required=True)
parser.add_argument("--moe_alpha", type=float, required=True, help="auxilirity coefficient for MoE Routing")
args = parser.parse_args()

# ─────────────────────────────────────────────
# Seed → then import datasets
# ─────────────────────────────────────────────
set_seed(args.seed)

from datasets.plantdoc_dataset_moe import build_datasets

# ─────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────
generator = torch.Generator()
generator.manual_seed(args.seed)

train_dataset, validation_dataset, _ = build_datasets(use_context=True)

train_loader = DataLoader(
    train_dataset,
    batch_size  = args.batch_size,
    shuffle     = True,
    generator   = generator,
    num_workers = 4,
    pin_memory  = True,
)

val_loader = DataLoader(
    validation_dataset,
    batch_size  = args.batch_size,
    shuffle     = False,
    num_workers = 4,
    pin_memory  = True,
)

# ─────────────────────────────────────────────
# Centroids + Model
# ─────────────────────────────────────────────
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = Path(__file__).parents[2]

centroids = load_centroids(
    root_dir     = root_dir,
    dataset_name = args.dataset_name,
    backbone_type  = args.backbone_type,
    backbone_name= args.backbone_name,
    model_clustering_name   = args.model_clustering_name,
    distance_metric= args.distance_metric,
    num_experts  = args.num_experts,
    seed         = args.seed,
)

labels      = train_dataset.labels
num_classes = len(set(labels))

model = HybridMoEModel(
    num_classes=num_classes,
    context_dim=6,
    num_experts=args.num_experts,
    centroids=centroids,
    top_k=args.top_k,
    backbone_name=args.backbone_name,
    pretrain_backbone=args.pretrain_backbone,
    lambda_=args.lambda_,
    temperature=args.temperature,
    metric=args.distance_metric,
    noise_stddev=1.0,
    context_proj_dim=32
)


# ─────────────────────────────────────────────
# Loss + Optimizer
# ─────────────────────────────────────────────
class_weights = compute_class_weight(
    class_weight = "balanced",
    classes      = np.arange(num_classes),
    y            = labels,
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = MoELoss(
    alpha=args.moe_alpha,
    class_weights=class_weights
)

optimizer = optim.AdamW(
    model.parameters(),
    lr           = args.lr,
    weight_decay = args.weight_decay,
)

# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────
checkpoint_dir = str(
    root_dir
    / "checkpoints"
    / args.dataset_name
    / "hybrid_moe"
    / args.backbone_type # eg: non_pretrain_backbone, pretrain_backbone
    / f"{args.backbone_name}_backbone"
    / args.model_clustering_name
    / f"temperature_{args.temperature}"
    / f"G{args.num_experts}_{args.distance_metric}_top{args.top_k}"
    / f"seed_{args.seed}"
)

trainer = ClusterMoETrainer(
    num_epochs              = args.num_epochs,
    device                  = device,
    batch_size              = args.batch_size,
    train_loader            = train_loader,
    val_loader              = val_loader,
    model                   = model,
    criterion               = criterion,
    optimizer               = optimizer,
    checkpoint_dir          = checkpoint_dir,
    warmup_epochs           = 10,
    min_lr                  = 1e-6,
    val_acc_threshold       = 1e-5,
    early_stopping_patience = 50,
    max_grad_norm           = 1.0,
    save_best               = True,
)


if __name__ == "__main__":
    print("=" * 60)
    print(f"  seed        : {args.seed}")
    print(f"  num_experts : {args.num_experts}")
    print(f"  top_k       : {args.top_k}")
    print(f"  metric      : {args.distance_metric}")
    print(f"  temperature : {args.temperature}")
    print(f"  device      : {device}")
    print(f"  checkpoint  : {checkpoint_dir}")
    print("=" * 60)
    trainer.train()
