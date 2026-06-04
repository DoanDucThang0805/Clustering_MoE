from pathlib import Path
import random
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

from utils.cluster_moe_trainer import ClusterMoETrainer
from models.moe_model.clustering.model import ClusteringMoEModel


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
    type_model: str,
    backbone_name: str,
    model_name:  str,
    num_experts: int,
    seed:        int,
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
# CLI
# ─────────────────────────────────────────────
parser = ArgumentParser(description="Train ClusteringMoEModel")
parser.add_argument("--seed",        type=int,   default=42)
parser.add_argument("--num_experts", type=int,   default=4,
                    help="G — number of experts / K-means clusters")
parser.add_argument("--top_k",       type=int,   default=2,
                    help="Number of experts activated per sample")
parser.add_argument("--metric",      type=str,   default="cosine",
                    choices=["cosine", "euclidean"])
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--pretrain_backbone", action="store_true",
                    help="Whether to use a pretrained backbone (default: False)")
parser.add_argument("--lr",          type=float, default=1e-3)
parser.add_argument("--weight_decay",type=float, default=1e-3)
parser.add_argument("--num_epochs",  type=int,   default=200)
parser.add_argument("--batch_size",  type=int,   default=64)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--type_model",   type=str)
parser.add_argument("--backbone_name",type=str)
parser.add_argument("--model_name",   type=str)

args = parser.parse_args()

# ─────────────────────────────────────────────
# Seed → then import datasets
# ─────────────────────────────────────────────
set_seed(args.seed)

from datasets.plantdoc_dataset import train_dataset, validation_dataset

# ─────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────
generator = torch.Generator()
generator.manual_seed(args.seed)

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
root_dir = Path.cwd().parents[0]

centroids = load_centroids(
    root_dir     = root_dir,
    dataset_name = args.dataset_name,
    type_model  = args.type_model,
    backbone_name= args.backbone_name,
    model_name   = args.model_name,
    num_experts  = args.num_experts,
    seed         = args.seed,
)

labels      = train_dataset.labels
num_classes = len(set(labels))

model = ClusteringMoEModel(
    num_classes        = num_classes,
    centroids          = centroids,
    top_k              = args.top_k,
    metric             = args.metric,
    pretrain_backbone  = args.pretrain_backbone,
    temperature        = args.temperature,
)

# ─────────────────────────────────────────────
# Loss + Optimizer
# ─────────────────────────────────────────────
class_weights = compute_class_weight(
    class_weight = "balanced",
    classes      = np.arange(num_classes),
    y            = labels,
)
criterion = nn.CrossEntropyLoss(
    weight = torch.tensor(class_weights, dtype=torch.float32).to(device)
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
    / args.type_model
    / "cluster_moe"
    / f"G{args.num_experts}_{args.metric}_top{args.top_k}"
    / f"temperature_{args.temperature}"
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
    print(f"  metric      : {args.metric}")
    print(f"  temperature : {args.temperature}")
    print(f"  device      : {device}")
    print(f"  checkpoint  : {checkpoint_dir}")
    print("=" * 60)
    trainer.train()
