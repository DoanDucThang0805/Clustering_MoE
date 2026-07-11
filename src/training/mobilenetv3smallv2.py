from pathlib import Path
import random
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

from utils.baseline_trainer import Trainer
from models.pretrain_baseline.mobilenetv3smallv2 import model
from datasets.registry import get_train_val


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parse = ArgumentParser()
parse.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parse.add_argument("--dataset_name", type=str, default="plantdoc",
                   help="plantdoc | plantvillage (chọn dataset + namespace checkpoint)")
args = parse.parse_args()

# Set seed BEFORE building datasets to ensure reproducible splits
set_seed(args.seed)

# Build datasets AFTER seed is set, theo dataset_name (registry)
train_dataset, validation_dataset = get_train_val(args.dataset_name)


BATCH_SIZE = 64
generator = torch.Generator()
generator.manual_seed(args.seed)

train_ds = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
)

val_ds = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = Path.cwd().parents[0]

labels = train_dataset.labels
num_classes = len(set(labels))

# Rebuild head theo num_classes (PlantDoc 8 / PlantVillage 10) — timm reset_classifier
model.reset_classifier(num_classes)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_classes),
    y=labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)

trainer = Trainer(
    num_epochs=200,
    device=device,
    batch_size=BATCH_SIZE,
    train_loader=train_ds,
    val_loader=val_ds,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    checkpoints_dir=str(
        output_dir
        / "checkpoints"
        / args.dataset_name
        / "pretrain_baseline"
        / "mobilenetv3small_timm_lamb1k"
        / f"seed_{args.seed}"
    )
)


if __name__ == "__main__":
    trainer.train()
