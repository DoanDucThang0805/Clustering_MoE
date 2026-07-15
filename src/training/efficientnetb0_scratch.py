"""Dense EfficientNet-B0 (torchvision) FROM SCRATCH — mirror training/efficientnetb0.py,
chỉ khác: model không dùng ImageNet weights + namespace non_pretrain_baseline."""
from pathlib import Path
import random
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from datasets.registry import get_train_val
from models.non_pretrain_baseline.efficientnetb0 import model

from utils.baseline_trainer import Trainer


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
parse.add_argument("--restart_id", type=int, default=0,
                   help="Restart thứ mấy (best-of-N theo VAL). Đổi init, KHÔNG đổi data split.")
parse.add_argument("--dataset_name", type=str, default="plantdoc",
                   help="plantdoc | plantvillage (chọn dataset + namespace checkpoint)")
args = parse.parse_args()

init_seed = args.seed + 1000 * args.restart_id
set_seed(init_seed)

train_dataset, validation_dataset = get_train_val(args.dataset_name)


BATCH_SIZE = 32
generator = torch.Generator()
generator.manual_seed(init_seed)

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

# Rebuild head theo num_classes (PlantDoc 8 / PlantVillage 10)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

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
        / "non_pretrain_baseline"
        / "efficientnetb0_torchvision"
        / f"seed_{args.seed}"
    )
)


if __name__ == "__main__":
    trainer.train()
