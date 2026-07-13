"""CP7 — Train Soft MoE baseline (classifier-side, all-expert).

Soft MoE = MoE thường, chỉ khác 3 điểm:
  1. Active FULL expert (bỏ top-k)
  2. Bỏ dispatch/mask (mọi mẫu qua mọi expert)
  3. Bỏ load-balance / aux loss  ->  criterion = CrossEntropyLoss thuần

Training budget giữ Y HỆT `moe_train.sh` (lr 1e-3, wd 1e-3, 400 epoch, batch 64, τ=0.5,
context_dim=6) để thoả yêu cầu PDF mục X: *"cùng backbone, số expert và training budget"*.

TỰ CHỨA — không import gì từ models/moe, loss/loss_fn, utils/moe_trainer.

Chạy (từ src/):
    python -m training.soft_moe --seed 42 --dataset_name plantdoc --pretrain_backbone
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from datasets.registry import get_moe_build
from models.soft_moe.model import SoftMoEModel
from utils.soft_moe_trainer import SoftMoETrainer


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CP7 — Soft MoE baseline (classifier-side, all-expert)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--restart_id", type=int, default=0,
                   help="Restart thứ mấy của seed này. Đổi KHỞI TẠO (init + thứ tự batch), "
                        "KHÔNG đổi data split (split cố định random_state=42 trong LoadDataset). "
                        "Dùng cho best-of-N restarts, chọn theo VALIDATION accuracy.")
    p.add_argument("--dataset_name", type=str, default="plantdoc",
                   help="plantdoc | plantvillage")
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--backbone_name", type=str, default="mobilenetv3small_torchvision")
    p.add_argument("--context_dim", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.5,
                   help="Giữ 0.5 y hệt MoE/Cluster-MoE để cô lập biến routing")
    p.add_argument("--expert_hidden_dim", type=int, default=1024)
    # Training budget — y hệt moe_train.sh
    p.add_argument("--num_epochs", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--pretrain_backbone", action="store_true")
    return p.parse_args()


def main() -> None:
    args = get_args()
    # restart_id đổi khởi tạo (init + thứ tự batch) nhưng KHÔNG đổi data split:
    # split do LoadDataset quyết định với random_state=42 cố định, độc lập với seed này.
    init_seed = args.seed + 1000 * args.restart_id
    set_seed(init_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.restart_id:
        print(f"Restart #{args.restart_id} của seed {args.seed} (init_seed={init_seed})")

    # ── Dataset (có context, context_dim=6) ──
    build_datasets = get_moe_build(args.dataset_name)
    train_dataset, validation_dataset, _ = build_datasets(use_context=True)

    generator = torch.Generator()
    generator.manual_seed(init_seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    labels = train_dataset.labels
    num_classes = len(set(labels))
    print(f"Number of classes: {num_classes}")

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(labels), y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # ── Model ──
    model = SoftMoEModel(
        num_classes=num_classes,
        context_dim=args.context_dim,
        num_experts=args.num_experts,
        backbone_name=args.backbone_name,
        pretrain_backbone=args.pretrain_backbone,
        temperature=args.temperature,
        expert_hidden_dim=args.expert_hidden_dim,
    )
    print(
        f"SoftMoE: {args.num_experts} expert, ALL-ACTIVE (không top-k), "
        f"τ={args.temperature}, backbone={args.backbone_name}, "
        f"pretrained={args.pretrain_backbone}"
    )

    # ── Loss: CrossEntropy thuần — KHÔNG aux / load-balance loss ──
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ── Namespace checkpoint ──
    output_dir = Path.cwd().parents[0]
    checkpoint_dir = (
        output_dir
        / "checkpoints"
        / args.dataset_name
        / "soft_moe"
        / f"{args.backbone_name}_softmoe"
        / f"{args.num_experts}_experts"
        / f"seed_{args.seed}"
    )

    trainer = SoftMoETrainer(
        num_epochs=args.num_epochs,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=args.batch_size,
        checkpoint_dir=str(checkpoint_dir),
    )
    trainer.train()


if __name__ == "__main__":
    main()
