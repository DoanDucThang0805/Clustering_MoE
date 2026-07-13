"""Trainer cho Soft MoE (CP7) — TỰ CHỨA.

KHÔNG dùng `MoETrainer` vì nó khoá chặt vào interface của MoE:
`model.router_mode`, model trả `topk_indices`, `criterion(logits, labels, router_logits,
top_k_indices)` (4 tham số), `_monitor_expert_usage(topk_indices)`.
Soft MoE không có top-k ⇒ sẽ phải "giả vờ" có `topk_indices` → chồng lấn, khó bảo trì.

Khác biệt so với MoETrainer:
  - model trả `(logits, gate_weights)` — không có topk_indices.
  - `criterion(logits, labels)` — 2 tham số (CrossEntropy thuần, KHÔNG aux/balance loss).
  - Theo dõi `_monitor_gate_weights` (trọng số trung bình mỗi expert) thay cho đếm top-k.
Lịch LR / early stopping / lưu checkpoint / vẽ plot: giữ y hệt để "cùng training budget".
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import tqdm
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from metrics.accuracy import accuracy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoftMoETrainer:
    def __init__(
        self,
        num_epochs: int,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,          # nn.CrossEntropyLoss — 2 tham số, không aux
        optimizer: optim.Optimizer,
        batch_size: int,
        checkpoint_dir: str = "checkpoints",
        warmup_epochs: int = 10,
        min_lr: float = 1e-6,
        val_acc_threshold: float = 1e-5,
        early_stopping_patience: int = 50,
        max_grad_norm: float = 1.0,
        save_best: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.val_acc_threshold = val_acc_threshold
        self.early_stopping_patience = early_stopping_patience
        self.max_grad_norm = max_grad_norm
        self.save_best = save_best

        # Soft: mọi expert luôn active → theo dõi TRỌNG SỐ TRUNG BÌNH mỗi expert
        # (không có top-k để đếm như MoE).
        self.gate_weight_sum = torch.zeros(self.model.num_experts)
        self.gate_batches = 0

        cosine_epochs = max(num_epochs - warmup_epochs, 1)
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cosine_epochs, eta_min=min_lr
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(self.checkpoint_dir, f"run_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)

        logger.propagate = False
        logger.handlers = [
            h for h in logger.handlers if not isinstance(h, logging.FileHandler)
        ]
        file_handler = logging.FileHandler(os.path.join(self.run_dir, "training.log"))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.train_acc_history: list[float] = []
        self.val_acc_history: list[float] = []
        self.lr_history: list[float] = []

    # ─────────────────────────────────────────────────────────
    def _save_checkpoint(self, path: str, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "train_loss_history": self.train_loss_history,
                "val_loss_history": self.val_loss_history,
                "train_acc_history": self.train_acc_history,
                "val_acc_history": self.val_acc_history,
                "lr_history": self.lr_history,
                # Metadata đủ để rebuild bằng build_soft_moe_from_checkpoint
                "num_classes": self.model.num_classes,
                "num_experts": self.model.num_experts,
                "context_dim": self.model.context_dim,
                "temperature": self.model.temperature,
                "backbone_name": self.model.backbone_name,
                "expert_hidden_dim": self.model.expert_hidden_dim,
            },
            path,
        )
        logger.info(f"Saved checkpoint: {path}")

    def _monitor_gate_weights(self, weights: torch.Tensor) -> None:
        """Soft không có top-k để đếm → theo dõi trọng số trung bình mỗi expert."""
        self.gate_weight_sum += weights.mean(dim=0).detach().cpu()
        self.gate_batches += 1

    # ─────────────────────────────────────────────────────────
    def train(self) -> None:
        best_val_acc = -float("inf")
        best_epoch = -1
        no_improve_count = 0
        epoch = 0

        for epoch in tqdm.tqdm(range(self.num_epochs), desc="Epochs"):
            # ---------------- TRAIN ----------------
            self.model.train()
            train_running_loss = 0.0
            train_running_correct = 0.0

            for images, labels, context in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                context = context.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                logits, gate_weights = self.model(images, context)
                loss = self.criterion(logits, labels)     # KHÔNG aux/balance loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    self._monitor_gate_weights(gate_weights)
                    preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                    acc = accuracy(preds, labels)

                train_running_loss += loss.item()
                train_running_correct += acc

            train_loss = train_running_loss / len(self.train_loader)
            train_acc = train_running_correct / len(self.train_loader)
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.lr_history.append(current_lr)
            logger.info(
                f"Epoch[{epoch+1}/{self.num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
                f"LR: {current_lr:.2e}"
            )

            # ---------------- VALIDATION ----------------
            self.model.eval()
            val_running_loss = 0.0
            val_running_correct = 0.0

            with torch.inference_mode():
                for images, labels, context in self.val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    context = context.to(self.device)

                    logits, _ = self.model(images, context)
                    loss = self.criterion(logits, labels)

                    preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                    val_running_loss += loss.item()
                    val_running_correct += accuracy(preds, labels)

            validation_loss = val_running_loss / len(self.val_loader)
            validation_acc = val_running_correct / len(self.val_loader)
            self.val_loss_history.append(validation_loss)
            self.val_acc_history.append(validation_acc)

            logger.info(
                f"Epoch[{epoch+1}/{self.num_epochs}] "
                f"Val Loss: {validation_loss:.4f}, Val Acc: {validation_acc:.2f}%"
            )

            self.scheduler.step()

            if validation_acc > best_val_acc + self.val_acc_threshold:
                logger.info(
                    f"Validation accuracy improved "
                    f"({best_val_acc:.4f} -> {validation_acc:.4f})."
                )
                best_val_acc = validation_acc
                best_epoch = epoch + 1
                no_improve_count = 0
                if self.save_best:
                    self._save_checkpoint(
                        os.path.join(self.run_dir, "best_checkpoint.pth"), epoch + 1
                    )
            else:
                no_improve_count += 1
                logger.info(f"No improvement for {no_improve_count} epoch(s).")

            if no_improve_count >= self.early_stopping_patience:
                logger.info("Early stopping triggered.")
                break

        self._save_checkpoint(os.path.join(self.run_dir, "last_checkpoint.pth"), epoch + 1)
        logger.info(
            f"Training finished. Best val acc: {best_val_acc:.4f} at epoch {best_epoch}"
        )

        if self.gate_batches > 0:
            mean_w = (self.gate_weight_sum / self.gate_batches).tolist()
            logger.info(
                "Mean gate weight per expert (soft, all-active): "
                + ", ".join(f"e{i}={w:.4f}" for i, w in enumerate(mean_w))
            )

        # ---------------- PLOTS ----------------
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self.train_loss_history, label="train_loss")
        plt.plot(self.val_loss_history, label="val_loss")
        plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(self.train_acc_history, label="train_acc")
        plt.plot(self.val_acc_history, label="val_acc")
        plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Acc (%)"); plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(self.lr_history, label="lr")
        plt.title("Learning rate"); plt.xlabel("Epoch"); plt.ylabel("LR"); plt.legend()

        plot_path = os.path.join(self.run_dir, "training_curves.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"Saved plots to {plot_path}")
