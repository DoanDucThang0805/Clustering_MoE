"""Soft MoE classifier-side (CP7 baseline).

TỰ CHỨA — chỉ dùng chung `BACKBONE_REGISTRY` (bắt buộc, để thoả yêu cầu "cùng backbone"
của PDF mục X). KHÔNG import gì từ `models/moe/*`, `loss/loss_fn.py`, `utils/moe_trainer.py`.

Giống `MoEModel` (learned-gate MoE) ở MỌI thứ — backbone, expert MLP, norms, classifier,
gate context-aware — chỉ khác đúng cơ chế assignment:
  1. Active FULL expert (bỏ top-k)
  2. Bỏ dispatch/mask (mọi mẫu qua mọi expert)
  3. Không load-balance/aux loss (xử lý ở phía training: CrossEntropy thuần)

⇒ so với `learned_gate_moe` (top-2), Soft MoE cô lập đúng biến **sparsity**.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from models.clustering_moe.backbone_registry import BACKBONE_REGISTRY

from .gating import SoftGating


class ExpertMLP(nn.Module):
    """Expert feed-forward — giữ y hệt expert của MoE/Cluster-MoE (capacity-matched)."""

    def __init__(self, model_dim: int, hidden_dim: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class SoftMoELayer(nn.Module):
    """Soft MoE layer: KHÔNG dispatch, KHÔNG mask — mọi expert chạy mọi mẫu."""

    def __init__(
        self,
        model_dim: int,
        context_dim: int,
        num_experts: int = 4,
        hidden_dim: int = 1024,
        temperature: float = 0.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim

        self.gating = SoftGating(
            model_dim=model_dim,
            context_dim=context_dim,
            num_experts=num_experts,
            temperature=temperature,
        )
        self.experts = nn.ModuleList(
            [ExpertMLP(model_dim, hidden_dim, dropout) for _ in range(num_experts)]
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x       : [B, D]
        context : [B, context_dim]

        Returns (moe_output [B, D], weights [B, E], logits [B, E])
        """
        weights, logits = self.gating(x, context)   # [B, E] — softmax trên TẤT CẢ expert

        # ALL-EXPERT: không mask, không dispatch — mọi expert xử lý mọi mẫu.
        moe_output = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            expert_out = self.experts[expert_idx](x)                 # [B, D]
            moe_output = moe_output + weights[:, expert_idx : expert_idx + 1] * expert_out

        return moe_output, weights, logits


class SoftMoEModel(nn.Module):
    """MobileNetV3-Small + Soft MoE (classifier-side, all-expert).

    Pipeline y hệt MoEModel:
        backbone -> pre_moe_norm -> SoftMoELayer -> residual -> post_moe_norm -> classifier
    """

    def __init__(
        self,
        num_classes: int,
        context_dim: int = 6,
        num_experts: int = 4,
        backbone_name: Literal[
            "mobilenetv3small_torchvision", "mobilenetv3small_timm"
        ] = "mobilenetv3small_torchvision",
        pretrain_backbone: bool = True,
        temperature: float = 0.5,
        expert_hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.num_classes = num_classes
        self.context_dim = context_dim
        self.num_experts = num_experts
        self.backbone_name = backbone_name
        self.pretrain_backbone = pretrain_backbone
        self.temperature = temperature
        self.expert_hidden_dim = expert_hidden_dim

        # Cùng backbone với MoE/Cluster-MoE (yêu cầu "cùng backbone" của PDF).
        self.feature_extractor = BACKBONE_REGISTRY[backbone_name](pretrain_backbone)
        # Tự dò model_dim bằng 1 forward giả — backbone của clustering_moe không có
        # thuộc tính `output_dim`, và ta KHÔNG import gì từ models/moe (giữ cô lập).
        model_dim = self._infer_model_dim(self.feature_extractor)
        self.model_dim = model_dim

        self.pre_moe_norm = nn.LayerNorm(model_dim)
        self.soft_moe_layer = SoftMoELayer(
            model_dim=model_dim,
            context_dim=context_dim,
            num_experts=num_experts,
            hidden_dim=expert_hidden_dim,
            temperature=temperature,
        )
        self.post_moe_norm = nn.LayerNorm(model_dim)

        # Classifier stack y hệt MoEModel / ClusteringMoEModel.
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _infer_model_dim(backbone: nn.Module) -> int:
        """Dò số chiều feature của backbone bằng 1 forward giả (224x224)."""
        was_training = backbone.training
        backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            dim = int(backbone(dummy).shape[1])
        if was_training:
            backbone.train()
        return dim

    def forward(
        self,
        images: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Trả (class_logits [B, C], gate_weights [B, E]).

        Interface RIÊNG, sạch — KHÔNG trả `top_k_indices` giả như MoE.
        """
        features = self.feature_extractor(images)          # [B, D] (đã pooled)
        normalized = self.pre_moe_norm(features)
        moe_output, weights, _ = self.soft_moe_layer(normalized, context)

        residual = features + moe_output
        normalized_out = self.post_moe_norm(residual)
        class_logits = self.classifier(normalized_out)
        return class_logits, weights


def build_soft_moe_from_checkpoint(checkpoint: dict) -> SoftMoEModel:
    """Dựng lại SoftMoEModel từ metadata trong checkpoint (dùng cho inference/benchmark)."""
    required = {
        "num_classes",
        "num_experts",
        "context_dim",
        "backbone_name",
        "expert_hidden_dim",
        "temperature",
        "model_state_dict",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"Soft MoE checkpoint thiếu key: {sorted(missing)}")

    model = SoftMoEModel(
        num_classes=int(checkpoint["num_classes"]),
        context_dim=int(checkpoint["context_dim"]),
        num_experts=int(checkpoint["num_experts"]),
        backbone_name=str(checkpoint["backbone_name"]),
        # Trọng số backbone đã train được nạp bên dưới — không cần tải ImageNet.
        pretrain_backbone=False,
        temperature=float(checkpoint["temperature"]),
        expert_hidden_dim=int(checkpoint["expert_hidden_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
