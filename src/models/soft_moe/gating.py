"""Soft gating cho Soft MoE (classifier-side).

TỰ CHỨA — không import gì từ `models/moe/*`.

Khác `ContextAwareLinearGating` của MoE ở đúng 2 điểm (đây là bản chất "soft assignment"):
  1. KHÔNG noise — noise chỉ để explore lựa chọn rời rạc của top-k; soft không có lựa chọn.
  2. KHÔNG top-k — softmax trên TOÀN BỘ expert ⇒ mọi expert luôn có trọng số > 0 (all-active).

Đường fusion (embedding + context) giữ y hệt gate của MoE để đảm bảo "cùng backbone / cùng
điều kiện", chỉ thay cơ chế assignment.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftGating(nn.Module):
    """Context-aware soft gate: fuse (feature, context) -> softmax trên TẤT CẢ expert."""

    def __init__(
        self,
        model_dim: int,
        context_dim: int,
        num_experts: int,
        temperature: float = 0.5,
        context_proj_dim: int = 32,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.model_dim = model_dim
        self.context_dim = context_dim
        self.num_experts = num_experts
        self.temperature = temperature

        fusion_dim = model_dim + context_proj_dim

        self.embedding_norm = nn.LayerNorm(model_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.context_projector = nn.Sequential(
            nn.Linear(context_dim, context_proj_dim),
            nn.GELU(),
            nn.Linear(context_proj_dim, context_proj_dim),
        )
        self.context_proj_norm = nn.LayerNorm(context_proj_dim)
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        # KHÔNG có noise_layer (so với gate của MoE) — soft không cần explore.
        self.gate_projector = nn.Linear(fusion_dim, num_experts)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        x       : [B, model_dim]   feature đã pooled
        context : [B, context_dim] context feature (brightness/blur/edge/sat/green...)

        Returns
        -------
        weights : [B, num_experts]  softmax trên TOÀN BỘ expert (sum = 1, mọi phần tử > 0)
        logits  : [B, num_experts]  raw gate logits
        """
        embedding = self.embedding_norm(x)
        context = self.context_norm(context)

        context_features = self.context_projector(context)
        context_features = self.context_proj_norm(context_features)

        fusion_features = torch.cat([embedding, context_features], dim=-1)
        fusion_features = self.fusion_norm(fusion_features)

        logits = self.gate_projector(fusion_features)

        # SOFT ASSIGNMENT: softmax trên TẤT CẢ expert (không topk, không noise)
        weights = F.softmax(logits / self.temperature, dim=-1)
        return weights, logits
