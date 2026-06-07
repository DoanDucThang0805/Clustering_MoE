from typing import Tuple, Literal, Optional

import torch
import torch.nn as nn
from torchinfo import summary

from .backbone_registry import BACKBONE_REGISTRY
from .gating import NoisyTopKGating, ContextAwareLinearGating

import warnings
warnings.filterwarnings("ignore")


class MoELayer(nn.Module):
    """MoE layer — dense computation for ONNX export.

    Thay thế sparse boolean masking bằng dense torch.stack + torch.gather
    để tương thích hoàn toàn với ONNX.

    Lưu ý: FLOPs cao hơn bản sparse vì tính tất cả expert.
    Chỉ dùng file này để xuất ONNX, không dùng để đo FLOPs.
    """

    def __init__(
        self,
        context_dim: Optional[int],
        model_dim: int,
        num_experts: int,
        top_k: int,
        router_mode: Literal["noisy", "context_aware"],
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.router_mode = router_mode
        self.temperature = temperature
        self.model_dim   = model_dim

        if not (0 < self.top_k <= self.num_experts):
            raise ValueError(
                "top_k must be a positive integer <= num_experts"
            )

        self._initialize_gating(model_dim, context_dim)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(1024, model_dim),
            )
            for _ in range(num_experts)
        ])

    def _initialize_gating(
        self,
        model_dim: int,
        context_dim: Optional[int],
    ) -> None:
        if self.router_mode == "noisy":
            self.gating = NoisyTopKGating(
                model_dim   = model_dim,
                num_experts = self.num_experts,
                top_k       = self.top_k,
                temperature = self.temperature,
            )
        elif self.router_mode == "context_aware":
            self.gating = ContextAwareLinearGating(
                model_dim   = model_dim,
                context_dim = context_dim,
                num_experts = self.num_experts,
                top_k       = self.top_k,
                temperature = self.temperature,
            )
        else:
            raise ValueError(
                f"Invalid router_mode: {self.router_mode}. "
                "Must be 'noisy' or 'context_aware'."
            )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ── Routing ──────────────────────────────
        if self.router_mode == "noisy":
            combined_weights, top_k_indices, clean_router_logits = self.gating(x)
        else:
            combined_weights, top_k_indices, clean_router_logits = self.gating(x, context)

        # ── Dense expert computation ──────────────
        all_outputs = torch.stack(
            [expert(x) for expert in self.experts],
            dim=1,
        )  # [B, num_experts, D]

        idx = top_k_indices.unsqueeze(-1).expand(
            -1, -1, self.model_dim,
        )  # [B, top_k, D]

        selected = torch.gather(
            all_outputs, 1, idx,
        )  # [B, top_k, D]

        moe_output = (
            combined_weights.unsqueeze(-1) * selected
        ).sum(dim=1)  # [B, D]

        return moe_output, clean_router_logits, top_k_indices


class MoEModel(nn.Module):
    """MoEModel — ONNX-exportable version."""

    def __init__(
        self,
        context_dim: Optional[int],
        num_classes: int,
        num_experts: int,
        top_k: int,
        router_mode: Literal["noisy", "context_aware"],
        backbone_name: Literal["mobilenetv3small_torchvision", "mobilenetv3small_timm"],
        pretrain_backbone: bool,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.context_dim       = context_dim
        self.num_classes       = num_classes
        self.num_experts       = num_experts
        self.top_k             = top_k
        self.router_mode       = router_mode
        self.temperature       = temperature
        self.backbone_name     = backbone_name
        self.pretrain_backbone = pretrain_backbone

        self.feature_extractor = BACKBONE_REGISTRY[backbone_name](pretrain_backbone)
        model_dim = self.feature_extractor.output_dim

        self.pre_moe_norm  = nn.LayerNorm(model_dim)
        self.post_moe_norm = nn.LayerNorm(model_dim)

        self.moe_layer = MoELayer(
            context_dim = context_dim,
            model_dim   = model_dim,
            num_experts = num_experts,
            top_k       = top_k,
            router_mode = router_mode,
            temperature = temperature,
        )

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

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature      = self.feature_extractor(x)
        residual     = feature
        feature_norm = self.pre_moe_norm(feature)

        if self.router_mode == "noisy":
            moe_output, clean_router_logits, top_k_indices = self.moe_layer(feature_norm)
        else:
            moe_output, clean_router_logits, top_k_indices = self.moe_layer(feature_norm, context)

        moe_residual      = residual + moe_output
        moe_residual_norm = self.post_moe_norm(moe_residual)
        class_logits      = self.classifier(moe_residual_norm)

        return class_logits, clean_router_logits, top_k_indices


if __name__ == "__main__":
    dummy_input   = torch.rand(1, 3, 224, 224)
    dummy_context = torch.rand(1, 6)

    model = MoEModel(
        context_dim       = 6,
        num_classes       = 8,
        backbone_name     = "mobilenetv3small_torchvision",
        num_experts       = 4,
        pretrain_backbone = False,
        router_mode       = "context_aware",
        temperature       = 0.5,
        top_k             = 2,
    )

    summary(model, input_data=[dummy_input, dummy_context])
