from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from .cluster_gating import ClusterPrototypeGating
from .backbone_registry import BACKBONE_REGISTRY


class MoELayer(nn.Module):
    def __init__(
        self,
        centroids:   torch.Tensor,
        top_k:       int,
        metric:      Literal["cosine", "euclidean"],
        temperature: float = 1.0,
    ):
        super().__init__()
        self.centroids   = centroids
        self.top_k       = top_k
        self.model_dim   = centroids.shape[1]
        self.num_experts = centroids.shape[0]
        self.temperature = temperature

        self._initialize_gating_router(
            centroids   = centroids,
            top_k       = top_k,
            metric      = metric,
            temperature = temperature,
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.model_dim, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(1024, self.model_dim),
            )
            for _ in range(self.num_experts)
        ])


    def _initialize_gating_router(
        self,
        centroids:   torch.Tensor,
        top_k:       int,
        metric:      Literal["cosine", "euclidean"],
        temperature: float = 1.0,
    ) -> None:
        self.gating = ClusterPrototypeGating(
            centroids   = centroids,
            top_k       = top_k,
            metric      = metric,
            temperature = temperature,
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [B, D]

        Returns
        -------
        output : torch.Tensor
            Shape [B, D]
        weights : torch.Tensor
            Shape [B, top_k]
        top_indices : torch.Tensor
            Shape [B, top_k]
        scores : torch.Tensor
            Shape [B, G]
        """
        # ----------------------------------
        # Routing
        # ----------------------------------
        weights, top_indices, scores = self.gating(x)

        # ----------------------------------
        # Expert computation — dense (ONNX compatible)
        # Tính tất cả expert, gather top-k sau
        # ----------------------------------
        all_outputs = torch.stack(
            [expert(x) for expert in self.experts],
            dim=1,
        )  # [B, num_experts, D]

        idx = top_indices.unsqueeze(-1).expand(
            -1, -1, self.model_dim,
        )  # [B, top_k, D]

        selected = torch.gather(
            all_outputs, 1, idx,
        )  # [B, top_k, D]

        output = (
            weights.unsqueeze(-1) * selected
        ).sum(dim=1)  # [B, D]

        return output, weights, top_indices, scores


class ClusteringMoEModel(nn.Module):
    def __init__(
        self,
        num_classes:       int,
        centroids:         torch.Tensor,
        top_k:             int,
        backbone_name:     Literal["mobilenetv3small_timm", "mobilenetv3small_torchvision"],
        metric:            Literal["cosine", "euclidean"],
        pretrain_backbone: bool,
        checkpoint_path:   Path,
        temperature:       float = 1.0,
    ):
        super().__init__()
        self.backbone = BACKBONE_REGISTRY[backbone_name](
            pretrained      = pretrain_backbone,
            checkpoint_path = checkpoint_path,
        )
        self.moe_layer = MoELayer(
            centroids   = centroids,
            top_k       = top_k,
            metric      = metric,
            temperature = temperature,
        )
        self.model_dim   = centroids.shape[1]
        self.num_experts = centroids.shape[0]
        self.num_classes = num_classes
        self.classifier  = nn.Sequential(
            nn.Linear(self.model_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )
        self.norm = nn.LayerNorm(self.model_dim)


    def forward(self, x: torch.Tensor):
        embedding = self.backbone(x)
        moe_output, weights, top_indices, scores = self.moe_layer(embedding)
        residual   = embedding + moe_output
        normalized = self.norm(residual)
        logits     = self.classifier(normalized)
        return logits, weights, top_indices, scores
