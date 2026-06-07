from typing import Literal

import torch
import torch.nn as nn
from torchinfo import summary

from .backbone_registry import BACKBONE_REGISTRY
from .gating import HybridMoEGating



class MoeLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        context_dim: int,
        num_experts: int,
        centroids: torch.Tensor,
        top_k: int,
        noise_stddev: float = 1.0,
        context_proj_dim: int = 32,
        lambda_: float = 0.5,
        temperature: float = 1.0,
        metric: Literal["cosine", "euclidean"] = "cosine",
    ):
        super().__init__()

        self.model_dim = model_dim
        self.context_dim = context_dim
        self.num_experts = num_experts
        self.centroids = centroids
        self.top_k = top_k
        self.noise_stddev = noise_stddev
        self.context_proj_dim = context_proj_dim
        self.lambda_ = lambda_
        self.temperature = temperature
        self.metric = metric

        self.gating = HybridMoEGating(
            model_dim=model_dim,
            context_dim=context_dim,
            num_experts=num_experts,
            centroids=centroids,
            top_k=top_k,
            noise_stddev=noise_stddev,
            context_proj_dim=context_proj_dim,
            lambda_=lambda_,
            temperature=temperature,
            metric=metric
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, 1024),  # Expand to 2x dimension
                nn.LayerNorm(1024),
                nn.GELU(),                   # Non-linear activation
                nn.Dropout(0.1),             # Regularization
                nn.Linear(1024, model_dim)   # Contract back to original dimension
            ) 
            for _ in range(num_experts)
        ])


    def forward(self, x: torch.Tensor, context: torch.Tensor):
        weights, top_k_indices, hybrid_logits = self.gating(x, context)
        # weights:       [B, top_k]
        # top_k_indices: [B, top_k]
        # hybrid_logits: [B, G]

        B, D  = x.shape
        output = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]        # [B]
            w          = weights[:, k].unsqueeze(-1) # [B, 1]

            for g in range(self.num_experts):
                mask = (expert_idx == g)             # [B] bool mask
                if mask.any():
                    expert_out    = self.experts[g](x[mask])  # [n_selected, D]
                    output[mask] += w[mask] * expert_out

        return output, weights, top_k_indices, hybrid_logits        


class HybridMoEModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        context_dim: int,
        num_experts: int,
        centroids: torch.Tensor,
        top_k: int,
        backbone_name: Literal["mobilenetv3small_torchvision", "mobilenetv3small_timm"],
        pretrain_backbone: bool,
        noise_stddev: float = 1.0,
        context_proj_dim: int = 32,
        lambda_: float = 0.5,
        temperature: float = 1.0,
        metric: Literal["cosine", "euclidean"] = "cosine",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.context_dim = context_dim
        self.num_experts = num_experts
        self.centroids = centroids
        self.top_k = top_k
        self.noise_stddev = noise_stddev
        self.context_proj_dim = context_proj_dim
        self.lambda_ = lambda_
        self.temperature = temperature
        self.metric = metric
        self.backbone_name = backbone_name
        self.pretrain_backbone = pretrain_backbone

        self.backbone = BACKBONE_REGISTRY[backbone_name](pretrain_backbone)
        self.model_dim = self.backbone.output_dim

        self.moe_layer = MoeLayer(
            model_dim        = self.model_dim,
            context_dim      = context_dim,
            num_experts      = num_experts,
            centroids        = centroids,
            top_k            = top_k,
            noise_stddev     = noise_stddev,
            context_proj_dim = context_proj_dim,
            lambda_          = lambda_,
            temperature      = temperature,
            metric           = metric,
        )

        self.norm = nn.LayerNorm(self.model_dim)

        self.classifier = nn.Sequential(
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


    def forward(
        self,
        x:       torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input images. Shape: [B, 3, 224, 224]
        context : torch.Tensor
            Context features. Shape: [B, C]

        Returns
        -------
        logits : torch.Tensor
            Class logits. Shape: [B, num_classes]
        weights : torch.Tensor
            Gating weights. Shape: [B, top_k]
        top_indices : torch.Tensor
            Selected expert indices. Shape: [B, top_k]
        hybrid_logits : torch.Tensor
            Hybrid routing scores. Shape: [B, G]
        """
        embedding = self.backbone(x)                                             # [B, D]
        moe_output, weights, top_indices, hybrid_logits = self.moe_layer(        # [B, D]
            embedding, context
        )
        residual   = embedding + moe_output                                      # [B, D]
        normalized = self.norm(residual)                                         # [B, D]
        logits     = self.classifier(normalized)                                 # [B, num_classes]

        return logits, weights, top_indices, hybrid_logits


if __name__ == "__main__":
    dummy_input     = torch.rand(1, 3, 224, 224)
    dummy_context   = torch.rand(1, 6)
    dummy_centroids = torch.randn(4, 576)

    model = HybridMoEModel(
        num_classes      = 8,
        context_dim      = 6,
        num_experts      = 4,
        centroids        = dummy_centroids,
        top_k            = 2,
        backbone_name    = "mobilenetv3small_torchvision",
        pretrain_backbone= False,
        noise_stddev     = 1.0,
        context_proj_dim = 32,
        lambda_          = 0.5,
        temperature      = 0.5,
        metric           = "cosine",
    )
    summary(model, input_data=[dummy_input, dummy_context])
