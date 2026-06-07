from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusterPrototypeGating(nn.Module):
    """Cluster Prototype Gating Router.

    Routes input features to top-k experts using K-Means centroids.

    Attributes
    ----------
    centroids : torch.Tensor
        Cluster centroids from K-Means. Shape: [G, D]
    num_experts : int
        Number of experts / clusters G.
    top_k : int
        Number of top experts to select.
    temperature : float
        Softmax temperature for gating weights.
    metric : str
        Similarity metric: 'cosine' or 'euclidean'.
    """

    def __init__(
        self,
        centroids:   torch.Tensor,
        top_k:       int,
        temperature: float = 1.0,
        metric:      Literal["cosine", "euclidean"] = "cosine",
    ):
        super().__init__()

        if metric not in {"cosine", "euclidean"}:
            raise ValueError(
                f"Unsupported metric: '{metric}'. "
                f"Choose from {{'cosine', 'euclidean'}}."
            )

        num_experts = centroids.shape[0]
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed "
                f"num_clusters ({num_experts})."
            )

        self.num_experts = num_experts
        self.top_k       = top_k
        self.temperature = temperature
        self.metric      = metric

        self.register_buffer("centroids", centroids.float())


    def compute_cosine_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between features and centroids.

        Parameters
        ----------
        x : torch.Tensor
            Feature embeddings. Shape: [B, D]

        Returns
        -------
        torch.Tensor
            Cosine similarity scores. Shape: [B, G]
        """
        x_norm        = F.normalize(x,              p=2, dim=-1)
        centroid_norm = F.normalize(self.centroids, p=2, dim=-1)
        return torch.matmul(x_norm, centroid_norm.T)


    def compute_euclidean_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute negative squared Euclidean distance.

        Parameters
        ----------
        x : torch.Tensor
            Feature embeddings. Shape: [B, D]

        Returns
        -------
        torch.Tensor
            Negative squared Euclidean distances. Shape: [B, G]
        """
        return -(torch.cdist(x, self.centroids, p=2) ** 2)


    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route input features to top-k experts.

        Parameters
        ----------
        x : torch.Tensor
            Feature embeddings. Shape: [B, D]

        Returns
        -------
        weights : torch.Tensor
            Normalized gating weights. Shape: [B, top_k]
        top_indices : torch.Tensor
            Indices of selected experts. Shape: [B, top_k]
        scores : torch.Tensor
            Similarity scores for all experts. Shape: [B, G]
        """
        if self.metric == "cosine":
            scores = self.compute_cosine_scores(x)
        elif self.metric == "euclidean":
            scores = self.compute_euclidean_scores(x)

        top_scores, top_indices = torch.topk(scores, k=self.top_k, dim=-1)
        weights = F.softmax(top_scores / self.temperature, dim=-1)

        return weights, top_indices, scores


class BaseNoiseGatingMixin:
    """Mixin to handle noise injection logic for gating mechanisms."""

    @staticmethod
    def apply_noise_to_logits(
        clean_logits:   torch.Tensor,
        noise_layer:    nn.Module,
        noise_stddev:   float,
        training:       bool,
        input_features: torch.Tensor,
    ) -> torch.Tensor:
        if not training:
            return clean_logits
        noise_magnitude = noise_layer(input_features)
        noise_scale     = F.softplus(noise_magnitude)
        sampled_noise   = torch.randn_like(clean_logits)
        return clean_logits + noise_scale * sampled_noise * noise_stddev


class ContextAwareLinearGating(nn.Module, BaseNoiseGatingMixin):
    """Context-aware linear gating mechanism for Mixture of Experts."""

    def __init__(
        self,
        model_dim:        int,
        context_dim:      int,
        num_experts:      int,
        top_k:            int,
        temperature:      float = 1.0,
        noise_stddev:     float = 1.0,
        context_proj_dim: int   = 32,
    ) -> None:
        super().__init__()
        assert top_k <= num_experts, "top_k must be <= num_experts"

        self.model_dim    = model_dim
        self.context_dim  = context_dim
        self.num_experts  = num_experts
        self.top_k        = top_k
        self.noise_stddev = noise_stddev
        self.temperature  = temperature

        fusion_dim = model_dim + context_proj_dim

        self.embedding_norm    = nn.LayerNorm(model_dim)
        self.context_norm      = nn.LayerNorm(context_dim)
        self.context_proj_norm = nn.LayerNorm(context_proj_dim)
        self.fusion_norm       = nn.LayerNorm(fusion_dim)

        self.context_projector = nn.Sequential(
            nn.Linear(context_dim, context_proj_dim),
            nn.GELU(),
            nn.Linear(context_proj_dim, context_proj_dim),
        )
        self.noise_layer    = nn.Linear(fusion_dim, num_experts, bias=False)
        self.gate_projector = nn.Linear(fusion_dim, num_experts)


    def forward(
        self,
        x:       torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.embedding_norm(x)
        context   = self.context_norm(context)

        context_features = self.context_projector(context)
        context_features = self.context_proj_norm(context_features)

        fusion_features = torch.cat([embedding, context_features], dim=-1)
        fusion_features = self.fusion_norm(fusion_features)

        clean_logits = self.gate_projector(fusion_features)
        noisy_logits = self.apply_noise_to_logits(
            clean_logits, self.noise_layer, self.noise_stddev,
            self.training, fusion_features,
        )

        top_k_logits, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=-1)
        combined_weights = F.softmax(top_k_logits / self.temperature, dim=-1)

        return combined_weights, top_k_indices, clean_logits


class HybridMoEGating(nn.Module):

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

        self.cluster_prototype_gating = ClusterPrototypeGating(
            centroids = centroids,
            top_k = top_k,
            temperature = temperature,
            metric = metric
        )

        self.moe_gating = ContextAwareLinearGating(
            model_dim = model_dim,
            context_dim = context_dim,
            num_experts = num_experts,
            top_k = top_k,
            temperature = temperature,
            noise_stddev = noise_stddev,
            context_proj_dim = context_proj_dim
        )

    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if self.metric == "cosine":
            score_clustergating = self.cluster_prototype_gating.compute_cosine_scores(x)
        elif self.metric == "euclidean":
            score_clustergating = self.cluster_prototype_gating.compute_euclidean_scores(x)

        x_norm = self.moe_gating.embedding_norm(x)
        context_norm = self.moe_gating.context_norm(context)

        context_features = self.moe_gating.context_projector(context_norm)
        context_features = self.moe_gating.context_proj_norm(context_features)
        
        fusion_features = torch.cat([x_norm, context_features], dim=-1)
        fusion_features = self.moe_gating.fusion_norm(fusion_features)

        clean_logits = self.moe_gating.gate_projector(fusion_features)
        noisy_logits = self.moe_gating.apply_noise_to_logits(
            clean_logits=clean_logits,
            noise_layer=self.moe_gating.noise_layer,
            noise_stddev=self.noise_stddev,
            training=self.moe_gating.training,
            input_features=fusion_features
        )

        sim_norm   = F.normalize(score_clustergating, dim=-1)
        learn_norm = F.normalize(noisy_logits,        dim=-1)
        hybrid_logits = self.lambda_ * sim_norm + (1 - self.lambda_) * learn_norm
        top_k_score, top_k_indices = torch.topk(input=hybrid_logits, k=self.top_k)
        weights = F.softmax(top_k_score/self.temperature, dim=-1)
        return weights, top_k_indices, hybrid_logits
    