from __future__ import annotations

from typing import Literal

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
    top_k : int
        Number of top experts to select.
    temperature : float
        Softmax temperature for gating weights.
    metric : str
        Similarity metric: 'cosine' or 'euclidean'.
    """

    def __init__(
        self,
        centroids: torch.Tensor,
        top_k: int,
        temperature: float = 0.5,
        metric: Literal["cosine", "euclidean"] = "cosine",
    ):
        super().__init__()

        # Validate metric
        valid_metrics = {"cosine", "euclidean"}
        if metric not in valid_metrics:
            raise ValueError(
                f"Unsupported metric: {metric}. "
                f"Choose from {valid_metrics}"
            )

        # Validate top_k
        num_clusters = centroids.shape[0]
        if top_k > num_clusters:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed "
                f"num_clusters ({num_clusters})"
            )

        self.top_k = top_k
        self.temperature = temperature
        self.metric = metric

        # Register centroids as non-trainable buffer
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
        x_norm = F.normalize(x, p=2, dim=-1)
        centroid_norm = F.normalize(self.centroids, p=2, dim=-1)
        scores = torch.matmul(x_norm, centroid_norm.T)
        return scores


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
        scores = -(torch.cdist(x, self.centroids, p=2) ** 2)
        return scores


    def forward(
        self, x: torch.Tensor
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
        # Compute similarity scores
        if self.metric == "cosine":
            scores = self.compute_cosine_scores(x)
        elif self.metric == "euclidean":
            scores = self.compute_euclidean_scores(x)
        else:
            raise ValueError(
                f"Unsupported metric: {self.metric}"
            )

        # Select top-k experts
        top_scores, top_indices = torch.topk(
            scores, k=self.top_k, dim=-1
        )

        # Compute gating weights with temperature scaling
        weights = F.softmax(top_scores / self.temperature, dim=-1)

        return weights, top_indices, scores
    

if __name__ == "__main__":

    print("=" * 80)
    print("TEST CLUSTER PROTOTYPE GATING")
    print("=" * 80)

    # --------------------------------------------------
    # Dummy data
    # --------------------------------------------------

    batch_size = 8
    embedding_dim = 576

    num_clusters = 4
    top_k = 2

    torch.manual_seed(42)

    centroids = torch.randn(num_clusters, embedding_dim)

    features = torch.randn(batch_size, embedding_dim)

    # ==================================================
    # Test Cosine Routing
    # ==================================================

    print("\n[Cosine Routing]")

    cosine_router = ClusterPrototypeGating(
        centroids=centroids,
        top_k=top_k,
        temperature=0.5,
        metric="cosine",
    )

    weights, top_indices, scores = cosine_router(features)

    print(f"weights shape     : {weights.shape}")

    print(f"top_indices shape : {top_indices.shape}")

    print(f"topk_indices: {top_indices}")

    print(f"scores shape      : {scores.shape}")

    print("\nWeights Sum:")

    print(weights.sum(dim=1))

    # Assertions
    assert weights.shape == (batch_size,top_k,)

    assert top_indices.shape == (
        batch_size,
        top_k,
    )

    assert scores.shape == (
        batch_size,
        num_clusters,
    )

    assert torch.allclose(
        weights.sum(dim=1),
        torch.ones(batch_size),
        atol=1e-6,
    )

    print(
        "\n✓ Cosine routing passed"
    )

    # ==================================================
    # Test Euclidean Routing
    # ==================================================

    print("\n[Euclidean Routing]")

    euclidean_router = ClusterPrototypeGating(
        centroids=centroids,
        top_k=top_k,
        temperature=0.5,
        metric="euclidean",
    )

    weights, top_indices, scores = euclidean_router(
        features
    )

    print(f"weights shape     : {weights.shape}")

    print(f"top_indices shape : {top_indices.shape}")

    print(f"topk_indices: {top_indices}")


    print(f"scores shape      : {scores.shape}")

    print(f"\nWeights Sum: {weights.sum(dim=1)}")

    print(weights.sum(dim=1))

    assert weights.shape == (
        batch_size,
        top_k,
    )

    assert top_indices.shape == (
        batch_size,
        top_k,
    )

    assert scores.shape == (
        batch_size,
        num_clusters,
    )

    assert torch.allclose(
        weights.sum(dim=1),
        torch.ones(batch_size),
        atol=1e-6,
    )

    print(
        "\n✓ Euclidean routing passed"
    )

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
    