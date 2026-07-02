from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from .backbone_registry import BACKBONE_REGISTRY
from .model import MoELayer


class GlobalResidualClusteringMoEModel(nn.Module):
    """Cluster-MoE that learns a residual on top of a dense baseline head."""

    def __init__(
        self,
        num_classes: int,
        centroids: torch.Tensor,
        top_k: int,
        backbone_name: Literal[
            "mobilenetv3small_timm",
            "mobilenetv3small_torchvision",
        ],
        metric: Literal["cosine", "euclidean"],
        pretrain_backbone: bool,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if backbone_name != "mobilenetv3small_torchvision":
            raise ValueError(
                "Dense global-head initialization currently supports only "
                "mobilenetv3small_torchvision."
            )

        self.backbone = BACKBONE_REGISTRY[backbone_name](
            pretrained=pretrain_backbone
        )
        self.moe_layer = MoELayer(
            centroids=centroids,
            top_k=top_k,
            metric=metric,
            temperature=temperature,
        )

        self.model_dim = centroids.shape[1]
        self.num_experts = centroids.shape[0]
        self.num_classes = num_classes

        # Matches torchvision MobileNetV3-Small's dense classifier exactly.
        self.global_classifier = nn.Sequential(
            nn.Linear(self.model_dim, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes),
        )

        self.moe_classifier = nn.Sequential(
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
        self.moe_norm = nn.LayerNorm(self.model_dim)

        # tanh(0) = 0, so initialization reproduces the dense baseline logits.
        # The bounded learnable scale lets the MoE branch become active safely.
        self.residual_logit_scale = nn.Parameter(torch.zeros(()))

        self.backbone_checkpoint_path: str | None = None
        self.centroid_backbone_type: str | None = None
        self._dense_branch_frozen = False

    def load_dense_checkpoint(self, checkpoint_path: str | Path) -> tuple[int, int]:
        """Load the feature extractor and dense classifier from one checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Dense checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        feature_state_dict = {
            key.removeprefix("features."): value
            for key, value in state_dict.items()
            if key.startswith("features.")
        }
        classifier_state_dict = {
            key.removeprefix("classifier."): value
            for key, value in state_dict.items()
            if key.startswith("classifier.")
        }

        if not feature_state_dict:
            raise ValueError(
                f"No features.* weights found in {checkpoint_path}"
            )
        if not classifier_state_dict:
            raise ValueError(
                f"No classifier.* weights found in {checkpoint_path}"
            )

        self.backbone.model.features.load_state_dict(
            feature_state_dict,
            strict=True,
        )
        self.global_classifier.load_state_dict(
            classifier_state_dict,
            strict=True,
        )
        self.backbone_checkpoint_path = str(checkpoint_path.resolve())

        return len(feature_state_dict), len(classifier_state_dict)

    @property
    def residual_scale(self) -> torch.Tensor:
        return torch.tanh(self.residual_logit_scale)

    def freeze_dense_branch(self) -> None:
        """Freeze the baseline feature space and its classifier."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.global_classifier.parameters():
            parameter.requires_grad = False

        self._dense_branch_frozen = True
        self.backbone.eval()
        self.global_classifier.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._dense_branch_frozen:
            # Keep BatchNorm statistics and dense-head dropout fixed.
            self.backbone.eval()
            self.global_classifier.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.backbone(x)
        global_logits = self.global_classifier(embedding)

        moe_output, weights, top_indices, scores = self.moe_layer(embedding)
        moe_embedding = self.moe_norm(embedding + moe_output)
        moe_logits = self.moe_classifier(moe_embedding)

        logits = global_logits + self.residual_scale * moe_logits
        return logits, weights, top_indices, scores
