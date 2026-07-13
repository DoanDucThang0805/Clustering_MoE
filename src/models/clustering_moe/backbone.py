import torch
import torch.nn as nn
from torchvision import models
import timm
from pathlib import Path


class Mobilenetv3SmallBackboneTorchvision(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        self.pretrained = pretrained
        if self.pretrained:
            self.model = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            )
        else:
            self.model = models.mobilenet_v3_small()

    
    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x


    def load_dense_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Load only the feature extractor from a fine-tuned dense model."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Dense backbone checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        feature_state_dict = {
            key.removeprefix("features."): value
            for key, value in state_dict.items()
            if key.startswith("features.")
        }

        if not feature_state_dict:
            raise ValueError(
                f"No features.* weights found in dense checkpoint: {checkpoint_path}"
            )

        self.model.features.load_state_dict(feature_state_dict, strict=True)
        return len(feature_state_dict)
        

class Mobilenetv3SmallBackboneTimm(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        self.model = timm.create_model(
            model_name  = "mobilenetv3_small_100.lamb_in1k",
            pretrained  = pretrained,
            num_classes = 0
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.forward_features(x)
        x = self.model.global_pool(x)
        x = torch.flatten(x, 1)
        return x


class EfficientNetB0BackboneTorchvision(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        self.pretrained = pretrained
        if self.pretrained:
            self.model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
        else:
            self.model = models.efficientnet_b0()


    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x


    def load_dense_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Load only the feature extractor from a fine-tuned dense model."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Dense backbone checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        feature_state_dict = {
            key.removeprefix("features."): value
            for key, value in state_dict.items()
            if key.startswith("features.")
        }

        if not feature_state_dict:
            raise ValueError(
                f"No features.* weights found in dense checkpoint: {checkpoint_path}"
            )

        self.model.features.load_state_dict(feature_state_dict, strict=True)
        return len(feature_state_dict)


class EfficientNetB0BackboneTimm(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        self.model = timm.create_model(
            model_name  = "efficientnet_b0.ra_in1k",
            pretrained  = pretrained,
            num_classes = 0,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)   # num_classes=0 -> (B, 1280) pooled features


    def load_dense_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Load backbone (mọi key TRỪ classifier.*) từ dense timm checkpoint.
        timm efficientnet dùng conv_stem/blocks/conv_head... (không có features.*)."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Dense backbone checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        feature_state_dict = {
            key: value for key, value in state_dict.items()
            if not key.startswith("classifier")
        }

        if not feature_state_dict:
            raise ValueError(
                f"No backbone weights found in dense checkpoint: {checkpoint_path}"
            )

        self.model.load_state_dict(feature_state_dict, strict=True)
        return len(feature_state_dict)
