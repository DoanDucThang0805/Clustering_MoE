from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
import timm


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


class Mobilenetv3SmallBackboneTorchvision(nn.Module):
    def __init__(self, pretrained: bool, checkpoint_path: Path):
        super().__init__()
        self.pretrained = pretrained
        if self.pretrained:
            self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            self.model = load_checkpoint(self.model, checkpoint_path)
        else:
            self.model = models.mobilenet_v3_small()
            self.model = load_checkpoint(self.model, checkpoint_path)
    
    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x
        

class Mobilenetv3SmallBackboneTimm(nn.Module):
    def __init__(self, pretrained: bool, checkpoint_path: Path):
        super().__init__()
        self.model = timm.create_model(
            model_name  = "mobilenetv3_small_100.lamb_in1k",
            pretrained  = pretrained,
            num_classes = 0
        )
        self.model = load_checkpoint(self.model, checkpoint_path)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.forward_features(x)
        x = self.model.global_pool(x)
        x = torch.flatten(x, 1)
        return x
