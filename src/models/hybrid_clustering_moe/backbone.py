import torch
import torch.nn as nn
from torchvision import models
import timm


class Mobilenetv3SmallBackboneTorchvision(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        self.pretrained = pretrained
        self.output_dim = 576

        if self.pretrained:
            self.model=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
        else:
            self.model=models.mobilenet_v3_small()

    
    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x
        

class Mobilenetv3SmallBackboneTimm(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        self.output_dim = 576
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
