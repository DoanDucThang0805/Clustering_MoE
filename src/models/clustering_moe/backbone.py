import torch
import torch.nn as nn
from torchvision import models
import timm


class Mobilenetv3SmallBackboneTorchvision(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        self.pretrained = pretrained
        if self.pretrained:
            self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            self.model = models.mobilenet_v3_small()

    
    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x
        

class Mobilenetv3SmallBackboneTimm(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        self.model = timm.create_model(
            model_name  = "tf_mobilenetv3_small_100.in1k",
            pretrained  = pretrained,
            num_classes = 0
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.forward_features(x)
        x = self.model.global_pool(x)
        x = torch.flatten(x, 1)
        return x
    


# Unit test for checking the output dimensions of the backbones
if __name__ == "__main__":
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Test torchvision backbone
    backbone_torchvision = Mobilenetv3SmallBackboneTorchvision(pretrained=False)
    output_torchvision = backbone_torchvision(dummy_input)
    print(f"Torchvision Backbone Output Shape: {output_torchvision.shape}")

    # Test timm backbone
    backbone_timm = Mobilenetv3SmallBackboneTimm(pretrained=False)
    output_timm = backbone_timm(dummy_input)
    print(f"Timm Backbone Output Shape: {output_timm.shape}")
