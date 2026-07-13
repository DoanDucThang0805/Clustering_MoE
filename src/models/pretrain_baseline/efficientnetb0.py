import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchinfo import summary

weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)
model.classifier[-1] = torch.nn.Linear(in_features=1280, out_features=8)
summary(model, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "mult_adds"])
