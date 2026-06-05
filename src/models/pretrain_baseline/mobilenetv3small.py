import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchinfo import summary

weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = mobilenet_v3_small(weights=weights)
model.classifier[-1] = torch.nn.Linear(in_features=1024, out_features=8)
summary(model, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "mult_adds"])
