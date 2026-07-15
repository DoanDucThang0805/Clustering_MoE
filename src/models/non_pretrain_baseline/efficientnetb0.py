import torch
from torchvision.models import efficientnet_b0

model = efficientnet_b0(weights=None)   # FROM SCRATCH — không dùng ImageNet weights
model.classifier[-1] = torch.nn.Linear(in_features=1280, out_features=8)
