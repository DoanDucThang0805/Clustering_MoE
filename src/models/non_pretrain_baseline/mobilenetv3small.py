import torch
from torchvision.models import mobilenet_v3_small
from torchinfo import summary


model = mobilenet_v3_small()
model.classifier[-1] = torch.nn.Linear(in_features=1024, out_features=8)
summary(model, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "mult_adds"])
