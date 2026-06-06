import torch
import torch.nn as nn
from torchvision import models
from torchinfo import summary

checkpoint_path = "/media/data/minhht/clustering_moe/checkpoints/plantdoc/non_pretrain_baseline/mobilenetv3small_torchvision/seed_42/run_20260531-135810/best_checkpoint.pth"

checkpoint = torch.load(checkpoint_path, map_location="cuda")

model = models.mobilenet_v3_small()
model.classifier[-1] = nn.Linear(in_features=1024, out_features=8)

model.load_state_dict(checkpoint["model_state_dict"])