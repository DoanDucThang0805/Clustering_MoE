import torch
from thop import profile, clever_format

from models.clustering_moe.model import ClusteringMoEModel

model = ClusteringMoEModel(
    num_classes=8,
    centroids=torch.randn(4, 576),
    top_k=2,
    backbone_name="mobilenetv3small_torchvision",
    metric="cosine",
    pretrain_backbone=False,
    temperature=0.5,
)

model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

macs, params = profile(
    model,
    inputs=(dummy_input,),
    verbose=False,
)

macs, params = clever_format(
    [macs, params],
    "%.4f"
)

print(f"MACs:   {macs}")
print(f"Params: {params}")
