from .mobilenetv3small import model as mobilenetv3small_torchvision
from .mobilenetv3smallv1 import model as mobilenetv3small_timm


MODEL_REGISTRY = {
    "mobilenetv3small_torchvision": mobilenetv3small_torchvision,
    "mobilenetv3small_timm": mobilenetv3small_timm,
}