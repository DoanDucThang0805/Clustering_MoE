from .mobilenetv3small import model as mobilenetv3small_torchvision
from .mobilenetv3smallv1 import model as mobilenetv3small_timm
from .mobilenetv3smallv2 import model as mobilenetv3small_timm_lamb1k

MODEL_REGISTRY = {
    "mobilenetv3small_torchvision": mobilenetv3small_torchvision,
    "mobilenetv3small_timm": mobilenetv3small_timm,
    "mobilenetv3small_timm_lamb1k": mobilenetv3small_timm_lamb1k
}