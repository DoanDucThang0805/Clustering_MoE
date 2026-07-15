from .mobilenetv3small import model as mobilenetv3small_torchvision
from .mobilenetv3smallv1 import model as mobilenetv3small_timm
from .efficientnetb0 import model as efficientnetb0_torchvision


MODEL_REGISTRY = {
    "mobilenetv3small_torchvision": mobilenetv3small_torchvision,
    "mobilenetv3small_timm": mobilenetv3small_timm,
    "efficientnetb0_torchvision": efficientnetb0_torchvision,
}