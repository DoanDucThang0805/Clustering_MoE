from .mobilenetv3small import model as mobilenetv3small_torchvision
from .mobilenetv3smallv1 import model as mobilenetv3small_timm
from .mobilenetv3smallv2 import model as mobilenetv3small_timm_lamb1k
from .efficientnetb0 import model as efficientnetb0_torchvision
from .efficientnetb0_timm import model as efficientnetb0_timm

MODEL_REGISTRY = {
    "mobilenetv3small_torchvision": mobilenetv3small_torchvision,
    "mobilenetv3small_timm": mobilenetv3small_timm,
    "mobilenetv3small_timm_retrain1": mobilenetv3small_timm,
    "mobilenetv3small_timm_lamb1k": mobilenetv3small_timm_lamb1k,
    "mobilenetv3small_timm_lamb1k_retrain1": mobilenetv3small_timm_lamb1k,
    "mobilenetv3small_timm_lamb1k_retrain2": mobilenetv3small_timm_lamb1k,
    "efficientnetb0_torchvision": efficientnetb0_torchvision,
    "efficientnetb0_timm": efficientnetb0_timm,
}
