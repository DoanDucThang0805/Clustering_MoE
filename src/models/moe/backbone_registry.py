from .backbone import (Mobilenetv3SmallBackboneTimm, Mobilenetv3SmallBackboneTorchvision,
                       EfficientNetB0BackboneTorchvision, EfficientNetB0BackboneTimm)


BACKBONE_REGISTRY = {
    "mobilenetv3small_timm": Mobilenetv3SmallBackboneTimm,
    "mobilenetv3small_torchvision": Mobilenetv3SmallBackboneTorchvision,
    "efficientnetb0_torchvision": EfficientNetB0BackboneTorchvision,
    "efficientnetb0_timm": EfficientNetB0BackboneTimm,
}
