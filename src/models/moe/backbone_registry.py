from .backbone import Mobilenetv3SmallBackboneTimm, Mobilenetv3SmallBackboneTorchvision


BACKBONE_REGISTRY = {
    "mobilenetv3small_timm": Mobilenetv3SmallBackboneTimm,
    "mobilenetv3small_torchvision": Mobilenetv3SmallBackboneTorchvision,
}
