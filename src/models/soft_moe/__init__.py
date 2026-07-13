"""Soft Mixture-of-Experts (classifier-side) — CP7 baseline. Package tự chứa."""

from .gating import SoftGating
from .model import (
    ExpertMLP,
    SoftMoELayer,
    SoftMoEModel,
    build_soft_moe_from_checkpoint,
)

__all__ = [
    "SoftGating",
    "ExpertMLP",
    "SoftMoELayer",
    "SoftMoEModel",
    "build_soft_moe_from_checkpoint",
]
