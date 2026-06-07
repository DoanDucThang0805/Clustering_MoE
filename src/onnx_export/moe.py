from pathlib import Path
import torch
from models.moe.model_adapt_onnx import MoEModel


from pathlib import Path
from typing import Optional

import torch


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    context_dim: Optional[int] = None,
    opset_version: int = 18,
) -> None:

    model.eval()

    dummy_input = torch.randn(*input_shape)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if context_dim is not None:
        dummy_context = torch.randn(
            input_shape[0],
            context_dim,
        )

        inputs = (dummy_input, dummy_context)

        input_names = [
            "image",
            "context",
        ]

        dynamic_axes = {
            "image": {
                0: "batch_size",
            },
            "context": {
                0: "batch_size",
            },
            "logits": {
                0: "batch_size",
            },
            "router_logits": {
                0: "batch_size",
            },
            "top_indices": {
                0: "batch_size",
            },
        }

    else:
        inputs = dummy_input

        input_names = [
            "image",
        ]

        dynamic_axes = {
            "image": {
                0: "batch_size",
            },
            "logits": {
                0: "batch_size",
            },
            "router_logits": {
                0: "batch_size",
            },
            "top_indices": {
                0: "batch_size",
            },
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=[
                "logits",
                "router_logits",
                "top_indices",
            ],
            dynamic_axes=dynamic_axes,
        )

    print(
        f"ONNX model exported to: {output_path}"
    )

model = MoEModel(
    context_dim       = 6,
    num_classes       = 8,
    backbone_name     = "mobilenetv3small_torchvision",
    num_experts       = 4,
    pretrain_backbone = False,
    router_mode       = "context_aware",
    temperature       = 0.5,
    top_k             = 2,
)


if __name__ == "__main__":

    output_path = (
        Path(__file__).parents[2]
        / "onnx_models"
        / "moe_mobilenetv3_torchvision_backbone.onnx"
    )

    export_to_onnx(
        model=model,
        output_path=output_path,
        context_dim=6,
    )