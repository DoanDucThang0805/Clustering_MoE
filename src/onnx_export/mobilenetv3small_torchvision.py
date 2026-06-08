from pathlib import Path
import torch
from models.non_pretrain_baseline.model_registry import MODEL_REGISTRY


def export_to_onnx(
    model,
    output_path,
    input_shape=(1, 3, 224, 224),
    opset_version=18,
):
    model.eval()

    device = next(model.parameters()).device

    dummy_input = torch.randn(
        *input_shape,
        device=device,
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["features"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "features": {0: "batch_size"},
        },
    )
    
model = MODEL_REGISTRY["mobilenetv3small_torchvision"]


if __name__ == "__main__":

    output_path = Path(__file__).parents[2] / "onnx_models" / "mobilenetv3small_torchvision.onnx"
    export_to_onnx(model, output_path)
