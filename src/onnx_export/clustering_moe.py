from pathlib import Path
import torch
from models.clustering_moe.model_adapt_onnx import ClusteringMoEModel


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    opset_version: int = 18,
) -> None:
    """
    Export PyTorch model to ONNX.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.

    output_path : str | Path
        Output ONNX file path.

    input_shape : tuple
        Dummy input shape.

    opset_version : int
        ONNX opset version.
    """

    model.eval()

    dummy_input = torch.randn(*input_shape)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=[
                "logits",
                "weights",
                "top_indices",
                "scores",
            ],
            dynamic_axes={
                "input": {
                    0: "batch_size",
                },
                "logits": {
                    0: "batch_size",
                },
                "weights": {
                    0: "batch_size",
                },
                "top_indices": {
                    0: "batch_size",
                },
                "scores": {
                    0: "batch_size",
                },
            },
        )

    print(
        f"ONNX model exported to: {output_path}"
    )


model = ClusteringMoEModel(
    num_classes=8,
    centroids=torch.randn(4, 576),
    top_k=2,
    backbone_name="mobilenetv3small_torchvision",
    metric="cosine",
    pretrain_backbone=False,
    checkpoint_path=Path("/media/data/minhht/clustering_moe/checkpoints/plantdoc/non_pretrain_baseline/mobilenetv3small_torchvision/seed_42/run_20260531-135810/best_checkpoint.pth"),
    temperature=1.0,
)


if __name__ == "__main__":

    output_path = Path(__file__).parents[2] / "onnx_models" / "clustering_moe_cosine_mobilenetv3_torchvision_backbone.onnx"
    export_to_onnx(model, output_path)
