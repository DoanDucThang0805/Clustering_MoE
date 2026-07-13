"""Export a trained CP7 Soft MoE checkpoint to ONNX and verify parity.

Kiến trúc mới (classifier-side, all-expert): model nhận (image, context) và trả
(logits, gate_weights) — graph ONNX expose 2 input, 1 output logits.
Checkpoint chọn theo VALIDATION accuracy (cùng protocol với cp7_results).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from benchmark.cp7_results import _best_run_by_val
from models.soft_moe import build_soft_moe_from_checkpoint


ROOT = Path(__file__).resolve().parents[2]


class LogitsOnlyWrapper(nn.Module):
    """Expose only classifier logits to the ONNX graph."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(image, context)
        return logits


def export_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    opset_version: int = 18,
) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_soft_moe_from_checkpoint(checkpoint).cpu().eval()
    wrapper = LogitsOnlyWrapper(model).eval()
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_context = torch.randn(1, int(checkpoint["context_dim"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected = wrapper(dummy_image, dummy_context).numpy()
        torch.onnx.export(
            wrapper,
            (dummy_image, dummy_context),
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["image", "context"],
            output_names=["logits"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "context": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            dynamo=False,
        )

    onnx_model = onnx.load(str(output_path), load_external_data=True)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(
        None,
        {"image": dummy_image.numpy(), "context": dummy_context.numpy()},
    )[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    if actual.shape != (1, int(checkpoint["num_classes"])):
        raise RuntimeError(f"Unexpected ONNX output shape: {actual.shape}")

    print(f"ONNX export verified: {output_path}")
    print(f"Maximum absolute parity error: {np.max(np.abs(actual - expected)):.3e}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CP7 Soft MoE to ONNX")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", type=str, default="plantdoc")
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path,
                        help="Đường dẫn checkpoint tường minh; mặc định chọn run "
                             "có VAL accuracy cao nhất của seed (cùng protocol cp7_results)")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "onnx_models/soft_moe_mobilenetv3_torchvision_backbone.onnx",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path, best_val, n_runs = _best_run_by_val(
            args.seed, args.dataset_name, args.num_experts
        )
        print(f"Chọn {checkpoint_path.parent.name} (best-of-{n_runs} theo VAL={best_val:.2f}%)")
    export_checkpoint(checkpoint_path, args.out)


if __name__ == "__main__":
    main()
