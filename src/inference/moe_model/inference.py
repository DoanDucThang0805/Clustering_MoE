from pathlib import Path
import random
from argparse import ArgumentParser, Namespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from models.moe.model import MoEModel


# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────
# Path builders
# ─────────────────────────────────────────────────────────────
def build_checkpoint_path(root_dir: Path, args: Namespace) -> Path:
    return (
        root_dir
        / "checkpoints"
        / args.dataset_name
        / args.type_model
        / f"{args.backbone_name}_moe"
        / f"{args.num_experts}_experts"
        / f"top_{args.top_k}"
        / f"seed_{args.seed}"
        / args.runtime
        / args.checkpoint
    )


def build_output_dir(root_dir: Path, args: Namespace) -> Path:
    return (
        root_dir
        / "reports"
        / args.dataset_name
        / args.type_model
        / f"{args.backbone_name}_moe"
        / f"{args.num_experts}_experts"
        / f"top_{args.top_k}"
        / f"seed_{args.seed}"
        / args.runtime
    )


# ─────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────
def load_model(
    ckpt:      dict,
    args:      Namespace,
) -> MoEModel:
    model = MoEModel(
        context_dim=ckpt["context_dim"],
        num_classes=ckpt["num_classes"],
        num_experts=ckpt["num_experts"],
        top_k=ckpt["top_k"],
        router_mode=ckpt["router_mode"],
        backbone_name=args.backbone_name,
        pretrain_backbone=args.pretrain_backbone,
        temperature=ckpt["temperature"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model


# ─────────────────────────────────────────────────────────────
# Inference engine
# ─────────────────────────────────────────────────────────────
class MoEInference:

    def __init__(
        self,
        model:       MoEModel,
        device:      torch.device,
        output_dir:  Path,
        class_names: list[str] | None = None,
    ):
        self.model       = model.to(device).eval()
        self.device      = device
        self.output_dir  = output_dir
        self.class_names = class_names
        self.output_dir.mkdir(parents=True, exist_ok=True)


    @torch.inference_mode()
    def run(self, loader: DataLoader) -> dict:
        """Run inference over loader, return raw results."""
        all_preds       = []
        all_labels      = []

        for images, labels, contexts in loader:
            images = images.to(self.device)
            contexts = contexts.to(self.device)

            class_logits, _, _ = self.model(images, contexts)
            preds = torch.argmax(class_logits, dim=1).cpu()

            all_preds.append(preds)
            all_labels.append(labels)

        return {
            "preds":       torch.cat(all_preds).numpy(),
            "labels":      torch.cat(all_labels).numpy(),
        }


    def report(self, results: dict) -> None:
        """Print and save classification report."""
        preds  = results["preds"]
        labels = results["labels"]
        acc    = accuracy_score(labels, preds) * 100

        report_str = classification_report(
            labels, preds,
            target_names = self.class_names,
            digits       = 4,
        )

        print(f"\n{'='*60}")
        print(f"  Test Accuracy: {acc:.2f}%")
        print(f"{'='*60}")
        print(report_str)

        path = self.output_dir / "classification_report.txt"
        path.write_text(f"Test Accuracy: {acc:.2f}%\n\n{report_str}")
        print(f"Saved → {path}")


    def plot_confusion_matrix(self, results: dict) -> None:
        cm = confusion_matrix(results["labels"], results["preds"])

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names or "auto",
            yticklabels=self.class_names or "auto",
        )

        plt.title("Confusion Matrix", fontsize=13)
        plt.xlabel("Predicted Labels", fontsize=11)
        plt.ylabel("True Labels", fontsize=11)

        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        plt.tight_layout()

        path = self.output_dir / "confusion_matrix.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved → {path}")


    def run_all(self, loader: DataLoader) -> None:
        """Run inference + generate all reports and plots."""
        print("Running inference on test set...")
        results = self.run(loader)
        self.report(results)
        self.plot_confusion_matrix(results)
        print(f"\nAll outputs saved to: {self.output_dir}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args() -> Namespace:
    parser = ArgumentParser(description="Inference ClusteringMoEModel on test set")
    parser.add_argument("--seed",                  type=int,   default=42)
    parser.add_argument("--num_experts",           type=int,   default=4)
    parser.add_argument("--top_k",                 type=int,   default=2)
    parser.add_argument("--temperature",           type=float, default=0.5)
    parser.add_argument("--pretrain_backbone",     action="store_true")
    parser.add_argument("--batch_size",            type=int,   default=64)
    parser.add_argument("--dataset_name",          type=str,   required=True)
    parser.add_argument("--type_model",            type=str,   required=True)
    parser.add_argument("--backbone_name",         type=str,   required=True)
    parser.add_argument("--checkpoint",            type=str,   default="best_checkpoint.pth", choices=["best_checkpoint.pth", "last_checkpoint.pth"])
    parser.add_argument("--runtime",               type=str,   help="e.g. run_20260531-101010")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    args     = parse_args()
    set_seed(args.seed)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = Path(__file__).parents[3]

    print("=" * 60)
    print(f"  seed        : {args.seed}")
    print(f"  num_experts : {args.num_experts}")
    print(f"  top_k       : {args.top_k}")
    print(f"  temperature : {args.temperature}")
    print(f"  checkpoint  : {args.checkpoint}")
    print(f"  device      : {device}")
    print("=" * 60)

    # ── Dataset ──────────────────────────────────────────────
    from datasets.plantdoc_dataset_moe import build_datasets

    _, _, test_dataset = build_datasets(use_context=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True,
    )
    class_names = (
        list(test_dataset.class_to_idx.keys())
        if hasattr(test_dataset, "class_to_idx") else None
    )

    # ── Checkpoint ────────────────────────────────────────────
    checkpoint_path = build_checkpoint_path(root_dir, args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n  {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint (epoch {checkpoint['epoch']}): {checkpoint_path.name}")

    # ── Model ─────────────────────────────────────────────────
    model = load_model(checkpoint, args)

    # ── Output dir ────────────────────────────────────────────
    output_dir = build_output_dir(root_dir, args)

    # ── Run inference ─────────────────────────────────────────
    inference = MoEInference(
        model       = model,
        device      = device,
        output_dir  = output_dir,
        class_names = class_names,
    )
    inference.run_all(test_loader)


if __name__ == "__main__":
    main()
