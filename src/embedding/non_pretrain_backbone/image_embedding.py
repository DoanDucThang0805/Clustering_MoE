import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.plantdoc_dataset import (
    extract_train_embedding_dataset,
    extract_validation_embedding_dataset,
    extract_test_embedding_dataset,
)
from models.non_pretrain_baseline.model_registry import MODEL_REGISTRY


class TorchvisionBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.pool     = model.avgpool
        self.flatten  = nn.Flatten(1)

    def forward(self, x):
        return self.flatten(self.pool(self.features(x)))


class TimmBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv_stem   = model.conv_stem
        self.bn1         = model.bn1
        self.blocks      = model.blocks
        self.global_pool = model.global_pool
        self.flatten     = nn.Flatten(1)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        return x


_BACKBONE_MAP = {
    "mobilenetv3small_torchvision": TorchvisionBackbone,
    "mobilenetv3small_timm":        TimmBackbone,
}

ROOT_CHECKPOINT_DIR = Path(__file__).parents[3] / "checkpoints"
ROOT_OUTPUT_DIR     = Path(__file__).parents[3] / "feature_embeddings"



class ImageEmbedding:
    def __init__(
        self,
        dataset_name: str,
        model_name:   str,
        type_model:   str,
        type_backbone: str,
        split:        Literal["train", "validation", "test"],
        seed:         int,
        run_time:     str,
        batch_size:   int = 64,
        num_workers:  int = 4,
    ):
        self.dataset_name = dataset_name
        self.model_name   = model_name
        self.type_model   = type_model
        self.type_backbone = type_backbone
        self.split        = split
        self.seed         = seed
        self.run_time     = run_time
        self.batch_size   = batch_size
        self.num_workers  = num_workers

        self.dataset = (
            extract_train_embedding_dataset      if split == "train"      else
            extract_validation_embedding_dataset if split == "validation" else
            extract_test_embedding_dataset
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.checkpoint_path = (
            ROOT_CHECKPOINT_DIR
            / self.dataset_name
            / self.type_model
            / self.model_name
            / f"seed_{self.seed}"
            / self.run_time
            / "best_checkpoint.pth"
        )
        self.output_dir = (
            ROOT_OUTPUT_DIR
            / self.dataset_name
            / self.type_backbone
            / f"{self.model_name}_backbone"
            / f"seed_{self.seed}"
        )


    def create_model(self, model_name: str) -> nn.Module:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model {model_name} not found in MODEL_REGISTRY.")
        return MODEL_REGISTRY[model_name]


    def load_checkpoint(self, model_name: str, checkpoint_path: Path) -> nn.Module:
        model      = self.create_model(model_name)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


    def create_backbone(self, model_name: str, checkpoint_path: Path) -> nn.Module:
        if model_name not in _BACKBONE_MAP:
            raise ValueError(f"No backbone wrapper for model: {model_name}")
        model    = self.load_checkpoint(model_name, checkpoint_path)
        backbone = _BACKBONE_MAP[model_name](model)
        return backbone


    def extract_embeddings(self) -> Path:
        # --- 1. Backbone ---
        backbone = self.create_backbone(self.model_name, self.checkpoint_path)
        backbone = backbone.to(self.device).eval()

        # --- 2. DataLoader ---
        data_loader = DataLoader(
            self.dataset,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = self.device.type == "cuda",
        )

        # --- 3. Extract ---
        all_features: list[np.ndarray] = []
        all_labels:   list[int]        = []

        with torch.no_grad():
            for images, labels in tqdm(
                data_loader,
                desc=f"Extracting [{self.split}] seed={self.seed}"
            ):
                features = backbone(images.to(self.device))
                all_features.append(features.cpu().numpy())
                all_labels.extend(labels.cpu().tolist())

        features_array = np.concatenate(all_features, axis=0)
        labels_array   = np.array(all_labels, dtype=np.int32)

        # --- 4. Sanity checks ---
        assert len(features_array) == len(self.dataset), (
            f"Feature count {len(features_array)} != dataset size {len(self.dataset)}"
        )
        assert features_array.ndim == 2, (
            f"Expected 2D array [N, d], got shape {features_array.shape}"
        )

        # --- 5. Save ---
        split_tag   = "val" if self.split == "validation" else self.split
        output_path = self.output_dir / f"features_{split_tag}_seed{self.seed}.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(output_path, features=features_array, labels=labels_array)
        print(f"Saved → {output_path}  shape={features_array.shape}")
        return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract feature embeddings from a trained backbone."
    )
    parser.add_argument("--dataset_name", type=str,  required=True)
    parser.add_argument("--model_name",   type=str,  default="mobilenetv3small_torchvision",
                        choices=list(_BACKBONE_MAP.keys()))
    parser.add_argument("--type_model",   type=str,  required=True)
    parser.add_argument("--type_backbone",   type=str,  required=True)
    parser.add_argument("--run_time",     type=str,  required=True,
                        help="Run timestamp folder name inside seed dir.")
    parser.add_argument("--split",        type=str,  default="all",
                        choices=["train", "validation", "test", "all"])
    parser.add_argument("--seed",         type=int,  default=42)
    parser.add_argument("--batch_size",   type=int,  default=64)
    parser.add_argument("--num_workers",  type=int,  default=4)
    return parser.parse_args()


def main():
    args   = parse_args()
    splits = ["train", "validation", "test"] if args.split == "all" else [args.split]

    for split in splits:
        ImageEmbedding(
            dataset_name = args.dataset_name,
            model_name   = args.model_name,
            type_model   = args.type_model,
            type_backbone= args.type_backbone,
            split        = split,
            seed         = args.seed,
            run_time     = args.run_time,
            batch_size   = args.batch_size,
            num_workers  = args.num_workers,
        ).extract_embeddings()


if __name__ == "__main__":
    print(ROOT_CHECKPOINT_DIR)
    main()
