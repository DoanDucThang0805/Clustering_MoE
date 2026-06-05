import argparse
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from models.pretrain_baseline.model_registry import MODEL_REGISTRY
from datasets.plantdoc_dataset import test_dataset


CHECKPOINT_ROOT_DIR = Path(__file__).parents[2] / "checkpoints"
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class GetAccvsMacroF1Pretrain:
    def __init__(
        self,
        dataset_name: str,
        type_model: str,
        model_name: str,
        csv_store_dir: Path,
        csv_filename: str,
        export_csv: bool = False
    ) -> None:
        self.checkpoint_dirs = CHECKPOINT_ROOT_DIR / dataset_name / type_model / model_name
        self.csv_store_dir = Path(csv_store_dir)
        self.csv_filename = csv_filename
        self.export_csv = export_csv
        self.model_name = model_name
        self.type_model = type_model
        self.dataset_name = dataset_name


    def get_checkpoint_paths(self) -> List[Dict[str, Any]]:
        """
        Find one best checkpoint for each seed directory.

        Expected structure:
            checkpoint_dirs/seed_42/run_YYYYmmdd-HHMMSS/best_checkpoint.pth

        If a seed has multiple runs, the newest checkpoint is selected.
        """
        if not self.checkpoint_dirs.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dirs}")

        checkpoint_infos = []
        seed_dirs = sorted(
            path for path in self.checkpoint_dirs.iterdir()
            if path.is_dir() and path.name.startswith("seed_")
        )

        for seed_dir in seed_dirs:
            checkpoints = sorted(
                seed_dir.rglob("best_checkpoint.pth"),
                key=lambda path: (path.stat().st_mtime, str(path)),
                reverse=True
            )

            if not checkpoints:
                logger.warning("No best_checkpoint.pth found under %s", seed_dir)
                continue

            checkpoint_infos.append(
                {
                    "seed": seed_dir.name.replace("seed_", ""),
                    "checkpoint_path": checkpoints[0]
                }
            )

        logger.info("Found %d checkpoint(s)", len(checkpoint_infos))
        return checkpoint_infos

    
    def load_checkpoint(self, model: nn.Module, checkpoint_path: Path) -> nn.Module:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        return model


    def create_model(self, model_name: str) -> nn.Module:
        model = MODEL_REGISTRY.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found in registry")
        return model

    
    def create_dataloader(self) -> DataLoader:
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return test_loader
    

    @torch.inference_mode()
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        all_labels = []
        all_predictions = []

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        macro_f1 = f1_score(all_labels, all_predictions, average="macro")
        return accuracy, macro_f1


    def acc_and_mf1_score(self) -> List[Dict[str, Any]]:
        results = []
        checkpoint_infos = self.get_checkpoint_paths()

        if not checkpoint_infos:
            logger.warning("No checkpoints found to evaluate")
            return results

        test_loader = self.create_dataloader()

        for idx, checkpoint_info in enumerate(checkpoint_infos, start=1):
            seed = checkpoint_info["seed"]
            checkpoint_path = checkpoint_info["checkpoint_path"]
            logger.info("[%d/%d] Evaluating seed=%s: %s", idx, len(checkpoint_infos), seed, checkpoint_path)

            try:
                model_instance = self.create_model(self.model_name)
                model_instance = self.load_checkpoint(model_instance, checkpoint_path)
                accuracy, macro_f1 = self.evaluate_model(model_instance, test_loader)
            except RuntimeError as exc:
                logger.error(
                    "Skip seed=%s because checkpoint cannot be loaded into torchvision model: %s",
                    seed,
                    str(exc).splitlines()[0]
                )
                continue

            results.append(
                {
                    "seed": seed,
                    "checkpoint_path": str(checkpoint_path),
                    "accuracy": accuracy,
                    "macro_f1": macro_f1
                }
            )
            logger.info("seed=%s accuracy=%.4f macro_f1=%.4f", seed, accuracy, macro_f1)

        return results


    def export_results_to_csv(self, results: pd.DataFrame) -> None:
        self.csv_store_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.csv_store_dir / self.csv_filename
        results.to_csv(csv_path, index=False)
        logger.info("Results exported to CSV: %s", csv_path)


    def export_to_df(self) -> pd.DataFrame:
        per_seed_results = self.acc_and_mf1_score()

        if not per_seed_results:
            return pd.DataFrame()

        df = pd.DataFrame(per_seed_results)
        aggregated_df = pd.DataFrame(
            [
                {
                    "num_seeds": len(df),
                    "accuracy_mean": df["accuracy"].mean(),
                    "accuracy_std": df["accuracy"].std(),
                    "macro_f1_mean": df["macro_f1"].mean(),
                    "macro_f1_std": df["macro_f1"].std(),
                }
            ]
        )

        if self.export_csv:
            self.export_results_to_csv(aggregated_df)

        return aggregated_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="plantdoc", help="Name of the dataset (e.g., plantdoc)")
    parser.add_argument("--type_model", type=str, required=True, help="Type of model (e.g., mobilenetv3_smallv1, mobilenetv3_smallv2, resnet18v1, resnet18v2)")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--csv_store_dir", type=str, default="./results", help="Directory to store the CSV results")
    parser.add_argument("--csv_filename", type=str, default="custom_mobilenetv3_smallv1_results.csv", help="Name of the CSV file to store results")
    parser.add_argument("--export_csv", action="store_true", help="Export results to CSV")
    args = parser.parse_args()

    evaluator = GetAccvsMacroF1Pretrain(
        dataset_name=args.dataset_name,
        type_model=args.type_model,
        model_name=args.model_name,
        csv_store_dir=Path(args.csv_store_dir),
        csv_filename=args.csv_filename,
        export_csv=args.export_csv,
    )
    df = evaluator.export_to_df()
    print(df)


if __name__ == "__main__":
    main()