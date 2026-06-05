import os
from pathlib import Path
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from datasets.plantdoc_dataset import test_dataset
from models.non_pretrain_baseline.model_registry import MODEL_REGISTRY


ROOT_CHECKPOINT_DIR = Path(__file__).parents[3] / "checkpoints"
ROOT_REPORT_DIR = Path(__file__).parents[3] / "reports"
BATCH_SIZE = 32
FIGURE_DPI = 300
FIGURE_SIZE = (10, 8)
FONT_SIZE = 15
HEATMAP_FMT = ".2f"
CM_FMT = "d"


class Inference:
    """Handles model inference and evaluation on test dataset.
    
    Attributes:
        device (str): Device to run inference on ('cuda' or 'cpu')
        checkpoint_path (Path): Path to the trained model checkpoint
        report_dir (Path): Directory to save evaluation reports
    """
    
    def __init__(
        self,
        dataset_name: str,
        type_model: str,
        model_name: str,
        seed: int,
        run_time: str,
    ) -> None:
        """Initialize inference pipeline.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'plantdoc')
            type_model: Type of model directory (e.g., 'non_pretrain_models')
            model_name: Name of the model to load
            seed: Random seed used in training
            run_time: Runtime identifier for the experiment
        """
        self.dataset_name = dataset_name
        self.type_model = type_model
        self.model_name = model_name
        self.seed = seed
        self.run_time = run_time
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.checkpoint_path = (
            ROOT_CHECKPOINT_DIR
            / self.dataset_name
            / self.type_model
            / self.model_name
            / f"seed_{self.seed}"
            / self.run_time
            / "best_checkpoint.pth"
        )
        self.report_dir = (
            ROOT_REPORT_DIR
            / self.dataset_name
            / self.type_model
            / self.model_name
            / f"seed_{self.seed}"
            / self.run_time
        )
        os.makedirs(self.report_dir, exist_ok=True)


    def load_checkpoint(self, checkpoint_path: Path) -> dict:
        """Load checkpoint from disk.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary containing checkpoint data
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        return checkpoint
    

    def create_model(self, model_name: str) -> torch.nn.Module:
        """Instantiate model from registry.
        
        Args:
            model_name: Name of model in registry
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model not found in registry
        """
        model = MODEL_REGISTRY.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found in registry")
        return model
    

    def load_model(self, checkpoint_path: Path, model_name: str) -> torch.nn.Module:
        """Load model with pretrained weights.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model_name: Name of model in registry
            
        Returns:
            Model in evaluation mode on appropriate device
        """
        checkpoint = self.load_checkpoint(checkpoint_path)
        model = self.create_model(model_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model
    

    def create_dataloader(self, dataset, batch_size: int = BATCH_SIZE) -> DataLoader:
        """Create dataloader for inference.
        
        Args:
            dataset: PyTorch dataset instance
            batch_size: Batch size for loading
            
        Returns:
            DataLoader configured for inference (no shuffling)
        """
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    

    def _get_target_names(self) -> list[str]:
        """Get class names from dataset.
        
        Returns:
            List of class names in order
        """
        return [
            test_dataset.idx_to_class[i]
            for i in range(len(test_dataset.idx_to_class))
        ]


    def run_inference(self) -> tuple[list, list]:
        """Run inference on test set.
        
        Returns:
            Tuple of (true_labels, predictions) as lists
        """
        model = self.load_model(self.checkpoint_path, self.model_name)
        test_loader = self.create_dataloader(test_dataset)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return all_labels, all_preds
    

    def print_classification_report(self, all_labels: list, all_preds: list) -> str:
        """Print classification metrics to console.
        
        Args:
            all_labels: Ground truth labels
            all_preds: Model predictions
            
        Returns:
            Classification report as string
        """
        target_names = self._get_target_names()
        report = classification_report(all_labels, all_preds, target_names=target_names)
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print("Classification Report:")
        print(report)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1 Score: {macro_f1:.4f}")
        
        return report
    

    def _plot_classification_heatmap(self, report_dict: dict, target_names: list[str]) -> None:
        """Plot and save classification metrics heatmap.
        
        Args:
            report_dict: Classification report as dictionary
            target_names: List of class names
        """
        os.makedirs(self.report_dir, exist_ok=True)
        df = pd.DataFrame(report_dict).transpose()
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        sns.heatmap(
            df.iloc[:-1, :-1],  # exclude accuracy row and support column
            annot=True,
            fmt=HEATMAP_FMT,
            cmap="Blues",
            cbar_kws={"label": "Score"},
            ax=ax,
        )
        ax.set_title("Classification Report (Precision / Recall / F1-score)", fontsize=FONT_SIZE)
        ax.set_xlabel("Evaluation Metrics")
        ax.set_ylabel("Disease Classes")
        plt.tight_layout()
        plt.savefig(
            self.report_dir / "classification_report_heatmap.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight"
        )
        plt.close(fig)


    def _plot_confusion_matrix(self, all_labels: list, all_preds: list, target_names: list[str]) -> None:
        """Plot and save confusion matrix.
        
        Args:
            all_labels: Ground truth labels
            all_preds: Model predictions
            target_names: List of class names
        """
        os.makedirs(self.report_dir, exist_ok=True)
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        sns.heatmap(
            cm,
            annot=True,
            fmt=CM_FMT,
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
            annot_kws={"size": FONT_SIZE},
            linewidths=0.5,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=FONT_SIZE, labelpad=10)
        ax.set_ylabel("True Label", fontsize=FONT_SIZE, labelpad=10)
        ax.tick_params(axis="x", labelsize=FONT_SIZE, rotation=45)
        ax.tick_params(axis="y", labelsize=FONT_SIZE, rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), ha="right", fontsize=FONT_SIZE)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT_SIZE)
        ax.collections[0].colorbar.ax.tick_params(labelsize=FONT_SIZE)
        plt.tight_layout()
        plt.savefig(
            self.report_dir / "confusion_matrix.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight"
        )
        plt.close(fig)


    def plot_report_and_confusion_matrix(self, all_labels: list, all_preds: list) -> None:
        """Generate and save classification report and confusion matrix visualizations.
        
        Args:
            all_labels: Ground truth labels
            all_preds: Model predictions
        """
        target_names = self._get_target_names()
        
        report_dict = classification_report(
            all_labels, all_preds,
            target_names=target_names,
            output_dict=True,
        )

        self._plot_classification_heatmap(report_dict, target_names)
        self._plot_confusion_matrix(all_labels, all_preds, target_names)
        
        print(f"✓ Saved reports to {self.report_dir}")

    

    def evaluate(self) -> None:
        """Run complete evaluation pipeline: inference -> metrics -> visualizations."""
        all_labels, all_preds = self.run_inference()
        self.print_classification_report(all_labels, all_preds)
        self.plot_report_and_confusion_matrix(all_labels, all_preds)
    

def main() -> None:
    """Main entry point for inference pipeline."""
    parser = ArgumentParser(description="Run model inference and generate evaluation reports")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="plantdoc",
        help="Name of the dataset (default: plantdoc)"
    )
    parser.add_argument(
        "--type_model",
        type=str,
        required=True,
        help="Type of model directory (e.g., non_pretrain_models)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to load"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed used in training"
    )
    parser.add_argument(
        "--run_time",
        type=str,
        required=True,
        help="Runtime identifier for the experiment"
    )
    args = parser.parse_args()

    inference = Inference(
        dataset_name=args.dataset_name,
        type_model=args.type_model,
        model_name=args.model_name,
        seed=args.seed,
        run_time=args.run_time,
    )
    inference.evaluate()


if __name__ == "__main__":
    main()