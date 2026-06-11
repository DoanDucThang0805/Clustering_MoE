import os
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Union, Literal, Any

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from models.clustering_moe.model import ClusteringMoEModel
from datasets.plantdoc_dataset import test_dataset

import logging
from datetime import datetime

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Console handler — INFO and above
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter(_LOG_FORMAT))

# File handler — WARNING and above, written to warnings/<timestamp>.log
_WARNINGS_DIR = Path(__file__).parents[2] / "warnings"
_WARNINGS_DIR.mkdir(parents=True, exist_ok=True)
_warning_log_path = _WARNINGS_DIR / f"warnings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_file_handler = logging.FileHandler(_warning_log_path, encoding="utf-8")
_file_handler.setLevel(logging.WARNING)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))

logging.basicConfig(level=logging.DEBUG, handlers=[_console_handler, _file_handler])
logger = logging.getLogger(__name__)
logger.info(f"Warning logs will be saved to: {_warning_log_path}")



class GetAccandmF1ScoreClusterMoE:
    """
    Evaluates Clustering MoE models by computing accuracy and macro-F1 scores on test dataset.
    
    Automatically discovers trained checkpoints from directory structure, loads models,
    and evaluates them. Supports aggregation by expert count, metric type, and top-k selection.
    
    Attributes:
        type_model (str): Model type/config (e.g., 'clustering_moe')
        dataset_name (str): Dataset name for checkpoint discovery (e.g., 'plantdoc')
        backbone_type (str): Backbone type ('pretrain_backbone' or 'non_pretrain_backbone')
        backbone_name (str): Backbone name (e.g., 'mobilenetv3small_torchvision')
        model_clustering_name (str): Clustering method name (e.g., 'kmeans')
        temperature (float): Temperature parameter for model
        csv_store_dir (Path): Directory to save CSV results
        export_to_csv (bool): Whether to export results to CSV
        csv_filename (str): Output CSV filename
    """
    
    def __init__(
        self, 
        dataset_name: str,
        type_model: str,
        backbone_type: Literal["pretrain_backbone", "non_pretrain_backbone"],
        backbone_name: Literal["mobilenetv3small_torchvision", "mobilenetv3small_timm"],
        model_clustering_name: str,
        temperature: float,
        csv_filename: str,
        csv_store_dir: Path = Path("./"), 
        export_to_csv: bool=False
    ) -> None:
        """
        Initialize the evaluator.
        
        Args:
            dataset_name: Dataset name for checkpoint discovery
            type_model: Model type/configuration variant
            backbone_type: Backbone type ('pretrain_backbone' or 'non_pretrain_backbone')
            backbone_name: Backbone name ('mobilenetv3small_torchvision' or 'mobilenetv3small_timm')
            model_clustering_name: Clustering method name (e.g., 'kmeans')
            temperature: Temperature parameter for model
            csv_filename: Output CSV filename
            csv_store_dir: Directory for CSV export (default: current directory)
            export_to_csv: Enable CSV export (default: False)
        """
        self.type_model = type_model
        self.dataset_name = dataset_name
        self.backbone_type = backbone_type
        self.backbone_name = backbone_name
        self.model_clustering_name = model_clustering_name
        self.temperature = temperature
        self.csv_store_dir = csv_store_dir
        self.export_to_csv = export_to_csv
        self.csv_filename = csv_filename

        logger.info(
            f"Initialized GetAccandmF1ScoreClusterMoE: "
            f"model_clustering_name={model_clustering_name}, backbone_name={backbone_name}, "
            f"type_model={type_model}, dataset={dataset_name}, temperature={temperature}, "
            f"export_csv={export_to_csv}"
        )


    def checkpoint_paths(self) -> List[Dict[str, Any]]:
        """
        Discover all checkpoint paths from directory structure.

        Searches:
            checkpoints/{dataset}/{type_model}/{backbone_type}/{backbone_name}_backbone/
            {model_clustering_name}/temperature_{temperature}/
            G{num_experts}_{metric}_top{k}/seed_{s}/run_{timestamp}/best_checkpoint.pth

        Returns:
            List of dicts with keys: 'num_experts', 'top_k', 'metric', 'seed', 'checkpoint_path'.

        Raises:
            FileNotFoundError: If the checkpoint root directory does not exist.
        """
        root_path = (
            Path(__file__).parents[2]
            / "checkpoints"
            / self.dataset_name
            / self.type_model
            / self.backbone_type
            / f"{self.backbone_name}_backbone"
            / self.model_clustering_name
            / f"temperature_{self.temperature}"
        )
        logger.info(f"Discovering checkpoints from: {root_path}")
        
        if not root_path.exists():
            logger.error(f"Checkpoint root directory not found: {root_path}")
            raise FileNotFoundError(f"Checkpoint root directory not found: {root_path}")
        
        list_paths = []
        metric_folders = sorted([d for d in os.listdir(root_path) if os.path.isdir(root_path / d)])
        logger.debug(f"Found {len(metric_folders)} metric configurations: {metric_folders}")
        
        for metric_folder in metric_folders:
            # Parse metric folder name: G{num_experts}_{metric}_top{k}
            # Example: G8_euclidean_top4
            try:
                parts = metric_folder.replace('G', '').split('_')
                num_experts = parts[0]
                metric = parts[1]
                top_k = parts[2].replace('top', '')
                logger.debug(f"  {metric_folder}: experts={num_experts}, metric={metric}, top_k={top_k}")
            except (IndexError, ValueError) as e:
                logger.warning(f"Failed to parse metric folder name: {metric_folder}. Error: {e}")
                continue
            
            metric_path = root_path / metric_folder
            seed_folders = sorted([d for d in os.listdir(metric_path) if d.startswith('seed_')])
            logger.debug(f"    Found {len(seed_folders)} seed variants")
            
            for seed_folder in seed_folders:
                try:
                    seed = seed_folder.split('_')[1]
                    seed_path = metric_path / seed_folder
                    
                    # Find all run_* directories
                    run_folders = sorted([d for d in os.listdir(seed_path) if d.startswith('run_')])
                    
                    if not run_folders:
                        logger.debug(f"    No run folders found for: {metric_folder}/{seed_folder}")
                        continue
                    
                    # Check if multiple checkpoints exist
                    if len(run_folders) > 1:
                        logger.warning(
                            f"Found {len(run_folders)} run folders for {metric_folder}/{seed_folder}. "
                            f"Run folders: {run_folders}. Selecting the latest: {run_folders[-1]}"
                        )
                        selected_run = run_folders[-1]
                    else:
                        selected_run = run_folders[0]
                    
                    # Get checkpoint path from selected run
                    run_path = seed_path / selected_run
                    checkpoint_file = run_path / "best_checkpoint.pth"
                    
                    if not checkpoint_file.exists():
                        logger.warning(f"Checkpoint file not found: {checkpoint_file}")
                        continue
                    
                    list_paths.append(
                        {
                            "num_experts": num_experts,
                            "top_k": top_k,
                            "metric": metric,
                            "seed": seed,
                            "checkpoint_path": checkpoint_file
                        }
                    )
                    logger.debug(
                        f"    Found checkpoint: experts={num_experts}, metric={metric}, top_k={top_k}, "
                        f"seed={seed} -> {selected_run}/best_checkpoint.pth"
                    )
                except Exception as e:
                    logger.warning(f"Error processing {metric_folder}/{seed_folder}: {e}")
                    continue
        
        logger.info(f"Discovered {len(list_paths)} checkpoints total")
        return list_paths
    

    def extract_checkpoint(self, checkpoint_path: Union[Path, str]) -> Dict[str, Any]:
        """
        Read model configuration metadata from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file (.pth).

        Returns:
            Dict containing: 'num_classes', 'num_experts', 'top_k', 'metric', 'temperature'.

        Raises:
            Exception: If the checkpoint file cannot be read.
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            num_classes = checkpoint["num_classes"]
            num_experts = checkpoint["num_experts"]
            top_k = checkpoint["top_k"]
            temperature = checkpoint["temperature"]
            metric = checkpoint["metric"]
            
            logger.debug(
                f"Extracted checkpoint config: experts={num_experts}, top_k={top_k}, "
                f"classes={num_classes}, metric={metric}, temp={temperature}"
            )
            
            return {
                "num_classes": num_classes,
                "num_experts": num_experts,
                "top_k": top_k,
                "metric": metric,
                "temperature": temperature,
            }
        except Exception as e:
            logger.error(f"Failed to extract checkpoint from {checkpoint_path}: {e}")
            raise
        

    def load_checkpoint(self, model: ClusteringMoEModel, checkpoint_path: Union[Path, str]) -> ClusteringMoEModel:
        """
        Load model weights from a checkpoint file into a model instance.

        Args:
            model: ClusteringMoEModel instance to load weights into.
            checkpoint_path: Path to the best_checkpoint.pth file.

        Returns:
            ClusteringMoEModel with loaded weights.

        Raises:
            Exception: If checkpoint loading fails.
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint["model_state_dict"]
            model.load_state_dict(state_dict=state_dict)
            logger.debug(f"Successfully loaded checkpoint from {checkpoint_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise
    
    
    def load_centroids(
        self,
        dataset_name:          str,
        backbone_type:         str,
        backbone_name:         str,
        model_clustering_name: str,
        metric:                Literal["cosine", "euclidean"],
        num_experts:           int,
        seed:                  int,
    ) -> torch.Tensor:
        """
        Load clustering centroids from a .npz file.

        File path pattern:
            clustering_results/{dataset}/{backbone_type}/{backbone_name}_backbone/
            {model_clustering_name}/{metric}/seed_{seed}/
            clusters_kmeans_G{num_experts}_seed{seed}.npz

        Args:
            dataset_name: Dataset name.
            backbone_type: Backbone type ('pretrain_backbone' or 'non_pretrain_backbone').
            backbone_name: Backbone name.
            model_clustering_name: Clustering method name (e.g., 'kmeans').
            metric: Distance metric ('cosine' or 'euclidean').
            num_experts: Number of clusters (experts).
            seed: Random seed used during clustering.

        Returns:
            Centroid tensor of shape (num_experts, D).

        Raises:
            FileNotFoundError: If the centroid file does not exist.
        """
        path = (
            Path(__file__).parents[2]
            / "clustering_results"
            / dataset_name
            / backbone_type
            / f"{backbone_name}_backbone"
            / model_clustering_name
            / metric
            / f"seed_{seed}"
            / f"clusters_kmeans_G{num_experts}_seed{seed}.npz"
        )
        if not path.exists():
            raise FileNotFoundError(f"Centroid file not found:\n  {path}")

        data      = np.load(path)
        centroids = data["centroids"]                               # (G, D)
        print(f"Loaded centroids: {centroids.shape}  from {path.name}")
        return torch.tensor(centroids, dtype=torch.float32)
    

    def create_model(
        self,
        num_classes: int,
        num_experts: int,
        top_k: int,
        temperature: float,
        dataset_name: str,
        backbone_type: str,
        backbone_name: Literal["mobilenetv3small_torchvision", "mobilenetv3small_timm"],
        metric: Literal["cosine", "euclidean"],
        model_clustering_name: str,
        seed: int,
        pretrain_backbone: bool
    ) -> ClusteringMoEModel:
        """
        Create ClusteringMoEModel instance with specified hyperparameters.
        
        Args:
            num_classes: Number of output classes
            num_experts: Number of experts in mixture
            top_k: Number of top experts to select
            temperature: Temperature for gating softmax
            dataset_name: Dataset name for loading centroids
            backbone_type: Backbone type ('pretrain_backbone' or 'non_pretrain_backbone')
            backbone_name: Backbone name ('mobilenetv3small_torchvision' or 'mobilenetv3small_timm')
            metric: Distance metric ('cosine' or 'euclidean')
            model_clustering_name: Clustering method name (e.g., 'kmeans')
            seed: Random seed for reproducibility
            pretrain_backbone: Whether to use pretrained backbone
            
        Returns:
            Initialized ClusteringMoEModel instance
        """
        logger.debug(
            f"Creating ClusteringMoE model: experts={num_experts}, top_k={top_k}, "
            f"classes={num_classes}, metric={metric}, temp={temperature}"
        )

        centroids = self.load_centroids(
            dataset_name=dataset_name,
            backbone_type=backbone_type,
            backbone_name=backbone_name,
            model_clustering_name=model_clustering_name,
            metric=metric,
            num_experts=num_experts,
            seed=seed
        )
        
        model = ClusteringMoEModel(
            num_classes=num_classes,
            centroids=centroids,
            top_k=top_k,
            backbone_name=backbone_name,
            metric=metric,
            pretrain_backbone=pretrain_backbone,
            temperature=temperature
        )
        return model


    def create_dataset(self) -> DataLoader:
        """
        Create a DataLoader for the test set.

        Returns:
            DataLoader with batch_size=32 and shuffle=False.
        """
        logger.info(f"Test dataset loaded: {len(test_dataset)} samples")
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return test_loader


    @torch.inference_mode(True)
    def run_inference(self, model: ClusteringMoEModel, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Run inference on the full test set and compute evaluation metrics.

        Args:
            model: ClusteringMoE model with loaded checkpoint weights.
            data_loader: DataLoader for the test set.

        Returns:
            Tuple of (accuracy, macro_f1) over the entire test set.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        logger.debug(f"Running inference on {device}...")
        
        all_labels = []
        all_predicts = []
        batch_count = 0
        
        for batch in data_loader:
            batch_count += 1
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            logits, _, _, _ = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_predicts.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_predicts = np.array(all_predicts)
        accuracy = accuracy_score(all_labels, all_predicts)
        macro_f1 = f1_score(all_labels, all_predicts, average="macro")
        
        logger.debug(f"Inference completed: {batch_count} batches processed")
        logger.debug(f"Results: Accuracy={accuracy:.4f}, Macro-F1={macro_f1:.4f}")
        
        return accuracy, macro_f1


    def acc_and_mf1_score(self) -> List[Dict[str, Any]]:
        """
        Evaluate all discovered checkpoints and compute metrics.
        
        Main evaluation pipeline that:
        1. Discovers all checkpoints
        2. Loads and evaluates each model
        3. Computes accuracy and macro-F1 for each
        4. Collects results with model configuration info
        
        Returns:
            List of dicts with keys: 'num_experts', 'top_k', 'metric', 'seed', 'accuracy', 'macro_f1'
        """
        logger.info("=" * 80)
        logger.info("Starting accuracy and Macro-F1 score calculation...")
        logger.info("=" * 80)
        
        results = []
        list_checkpoint_paths = self.checkpoint_paths()
        
        if not list_checkpoint_paths:
            logger.error("No checkpoints found!")
            return results
        
        test_loader = self.create_dataset()
        
        logger.info(f"Processing {len(list_checkpoint_paths)} checkpoints...")
        logger.info("-" * 80)

        for idx, checkpoint_info in enumerate(list_checkpoint_paths, 1):
            checkpoint_path = checkpoint_info['checkpoint_path']
            num_experts = checkpoint_info["num_experts"]
            top_k = checkpoint_info["top_k"]
            metric = checkpoint_info["metric"]
            seed = checkpoint_info["seed"]
            
            logger.info(f"[{idx}/{len(list_checkpoint_paths)}] experts={num_experts}, metric={metric}, top_k={top_k}, seed={seed}")
            
            try:
                checkpoint_config = self.extract_checkpoint(checkpoint_path)
                model = self.create_model(
                    num_classes=checkpoint_config["num_classes"],
                    num_experts=checkpoint_config["num_experts"],
                    top_k=checkpoint_config["top_k"],
                    temperature=checkpoint_config["temperature"],
                    dataset_name=self.dataset_name,
                    backbone_type=self.backbone_type,
                    backbone_name=self.backbone_name,
                    metric=checkpoint_config["metric"],
                    model_clustering_name=self.model_clustering_name,
                    seed=int(seed),
                    pretrain_backbone=(self.backbone_type == "pretrain_backbone")
                )
                model = self.load_checkpoint(model=model, checkpoint_path=checkpoint_path)
                accuracy, macro_f1 = self.run_inference(model=model, data_loader=test_loader)
                
                results.append(
                    {
                        "num_experts": num_experts,
                        "top_k": top_k,
                        "metric": metric,
                        "seed": seed,
                        "accuracy": accuracy,
                        "macro_f1": macro_f1
                    }
                )
                
                logger.info(f"  ✓ Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to process checkpoint: {e}")
                continue
        
        logger.info("-" * 80)
        logger.info(f"Completed processing. Results: {len(results)}/{len(list_checkpoint_paths)} successful")
        
        return results


    def export_to_df(self) -> pd.DataFrame:
        """
        Evaluate all checkpoints and return aggregated results as a DataFrame.

        Steps:
            1. Call acc_and_mf1_score() to evaluate all discovered checkpoints.
            2. Convert raw per-seed results into a DataFrame.
            3. Group by (num_experts, metric, top_k) and compute mean/std across seeds.
            4. Optionally export the aggregated DataFrame to CSV.

        Returns:
            DataFrame with columns: 'num_experts', 'metric', 'top_k',
            'accuracy_mean', 'accuracy_std', 'macro_f1_mean', 'macro_f1_std'.
            Returns an empty DataFrame if no results are available.
        """
        logger.info("Exporting results to DataFrame...")
        acc_and_macrof1 = self.acc_and_mf1_score()
        
        if not acc_and_macrof1:
            logger.warning("No results to export!")
            return pd.DataFrame()
        
        df = pd.DataFrame(acc_and_macrof1)
        logger.debug(f"Raw results DataFrame: {len(df)} rows")
        logger.debug(f"Columns: {list(df.columns)}")
        
        # Calculate both mean and std for accuracy and macro_f1
        agg_dict = {
            "accuracy": ["mean", "std"],
            "macro_f1": ["mean", "std"]
        }
        df = df.groupby(["num_experts", "metric", "top_k"])[["accuracy", "macro_f1"]].agg(agg_dict).reset_index()
        df.columns = ["num_experts", "metric", "top_k", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]
        
        logger.info(f"Aggregated results by (num_experts, metric, top_k): {len(df)} rows")
        logger.debug(f"\nAggregated Results:\n{df.to_string()}")
        
        if self.export_to_csv:
            csv_path = self.csv_store_dir / self.csv_filename
            df.to_csv(csv_path, index=False)
            logger.info(f"Results exported to CSV: {csv_path}")
        else:
            logger.info("CSV export disabled")
        
        return df


def main():
    """
    Command-line interface for model evaluation.
    
    Usage:
        python get_acc_mF1_cluster_moe.py --dataset_name plantdoc \\
            --type_model clustering_moe --backbone_type non_pretrain_backbone \\
            --backbone_name mobilenetv3small_torchvision --model_clustering_name kmeans \\
            --temperature 0.5 --export_to_csv --csv_filename results.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, default="plantdoc")
    parser.add_argument("--type_model", type=str, required=True, default="clustering_moe")
    parser.add_argument("--backbone_type", type=str, required=True, 
                       choices=["pretrain_backbone", "non_pretrain_backbone"])
    parser.add_argument("--backbone_name", type=str, required=True,
                       choices=["mobilenetv3small_torchvision", "mobilenetv3small_timm"])
    parser.add_argument("--model_clustering_name", type=str, required=True, default="kmeans")
    parser.add_argument("--temperature", type=float, required=True, default=0.5)
    parser.add_argument("--csv_store_dir", type=str, default="./results")
    parser.add_argument("--export_to_csv", action="store_true")
    parser.add_argument("--csv_filename", type=str, default="cluster_moe_results.csv")

    args = parser.parse_args()

    evaluator = GetAccandmF1ScoreClusterMoE(
        dataset_name=args.dataset_name,
        type_model=args.type_model,
        backbone_type=args.backbone_type,
        backbone_name=args.backbone_name,
        model_clustering_name=args.model_clustering_name,
        temperature=args.temperature,
        csv_filename=args.csv_filename,
        csv_store_dir=Path(args.csv_store_dir),
        export_to_csv=args.export_to_csv
    )

    df = evaluator.export_to_df()

    print(df)


if __name__ == "__main__":
    main()