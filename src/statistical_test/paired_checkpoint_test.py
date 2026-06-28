"""Paired statistical tests for the four selected PlantDoc configurations.

Cluster-MoE G4 cosine top-2 is model A. Accuracy and macro-F1 over seeds
42--46 are compared with MobileNetV3-Small, its context-aware MoE baseline,
and Cluster-MoE G4 euclidean top-2.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.clustering_moe.model import ClusteringMoEModel
from models.moe.model import MoEModel

SEEDS = (42, 43, 44, 45, 46)
BATCH_SIZE = 32
NUM_CLASSES = 8
NUM_EXPERTS = 4
TOP_K = 2
TEMPERATURE = 0.5
CONTEXT_DIM = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COSINE_NAME = "Cluster-MoE G4 Cosine Top-2"
MOBILENET_NAME = "MobileNetV3-Small"
MOE_NAME = "MobileNetV3-Small-MoE"
EUCLIDEAN_NAME = "Cluster-MoE G4 Euclidean Top-2"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: Literal["baseline", "moe", "cluster_moe"]
    checkpoint_root: Path
    metric: Literal["cosine", "euclidean"] | None = None

    @property
    def needs_context(self) -> bool:
        return self.kind == "moe"


MODEL_SPECS = {
    COSINE_NAME: ModelSpec(
        COSINE_NAME,
        "cluster_moe",
        Path(
            "/media/data/minhht/clustering_moe/checkpoints/plantdoc/"
            "clustering_moe/non_pretrain_backbone/"
            "mobilenetv3small_torchvision_backbone/kmeans/temperature_0.5/"
            "G4_cosine_top2"
        ),
        "cosine",
    ),
    MOBILENET_NAME: ModelSpec(
        MOBILENET_NAME,
        "baseline",
        Path(
            "/media/data/minhht/clustering_moe/checkpoints/plantdoc/"
            "non_pretrain_baseline/mobilenetv3small_torchvision"
        ),
    ),
    MOE_NAME: ModelSpec(
        MOE_NAME,
        "moe",
        Path(
            "/media/data/minhht/clustering_moe/checkpoints/plantdoc/"
            "moe_temperature_0.5/mobilenetv3small_torchvision_moe/"
            "4_experts/top_2"
        ),
    ),
    EUCLIDEAN_NAME: ModelSpec(
        EUCLIDEAN_NAME,
        "cluster_moe",
        Path(
            "/media/data/minhht/clustering_moe/checkpoints/plantdoc/"
            "clustering_moe/non_pretrain_backbone/"
            "mobilenetv3small_torchvision_backbone/kmeans/temperature_0.5/"
            "G4_euclidean_top2"
        ),
        "euclidean",
    ),
}

COMPARISONS = (
    (COSINE_NAME, MOBILENET_NAME),
    (COSINE_NAME, MOE_NAME),
    (COSINE_NAME, EUCLIDEAN_NAME),
)

CENTROID_ROOT = (
    REPO_ROOT
    / "clustering_results"
    / "plantdoc"
    / "non_pretrain_backbone"
    / "mobilenetv3small_torchvision_backbone"
    / "kmeans"
)


def find_checkpoint(root: Path, seed: int) -> Path:
    """Find best_checkpoint.pth in the latest run for one seed."""
    seed_dir = root / f"seed_{seed}"
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {seed_dir}")

    run_dirs = sorted(
        path
        for path in seed_dir.iterdir()
        if path.is_dir() and path.name.startswith("run_")
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directory found in: {seed_dir}")
    if len(run_dirs) > 1:
        print(
            f"  [WARN] {seed_dir} contains {len(run_dirs)} runs; "
            f"using latest: {run_dirs[-1].name}"
        )

    checkpoint_path = run_dirs[-1] / "best_checkpoint.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")
    return checkpoint_path


def centroid_path(metric: str, seed: int) -> Path:
    return (
        CENTROID_ROOT
        / metric
        / f"seed_{seed}"
        / f"clusters_kmeans_G{NUM_EXPERTS}_seed{seed}.npz"
    )


def load_centroids(metric: str, seed: int) -> torch.Tensor:
    path = centroid_path(metric, seed)
    if not path.is_file():
        raise FileNotFoundError(f"Missing centroid file: {path}")
    with np.load(path) as data:
        if "centroids" not in data:
            raise KeyError(f"Centroid archive has no 'centroids' array: {path}")
        centroids = np.asarray(data["centroids"], dtype=np.float32)
    if centroids.ndim != 2 or centroids.shape[0] != NUM_EXPERTS:
        raise ValueError(
            f"Invalid centroid shape {centroids.shape} in {path}; "
            f"expected ({NUM_EXPERTS}, embedding_dim)"
        )
    return torch.from_numpy(centroids)


def validate_required_files() -> None:
    """Fail before inference when any paired checkpoint or centroid is absent."""
    for spec in MODEL_SPECS.values():
        if not spec.checkpoint_root.is_dir():
            raise FileNotFoundError(
                f"Checkpoint root for {spec.name} does not exist: "
                f"{spec.checkpoint_root}"
            )
        for seed in SEEDS:
            find_checkpoint(spec.checkpoint_root, seed)
            if spec.kind == "cluster_moe":
                assert spec.metric is not None
                path = centroid_path(spec.metric, seed)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing centroid file: {path}")


def make_plain_test_loader() -> DataLoader:
    from datasets.plantdoc_dataset import test_dataset

    return DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=DEVICE.type == "cuda",
    )


def make_context_test_loader() -> DataLoader:
    from datasets.plantdoc_dataset_moe import build_datasets

    _, _, test_dataset = build_datasets(use_context=True)
    return DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=DEVICE.type == "cuda",
    )


def validate_dataset_alignment(
    plain_loader: DataLoader,
    context_loader: DataLoader,
) -> None:
    """Verify that both dataset modules expose the identical paired test set."""
    plain_dataset = plain_loader.dataset
    context_dataset = context_loader.dataset
    if len(plain_dataset) != len(context_dataset):
        raise ValueError(
            "PlantDoc test loaders have different lengths: "
            f"plain={len(plain_dataset)}, context={len(context_dataset)}"
        )

    plain_paths = [str(Path(path).resolve()) for path in plain_dataset.image_paths]
    context_paths = [str(Path(path).resolve()) for path in context_dataset.image_paths]
    if plain_paths != context_paths:
        raise ValueError(
            "plantdoc_dataset and plantdoc_dataset_moe have different test image order"
        )
    if list(plain_dataset.labels) != list(context_dataset.labels):
        raise ValueError(
            "plantdoc_dataset and plantdoc_dataset_moe have different test labels"
        )
    if plain_dataset.class_to_idx != context_dataset.class_to_idx:
        raise ValueError(
            "plantdoc_dataset and plantdoc_dataset_moe use different class mappings"
        )


def require_metadata(checkpoint: dict, expected: dict, path: Path) -> None:
    for key, expected_value in expected.items():
        if key not in checkpoint:
            raise KeyError(f"Checkpoint is missing metadata '{key}': {path}")
        actual_value = checkpoint[key]
        if isinstance(expected_value, float):
            matches = bool(np.isclose(float(actual_value), expected_value))
        else:
            matches = actual_value == expected_value
        if not matches:
            raise ValueError(
                f"Checkpoint metadata mismatch for '{key}' in {path}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )


def make_model(spec: ModelSpec, seed: int, checkpoint: dict, path: Path) -> nn.Module:
    if spec.kind == "baseline":
        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            NUM_CLASSES,
        )
        return model

    if spec.kind == "moe":
        require_metadata(
            checkpoint,
            {
                "context_dim": CONTEXT_DIM,
                "num_classes": NUM_CLASSES,
                "num_experts": NUM_EXPERTS,
                "top_k": TOP_K,
                "router_mode": "context_aware",
                "temperature": TEMPERATURE,
            },
            path,
        )
        return MoEModel(
            context_dim=CONTEXT_DIM,
            num_classes=NUM_CLASSES,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            router_mode="context_aware",
            backbone_name="mobilenetv3small_torchvision",
            pretrain_backbone=False,
            temperature=TEMPERATURE,
        )

    if spec.kind == "cluster_moe":
        assert spec.metric is not None
        require_metadata(
            checkpoint,
            {
                "num_classes": NUM_CLASSES,
                "num_experts": NUM_EXPERTS,
                "top_k": TOP_K,
                "metric": spec.metric,
                "temperature": TEMPERATURE,
            },
            path,
        )
        return ClusteringMoEModel(
            num_classes=NUM_CLASSES,
            centroids=load_centroids(spec.metric, seed),
            top_k=TOP_K,
            backbone_name="mobilenetv3small_torchvision",
            metric=spec.metric,
            pretrain_backbone=False,
            temperature=TEMPERATURE,
        )

    raise ValueError(f"Unsupported model kind: {spec.kind}")


def load_model(spec: ModelSpec, seed: int) -> nn.Module:
    path = find_checkpoint(spec.checkpoint_root, seed)
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Could not read checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must contain a dictionary: {path}")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'model_state_dict': {path}")

    try:
        model = make_model(spec, seed, checkpoint, path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except Exception as exc:
        raise RuntimeError(
            f"Checkpoint is incompatible with {spec.name} (seed {seed}): "
            f"{path}\n{exc}"
        ) from exc
    return model.to(DEVICE).eval()


@torch.inference_mode()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    needs_context: bool,
) -> tuple[float, float]:
    predictions: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    for batch in loader:
        if needs_context:
            if len(batch) != 3:
                raise ValueError("Context-aware MoE expects (images, labels, contexts)")
            images, labels, contexts = batch
            output = model(images.to(DEVICE), contexts.to(DEVICE))
        else:
            if len(batch) != 2:
                raise ValueError("Image-only model expects (images, labels)")
            images, labels = batch
            output = model(images.to(DEVICE))

        logits = output[0] if isinstance(output, (tuple, list)) else output
        predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
        labels_all.append(labels.numpy())

    if not predictions:
        raise ValueError("PlantDoc test loader is empty")
    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(labels_all)
    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


def collect_seed_metrics(spec: ModelSpec, loader: DataLoader) -> dict[str, np.ndarray]:
    accuracies: list[float] = []
    macro_f1_scores: list[float] = []
    for seed in SEEDS:
        checkpoint = find_checkpoint(spec.checkpoint_root, seed)
        print(f"    seed {seed}: {checkpoint.parent.name}")
        model = load_model(spec, seed)
        accuracy, macro_f1 = run_inference(model, loader, spec.needs_context)
        accuracies.append(accuracy)
        macro_f1_scores.append(macro_f1)
        print(f"      Accuracy={accuracy:.4f}  Macro-F1={macro_f1:.4f}")
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "accuracy": np.asarray(accuracies, dtype=np.float64),
        "macro_f1": np.asarray(macro_f1_scores, dtype=np.float64),
    }


def paired_tests(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Return mean(A-B), paired t-test p, and Wilcoxon p."""
    if a.shape != b.shape or a.shape != (len(SEEDS),):
        raise ValueError(
            f"Paired arrays must both have shape ({len(SEEDS)},); "
            f"got {a.shape} and {b.shape}"
        )
    delta = a - b
    mean_delta = float(delta.mean())

    # scipy returns NaN for constant differences. Define the limiting cases so
    # multiple-testing correction always receives finite p-values.
    if np.allclose(delta, 0.0):
        paired_t_p = 1.0
        wilcoxon_p = 1.0
    else:
        paired_t_p = (
            0.0
            if np.isclose(delta.std(ddof=1), 0.0)
            else float(ttest_rel(a, b).pvalue)
        )
        try:
            wilcoxon_p = float(wilcoxon(delta).pvalue)
        except ValueError:
            wilcoxon_p = float("nan")

    if not np.isfinite(paired_t_p):
        raise ValueError(f"Paired t-test produced a non-finite p-value: {delta}")
    return mean_delta, paired_t_p, wilcoxon_p


def holm_bonferroni(p_values: list[float]) -> list[float]:
    count = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * count
    previous = 0.0
    for rank, (original_index, p_value) in enumerate(indexed):
        current = min(1.0, max(previous, p_value * (count - rank)))
        adjusted[original_index] = current
        previous = current
    return adjusted


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    count = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * count
    previous = 1.0
    for index in range(count - 1, -1, -1):
        original_index, p_value = indexed[index]
        rank = index + 1
        current = min(1.0, previous, p_value * count / rank)
        adjusted[original_index] = current
        previous = current
    return adjusted


def make_conclusion(mean_delta: float, holm_p: float) -> str:
    if holm_p >= 0.05:
        return "not_significant"
    direction = "A>B" if mean_delta > 0 else "A<B"
    return f"significant ({direction}, Holm-adjusted)"


def build_statistical_rows(
    metrics_by_model: dict[str, dict[str, np.ndarray]],
) -> list[dict]:
    rows: list[dict] = []
    metric_labels = (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"))
    for model_a, model_b in COMPARISONS:
        for metric_key, metric_label in metric_labels:
            a = metrics_by_model[model_a][metric_key]
            b = metrics_by_model[model_b][metric_key]
            mean_delta, paired_t_p, wilcoxon_p = paired_tests(a, b)
            rows.append(
                {
                    "dataset": "plantdoc",
                    "metric": metric_label,
                    "model_A": model_a,
                    "model_B": model_b,
                    "n_seeds": len(SEEDS),
                    "mean_A": float(a.mean()),
                    "std_A": float(a.std(ddof=1)),
                    "mean_B": float(b.mean()),
                    "std_B": float(b.std(ddof=1)),
                    "mean_delta": mean_delta,
                    "paired_t_p": paired_t_p,
                    "wilcoxon_p": wilcoxon_p,
                }
            )

    if len(rows) != 6:
        raise AssertionError(f"Expected 6 statistical rows, got {len(rows)}")
    raw_p_values = [row["paired_t_p"] for row in rows]
    for row, holm_p, bh_p in zip(
        rows,
        holm_bonferroni(raw_p_values),
        benjamini_hochberg(raw_p_values),
    ):
        row["holm_p"] = holm_p
        row["bh_p"] = bh_p
        row["conclusion"] = make_conclusion(row["mean_delta"], holm_p)
    return rows


FIELDNAMES = (
    "dataset",
    "metric",
    "model_A",
    "model_B",
    "n_seeds",
    "mean_A",
    "std_A",
    "mean_B",
    "std_B",
    "mean_delta",
    "paired_t_p",
    "wilcoxon_p",
    "holm_p",
    "bh_p",
    "conclusion",
)


VARIABLE_DESCRIPTION_FIELDNAMES = (
    "Variable",
    "Symbol",
    "Description / Calculation",
)

VARIABLE_DESCRIPTIONS = (
    ("dataset", "–", "The benchmark dataset evaluated (PlantDoc in this analysis)."),
    ("metric", r"$m$", "The performance metric evaluated (Accuracy or Macro-F1)."),
    ("model_A", r"$A$", "The reference architecture: Cluster-MoE G4 cosine top-2."),
    (
        "model_B",
        r"$B$",
        "The baseline or alternative architecture compared against Model A.",
    ),
    (
        "n_seeds",
        r"$N$",
        "The number of paired random seeds (N = 5; seeds 42–46).",
    ),
    (
        "mean_A",
        r"$\bar{m}^{(A)}$",
        "The empirical mean of the metric for Model A across N seeds.",
    ),
    (
        "std_A",
        r"$s^{(A)}$",
        "The sample standard deviation for Model A (denominator N - 1).",
    ),
    (
        "mean_B",
        r"$\bar{m}^{(B)}$",
        "The empirical mean of the metric for Model B across N seeds.",
    ),
    (
        "std_B",
        r"$s^{(B)}$",
        "The sample standard deviation for Model B (denominator N - 1).",
    ),
    (
        "mean_delta",
        r"$\bar{\Delta}$",
        r"The average paired difference: $\bar{\Delta}=N^{-1}\sum_{i=1}^{N}(m_i^{(A)}-m_i^{(B)})$.",
    ),
    (
        "paired_t_p",
        r"$p_t$",
        r"The raw two-sided paired t-test p-value for $H_0: E[\Delta]=0$.",
    ),
    (
        "wilcoxon_p",
        r"$p_W$",
        "The raw two-sided Wilcoxon signed-rank p-value for paired differences.",
    ),
    (
        "holm_p",
        r"$p_{\mathrm{Holm}}$",
        "The Holm-Bonferroni-adjusted paired t-test p-value across all six tests.",
    ),
    (
        "bh_p",
        r"$p_{\mathrm{BH}}$",
        "The Benjamini-Hochberg-adjusted paired t-test p-value across all six tests.",
    ),
    (
        "conclusion",
        "–",
        "Significance at alpha = 0.05 based on holm_p; direction follows mean_delta.",
    ),
)

def write_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            output_row = {
                key: (
                    "nan"
                    if isinstance(row[key], float) and np.isnan(row[key])
                    else f"{row[key]:.6f}"
                    if isinstance(row[key], float)
                    else row[key]
                )
                for key in FIELDNAMES
            }
            writer.writerow(output_row)


def write_variable_description_csv(output_csv: Path) -> None:
    """Write a paper-ready data dictionary beside the statistical results."""
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(VARIABLE_DESCRIPTION_FIELDNAMES)
        writer.writerows(VARIABLE_DESCRIPTIONS)


def print_summary(
    rows: list[dict],
    output_csv: Path,
    variable_description_csv: Path,
) -> None:
    print("\n" + "=" * 120)
    print(
        f"{'Metric':<10} {'Model A':<34} {'Model B':<34} "
        f"{'Delta':>9} {'t-p':>9} {'W-p':>9} {'Holm-p':>9} {'BH-p':>9}  Conclusion"
    )
    print("-" * 120)
    for row in rows:
        wilcoxon_text = (
            "nan" if np.isnan(row["wilcoxon_p"]) else f"{row['wilcoxon_p']:.4f}"
        )
        print(
            f"{row['metric']:<10} {row['model_A']:<34} {row['model_B']:<34} "
            f"{row['mean_delta']:+9.4f} {row['paired_t_p']:9.4f} "
            f"{wilcoxon_text:>9} {row['holm_p']:9.4f} {row['bh_p']:9.4f}  "
            f"{row['conclusion']}"
        )
    print("=" * 120)
    print(f"\nResults saved to: {output_csv}")
    print(f"Variable descriptions saved to: {variable_description_csv}")


def main(output_csv: str | Path) -> list[dict]:
    output_path = Path(output_csv).expanduser().resolve()
    print(f"Device: {DEVICE}")
    print("Validating all checkpoints and centroid files...")
    validate_required_files()

    print("Building and validating PlantDoc test loaders...")
    plain_loader = make_plain_test_loader()
    context_loader = make_context_test_loader()
    validate_dataset_alignment(plain_loader, context_loader)
    print(f"PlantDoc test samples: {len(plain_loader.dataset)}")

    metrics_by_model: dict[str, dict[str, np.ndarray]] = {}
    for model_name, spec in MODEL_SPECS.items():
        print(f"\n[{model_name}]")
        loader = context_loader if spec.needs_context else plain_loader
        metrics_by_model[model_name] = collect_seed_metrics(spec, loader)

    rows = build_statistical_rows(metrics_by_model)
    write_csv(rows, output_path)
    variable_description_path = output_path.with_name(
        f"{output_path.stem}_variables.csv"
    )
    write_variable_description_csv(variable_description_path)
    print_summary(rows, output_path, variable_description_path)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "PlantDoc seed-wise paired tests with Cluster-MoE G4 cosine top-2 "
            "as model A."
        )
    )
    parser.add_argument(
        "--output_csv",
        default=str(REPO_ROOT / "reports" / "statistical_test" / "plantdoc" / "paired_statistics.csv"),
        help="Destination CSV path.",
    )
    arguments = parser.parse_args()
    main(arguments.output_csv)
