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
from scipy.stats import ttest_rel, wilcoxon, norm, t as tdist
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


# ═══════════════════════════════════════════════════════════════════════════
# CP4 — 10-seed pretrained: seedwise + extended paired stats + power analysis
# Tái dùng holm_bonferroni / benjamini_hochberg ở trên (không viết lại).
# Khác chế độ pilot: đọc per-seed acc/macro_f1/weighted_f1 từ CP1 CSV (KHÔNG
# inference lại), 10 seed pretrained, 3 model chính, + std_diff/CI95/d_z + power.
# ═══════════════════════════════════════════════════════════════════════════
import math

CP4_MODELS = ("dense", "learned_gate_moe", "cluster_moe")
CP4_ROUTING = {"dense": "none", "learned_gate_moe": "learned", "cluster_moe": "cosine"}
# params_m, flops_g (params_flops.csv) + latency_ms (edge Pi ONNX)
CP4_COMPLEXITY = {
    "dense":            dict(params_m=1.5261, flops_g=0.1229, latency_ms=6.2622),
    "learned_gate_moe": dict(params_m=3.4845, flops_g=0.1269, latency_ms=8.2498),
    "cluster_moe":      dict(params_m=3.4772, flops_g=0.1268, latency_ms=7.9588),
}
CP4_PAIRS = (("cluster_moe", "dense"), ("cluster_moe", "learned_gate_moe"),
             ("learned_gate_moe", "dense"))
CP4_METRICS = ("accuracy", "macro_f1")
CP4_ALPHA, CP4_POWER = 0.05, 0.80


def cp4_load_per_seed(cp1_csv: Path) -> dict[str, dict[int, dict[str, float]]]:
    """model -> {seed: {accuracy, macro_f1, weighted_f1}} cho init pretrained."""
    data: dict[str, dict[int, dict[str, float]]] = {m: {} for m in CP4_MODELS}
    with open(cp1_csv) as f:
        for r in csv.DictReader(f):
            if r["initialization"] != "imagenet_pretrained":
                continue
            data[r["model"]][int(r["seed"])] = {
                k: float(r[k]) for k in ("accuracy", "macro_f1", "weighted_f1")
            }
    return data


def cp4_write_seedwise(data, seeds, out_csv: Path) -> None:
    fields = ["seed", "dataset", "initialization", "backbone", "model", "routing",
              "G", "top_k", "tau", "accuracy", "macro_f1", "weighted_f1",
              "params_m", "flops_g", "latency_ms"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for mdl in CP4_MODELS:
            for s in seeds:
                d = data[mdl][s]
                w.writerow({
                    "seed": s, "dataset": "plantdoc",
                    "initialization": "imagenet_pretrained",
                    "backbone": "mobilenetv3small", "model": mdl,
                    "routing": CP4_ROUTING[mdl],
                    "G": 4 if mdl != "dense" else "",
                    "top_k": 2 if mdl != "dense" else "",
                    "tau": 0.5 if mdl != "dense" else "",
                    "accuracy": round(d["accuracy"], 4),
                    "macro_f1": round(d["macro_f1"], 4),
                    "weighted_f1": round(d["weighted_f1"], 4),
                    **CP4_COMPLEXITY[mdl],
                })


def cp4_run(cp1_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = cp4_load_per_seed(cp1_csv)
    seeds = sorted(data["cluster_moe"])
    n = len(seeds)
    if not all(len(data[m]) == n for m in CP4_MODELS):
        raise AssertionError("3 model phải cùng số seed")

    cp4_write_seedwise(data, seeds, out_dir / "seedwise_main_results.csv")

    t_crit = float(tdist.ppf(1 - CP4_ALPHA / 2, df=n - 1))
    z_beta = float(norm.ppf(CP4_POWER))

    ps_fields = ["metric", "model_a", "model_b", "n_seeds", "mean_a", "mean_b",
                 "mean_diff", "std_diff", "ci95_low", "ci95_high", "paired_t_p",
                 "wilcoxon_p", "holm_p", "bh_p", "effect_size_dz"]
    pw_fields = ["comparison", "metric", "alpha", "power_target", "pilot_mean_diff",
                 "pilot_std_diff", "effect_size_dz", "required_n_uncorrected",
                 "required_n_holm", "current_n", "decision"]
    ps_rows, pw_rows = [], []

    for metric in CP4_METRICS:
        recs = []
        for a, b in CP4_PAIRS:
            xa = np.array([data[a][s][metric] for s in seeds])
            xb = np.array([data[b][s][metric] for s in seeds])
            mean_delta, t_p, w_p = paired_tests_free(xa, xb)  # reuse-friendly helper
            std_diff = float((xa - xb).std(ddof=1))
            dz = mean_delta / std_diff if std_diff > 0 else float("inf")
            recs.append(dict(a=a, b=b, xa=xa, xb=xb, mean_diff=mean_delta,
                             std_diff=std_diff, t_p=t_p, w_p=w_p, dz=dz))
        # Holm/BH trong cùng metric family (m = số cặp = 3) — tái dùng helper pilot
        holm = holm_bonferroni([r["t_p"] for r in recs])
        bh = benjamini_hochberg([r["t_p"] for r in recs])

        for r, hp, bp in zip(recs, holm, bh):
            se = r["std_diff"] / math.sqrt(n)
            ps_rows.append({
                "metric": metric, "model_a": r["a"], "model_b": r["b"], "n_seeds": n,
                "mean_a": round(float(r["xa"].mean()), 4),
                "mean_b": round(float(r["xb"].mean()), 4),
                "mean_diff": round(r["mean_diff"], 4), "std_diff": round(r["std_diff"], 4),
                "ci95_low": round(r["mean_diff"] - t_crit * se, 4),
                "ci95_high": round(r["mean_diff"] + t_crit * se, 4),
                "paired_t_p": round(r["t_p"], 4), "wilcoxon_p": round(r["w_p"], 4),
                "holm_p": round(hp, 4), "bh_p": round(bp, 4),
                "effect_size_dz": round(r["dz"], 4),
            })
            dz_abs = abs(r["dz"])

            def req_n(alpha):
                z_a = float(norm.ppf(1 - alpha / 2))
                return int(math.ceil(((z_a + z_beta) / dz_abs) ** 2)) if dz_abs > 0 else 10 ** 9

            req_holm = req_n(CP4_ALPHA / len(CP4_PAIRS))
            decision = ("sufficient" if n >= req_holm
                        else "insufficient - report measured gain under protocol")
            pw_rows.append({
                "comparison": f"{r['a']}_vs_{r['b']}", "metric": metric,
                "alpha": CP4_ALPHA, "power_target": CP4_POWER,
                "pilot_mean_diff": round(r["mean_diff"], 4),
                "pilot_std_diff": round(r["std_diff"], 4),
                "effect_size_dz": round(r["dz"], 4),
                "required_n_uncorrected": req_n(CP4_ALPHA),
                "required_n_holm": req_holm, "current_n": n, "decision": decision,
            })

    for path, fields, rows in [
        (out_dir / "paired_statistics_extended.csv", ps_fields, ps_rows),
        (out_dir / "power_analysis.csv", pw_fields, pw_rows),
    ]:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    print(f"CP4  n_seeds={n}  seeds={seeds}")
    for r in ps_rows:
        sig = "SIG" if r["holm_p"] < CP4_ALPHA else "ns "
        print(f"  [{r['metric']:8s}] {r['model_a']:16s} - {r['model_b']:16s} "
              f"Δ={r['mean_diff']:+.4f} holm={r['holm_p']:.4f} [{sig}] dz={r['effect_size_dz']:+.3f}")
    for r in pw_rows:
        print(f"  power {r['comparison']:32s}[{r['metric']:8s}] req_n(holm)={r['required_n_holm']:>4} "
              f"cur={r['current_n']} -> {r['decision']}")
    print(f"Saved 3 CSV -> {out_dir}")


def paired_tests_free(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Như paired_tests nhưng không ràng buộc shape = len(SEEDS) (cho CP4 n=10)."""
    delta = a - b
    mean_delta = float(delta.mean())
    if np.allclose(delta, 0.0):
        return mean_delta, 1.0, 1.0
    t_p = 0.0 if np.isclose(delta.std(ddof=1), 0.0) else float(ttest_rel(a, b).pvalue)
    try:
        w_p = float(wilcoxon(a, b).pvalue)
    except ValueError:
        w_p = 1.0
    return mean_delta, t_p, w_p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PlantDoc paired tests. mode=pilot (5-seed non-pretrained, "
                    "inference) hoặc mode=cp4 (10-seed pretrained từ CP1 CSV + power)."
    )
    parser.add_argument("--mode", choices=["pilot", "cp4"], default="pilot")
    parser.add_argument(
        "--output_csv",
        default=str(REPO_ROOT / "reports" / "statistical_test" / "plantdoc" / "paired_statistics.csv"),
        help="[pilot] Destination CSV path.",
    )
    parser.add_argument(
        "--cp1_csv",
        default=str(REPO_ROOT / "paper_results" / "tables" / "pretrained_backbone_results.csv"),
        help="[cp4] Nguồn per-seed pretrained (CP1).",
    )
    parser.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "paper_results" / "tables"),
        help="[cp4] Thư mục xuất 3 CSV CP4.",
    )
    arguments = parser.parse_args()
    if arguments.mode == "cp4":
        cp4_run(Path(arguments.cp1_csv), Path(arguments.out_dir))
    else:
        main(arguments.output_csv)
