import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import argparse
import numpy as np
from sklearn.preprocessing import normalize

from models.clustering_models.kmean import KMeansClustering


def build_cluster_to_class_count(
    cluster_assignments: np.ndarray,
    labels: np.ndarray,
    num_clusters: int,
) -> np.ndarray:

    num_classes = len(np.unique(labels))

    cluster_to_class_count = np.zeros(
        (num_clusters, num_classes),
        dtype=np.int32,
    )

    for cluster_id, class_id in zip(
        cluster_assignments,
        labels,
    ):
        cluster_to_class_count[
            cluster_id,
            class_id,
        ] += 1

    return cluster_to_class_count


def save_cluster_file(
    output_path: Path,
    centroids: np.ndarray,
    cluster_assignments: np.ndarray,
    cluster_to_class_count: np.ndarray,
    num_clusters: int,
    seed: int,
    metric: str,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalize_method = (
        "l2"
        if metric == "cosine"
        else "none"
    )

    np.savez_compressed(
        output_path,
        centroids=centroids,
        cluster_assignments=cluster_assignments,
        cluster_to_class_count=cluster_to_class_count,
        method="kmeans",
        metric=metric,
        normalize=normalize_method,
        num_clusters=num_clusters,
        seed=seed,
    )


def train_kmeans(
    features: np.ndarray,
    labels: np.ndarray,
    num_clusters: int,
    seed: int,
    output_dir: Path,
    metric: str,
) -> None:

    print(f"\nTraining KMeans (G={num_clusters})")

    kmeans = KMeansClustering(
        n_clusters=num_clusters,
        random_state=seed,
    )

    cluster_assignments = kmeans.fit_predict(
        features
    )

    counts = np.bincount(
        cluster_assignments,
        minlength=num_clusters,
    )

    if np.any(counts == 0):
        raise RuntimeError(
            f"Empty cluster detected: {counts}"
        )

    cluster_to_class_count = (
        build_cluster_to_class_count(
            cluster_assignments=cluster_assignments,
            labels=labels,
            num_clusters=num_clusters,
        )
    )

    output_path = (
        output_dir
        / f"clusters_kmeans_G{num_clusters}_seed{seed}.npz"
    )

    save_cluster_file(
        output_path=output_path,
        centroids=kmeans.centroids,
        cluster_assignments=cluster_assignments,
        cluster_to_class_count=cluster_to_class_count,
        num_clusters=num_clusters,
        seed=seed,
        metric=metric,
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Cluster distribution: "
        f"{counts.tolist()}"
    )


def main():

    parser = argparse.ArgumentParser(
        description="Train KMeans clustering model on extracted features"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="plantdoc",
    )

    parser.add_argument(
        "--backbone_type",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--backbone_name",
        type=str,
        default="mobilenetv3small_torchvision",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num_clusters",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 8],
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=["euclidean", "cosine"],
        required=True,
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name
    backbone_type = args.backbone_type
    backbone_name = args.backbone_name
    model_name = "kmeans"
    seed = args.seed
    cluster_list = args.num_clusters
    metric = args.metric

    np.random.seed(seed)

    root_embedding_feature_dir = (
        Path(__file__).parents[2]
        / "feature_embeddings"
    )

    feature_file = (
        root_embedding_feature_dir
        / dataset_name
        / backbone_type
        / f"{backbone_name}_backbone"
        / f"seed_{seed}"
        / f"features_train_seed{seed}.npz"
    )

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_file}"
        )

    data = np.load(feature_file)

    features = data["features"]

    if metric == "cosine":
        features = normalize(
            features,
            norm="l2",
            axis=1,
        )

    labels = data["labels"]

    print(
        f"Features shape: {features.shape}"
    )

    print(
        f"Labels shape: {labels.shape}"
    )

    output_dir = (
        Path(__file__).parents[2]
        / "clustering_results"
        / dataset_name
        / backbone_type
        / f"{backbone_name}_backbone"
        / model_name
        / metric
        / f"seed_{seed}"
    )

    for num_clusters in cluster_list:

        train_kmeans(
            features=features,
            labels=labels,
            num_clusters=num_clusters,
            seed=seed,
            output_dir=output_dir,
            metric=metric,
        )


if __name__ == "__main__":
    main()