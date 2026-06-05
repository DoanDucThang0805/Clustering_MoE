import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from pathlib import Path
import argparse
import numpy as np
from sklearn.preprocessing import normalize
from models.clustering_models.kmean import KMeansClustering
np.random.seed(42)

def build_cluster_to_class_count(
    cluster_assignments: np.ndarray,
    labels: np.ndarray,
    num_clusters: int,
) -> np.ndarray:
    """
    Build cluster-to-class statistics matrix.

    Shape:
        (num_clusters, num_classes)
    """

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
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        centroids=centroids,
        cluster_assignments=cluster_assignments,
        cluster_to_class_count=cluster_to_class_count,
        method="kmeans",
        normalize="l2",
        num_clusters=num_clusters,
        seed=seed,
    )


def train_kmeans(
    features: np.ndarray,
    labels: np.ndarray,
    num_clusters: int,
    seed: int,
    output_dir: Path,
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
        help="Name of the dataset (default: plantdoc)",
    )
    parser.add_argument(
        "--backbone_type",
        type=str,
        help="Type of backbone: non_pretrain_backbone, pretrain_backbone (default: non_pretrain_backbone)",
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="mobilenetv3small_torchvision",
        help="Name of the backbone model (default: mobilenetv3small_torchvision)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 8],
        help="Number of clusters to train (default: 2 3 4 5 6 8)",
    )

    args = parser.parse_args()

    # ==========================================
    # Metadata
    # ==========================================

    dataset_name = args.dataset_name
    backbone_type = args.backbone_type
    backbone_name = args.backbone_name
    model_name = "kmeans"
    seed = args.seed
    cluster_list = args.num_clusters

    # ==========================================
    # Load train embeddings
    # ==========================================
    root_embedding_feature_dir = Path(__file__).parents[2] / "feature_embeddings"
    feature_file = (
        root_embedding_feature_dir
        / dataset_name
        / backbone_type
        / f"{backbone_name}_backbone"
        / f"seed_{seed}"
        / f"features_train_seed{seed}.npz"
    )

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    data = np.load(feature_file)

    features = data["features"]
    features = normalize(features, norm="l2", axis=1)
    labels = data["labels"]

    print(f"Features shape: {features.shape}")
    print(f"Labels shape:   {labels.shape}")

    # ==========================================
    # Output directory
    # ==========================================

    output_dir = (
        Path(__file__).parents[2] / "clustering_results"
        / dataset_name
        / backbone_type
        / f'{backbone_name}_backbone'
        / model_name
        / f"seed_{seed}"
    )

    # ==========================================
    # Run clustering
    # ==========================================

    for num_clusters in cluster_list:

        train_kmeans(
            features=features,
            labels=labels,
            num_clusters=num_clusters,
            seed=seed,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
