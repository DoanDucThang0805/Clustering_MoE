# src/cluster_analysis/umap_visualization.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap


def visualize_umap(
    feature_file: Path,
    cluster_file: Path,
    output_dir: Path,
) -> None:
    """
    Visualize feature embeddings using UMAP.

    Two plots are generated:

    1. Colored by cluster assignment.
    2. Colored by ground-truth class.

    Parameters
    ----------
    feature_file : Path
        features_train_seedXX.npz

    cluster_file : Path
        clusters_kmeans_GX_seedXX.npz

    output_dir : Path
        Output directory.
    """

    # --------------------------------------------------
    # Load feature embeddings
    # --------------------------------------------------

    feature_data = np.load(
        feature_file
    )

    features = feature_data[
        "features"
    ]

    labels = feature_data[
        "labels"
    ]

    # --------------------------------------------------
    # Load clustering results
    # --------------------------------------------------

    cluster_data = np.load(
        cluster_file
    )

    assignments = cluster_data[
        "cluster_assignments"
    ]

    num_clusters = int(
        cluster_data["num_clusters"]
    )

    print("\n" + "=" * 80)
    print(cluster_file.name)
    print("=" * 80)

    print(
        f"Features shape : {features.shape}"
    )

    print(
        f"Clusters       : {num_clusters}"
    )

    # --------------------------------------------------
    # UMAP
    # --------------------------------------------------

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42,
    )

    embedding_2d = reducer.fit_transform(
        features
    )

    # ==================================================
    # Plot 1: Cluster Assignment
    # ==================================================

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=assignments,
        s=8,
    )

    plt.colorbar(
        scatter,
        label="Cluster ID",
    )

    plt.title(
        f"UMAP - Cluster Assignment (G={num_clusters})"
    )

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    plt.tight_layout()

    save_path = (
        output_dir
        / f"umap_cluster_G{num_clusters}.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {save_path}"
    )

    # ==================================================
    # Plot 2: Ground Truth Labels
    # ==================================================

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels,
        s=8,
    )

    plt.colorbar(
        scatter,
        label="Class ID",
    )

    plt.title(
        f"UMAP - Ground Truth Classes (G={num_clusters})"
    )

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    plt.tight_layout()

    save_path = (
        output_dir
        / f"umap_class_G{num_clusters}.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {save_path}"
    )


def main() -> None:

    dataset_name = "plantdoc"

    model_type = (
        "non_pretrain_models"
    )

    backbone_name = (
        "mobilenetv3small_torchvision_backbone"
    )

    model_name = "kmeans"

    seed = 42

    # --------------------------------------------------
    # Feature file
    # --------------------------------------------------

    feature_file = (
        Path(__file__).parents[2]
        / "feature_embeddings"
        / dataset_name
        / model_type
        / "mobilenetv3small_torchvision"
        / f"seed_{seed}"
        / f"features_train_seed{seed}.npz"
    )

    # --------------------------------------------------
    # Cluster directory
    # --------------------------------------------------

    cluster_dir = (
        Path(__file__).parents[2]
        / "clustering_results"
        / dataset_name
        / model_type
        / backbone_name
        / model_name
        / f"seed_{seed}"
    )

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    output_dir = (
        Path(__file__).parents[2]
        / "cluster_analysis"
        / dataset_name
        / model_type
        / backbone_name
        / model_name
        / f"seed_{seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cluster_files = sorted(
        cluster_dir.glob(
            "clusters_kmeans_G*_seed*.npz"
        )
    )

    for cluster_file in cluster_files:

        visualize_umap(
            feature_file=feature_file,
            cluster_file=cluster_file,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
    