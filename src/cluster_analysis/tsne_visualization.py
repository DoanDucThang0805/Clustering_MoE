from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def visualize_tsne(
    feature_file: Path,
    cluster_file: Path,
    output_dir: Path,
) -> None:
    """
    Visualize feature embeddings using t-SNE.

    Generate:
        1. t-SNE colored by cluster assignment.
        2. t-SNE colored by ground-truth labels.

    Parameters
    ----------
    feature_file : Path
        features_train_seedXX.npz

    cluster_file : Path
        clusters_kmeans_GX_seedXX.npz

    output_dir : Path
        Directory for saving figures.
    """

    # --------------------------------------------------
    # Load features
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
    # t-SNE
    # --------------------------------------------------

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )

    embedding_2d = tsne.fit_transform(
        features
    )

    # ==================================================
    # Plot 1: Cluster assignment
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
        f"t-SNE - Cluster Assignment (G={num_clusters})"
    )

    plt.xlabel("tSNE-1")
    plt.ylabel("tSNE-2")

    plt.tight_layout()

    save_path = (
        output_dir
        / f"tsne_cluster_G{num_clusters}.png"
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
    # Plot 2: Ground truth labels
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
        f"t-SNE - Ground Truth Classes (G={num_clusters})"
    )

    plt.xlabel("tSNE-1")
    plt.ylabel("tSNE-2")

    plt.tight_layout()

    save_path = (
        output_dir
        / f"tsne_class_G{num_clusters}.png"
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

    if len(cluster_files) == 0:
        raise FileNotFoundError(
            f"No cluster files found in:\n{cluster_dir}"
        )

    for cluster_file in cluster_files:

        visualize_tsne(
            feature_file=feature_file,
            cluster_file=cluster_file,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
