# src/cluster_analysis/umap_visualization.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import normalize
import umap


def visualize_umap(
    feature_file: Path,
    cluster_file: Path,
    output_dir: Path,
    distance_metric: str = "euclidean",
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

    # Normalize features if cosine metric is used
    if distance_metric == "cosine":
        features = normalize(
            features,
            norm="l2",
            axis=1,
        )

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
        metric=distance_metric,
        random_state=42,
    )

    embedding_2d = reducer.fit_transform(
        features
    )

    # ==================================================
    # Plot 1: Cluster Assignment
    # ==================================================

    sns.set_theme(style="ticks", context="paper", font_scale=1.5)

    plt.figure(figsize=(8, 6))

    unique_clusters = np.unique(assignments)
    assignments_str = [f"Cluster {i}" for i in assignments]
    cluster_order = [f"Cluster {i}" for i in sorted(unique_clusters)]
    
    colors_cluster = sns.color_palette("tab10", len(unique_clusters))
    palette_cluster = dict(zip(cluster_order, colors_cluster))

    sns.scatterplot(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        hue=assignments_str,
        hue_order=cluster_order,
        palette=palette_cluster,
        s=20,
        alpha=0.8,
        linewidth=0,
    )

    plt.title(
        f"UMAP Projection by Cluster Assignment (K={num_clusters})",
        fontweight="bold",
        pad=15
    )

    plt.xlabel("UMAP Dimension 1", fontweight="bold")
    plt.ylabel("UMAP Dimension 2", fontweight="bold")
    
    plt.legend(
        title="Cluster ID", 
        bbox_to_anchor=(1.02, 1), 
        loc="upper left",
        frameon=False
    )
    
    sns.despine()
    plt.tight_layout()

    save_path = (
        output_dir
        / f"umap_cluster_G{num_clusters}.png"
    )

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {save_path}"
    )

    # ==================================================
    # Plot 2: Ground Truth Labels
    # ==================================================

    plt.figure(figsize=(9, 6))

    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    labels_str = [f"Class {i}" for i in labels]
    class_order = [f"Class {i}" for i in sorted(unique_classes)]
    
    if num_classes <= 10:
        colors_class = sns.color_palette("tab10", num_classes)
    elif num_classes <= 20:
        colors_class = sns.color_palette("tab20", num_classes)
    else:
        colors_class = sns.color_palette("husl", num_classes)
        
    palette_class = dict(zip(class_order, colors_class))

    sns.scatterplot(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        hue=labels_str,
        hue_order=class_order,
        palette=palette_class,
        s=20,
        alpha=0.8,
        linewidth=0,
    )

    plt.title(
        f"UMAP Projection by Ground Truth Classes",
        fontweight="bold",
        pad=15
    )

    plt.xlabel("UMAP Dimension 1", fontweight="bold")
    plt.ylabel("UMAP Dimension 2", fontweight="bold")
    
    plt.legend(
        title="Class ID", 
        bbox_to_anchor=(1.02, 1), 
        loc="upper left",
        ncol=2 if num_classes > 15 else 1,
        fontsize="x-small",
        title_fontsize="small",
        frameon=False
    )
    
    sns.despine()
    plt.tight_layout()

    save_path = (
        output_dir
        / f"umap_class_G{num_clusters}.png"
    )

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {save_path}"
    )


def main() -> None:

    dataset_name = "plantdoc"

    model_type = (
        "non_pretrain_backbone"
    )

    backbone_name = (
        "mobilenetv3small_torchvision_backbone"
    )

    model_name = "kmeans"
    distance_metric = "cosine"

    seed = 42

    # --------------------------------------------------
    # Feature file
    # --------------------------------------------------

    feature_file = (
        Path(__file__).parents[2]
        / "feature_embeddings"
        / dataset_name
        / model_type
        / backbone_name
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
        / distance_metric
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
        / distance_metric
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
            distance_metric=distance_metric,
        )


if __name__ == "__main__":
    main()
    