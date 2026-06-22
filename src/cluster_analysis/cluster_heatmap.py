# src/cluster_analysis/cluster_heatmap.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_cluster_heatmap(
    cluster_file: Path,
    output_dir: Path,
) -> None:
    """
    Visualize cluster-to-class relationships.

    This function:
        1. Loads clustering results.
        2. Prints cluster-to-class statistics.
        3. Prints normalized cluster-to-class statistics.
        4. Creates an absolute-count heatmap.
        5. Creates a row-normalized heatmap.
        6. Saves both figures.

    Parameters
    ----------
    cluster_file : Path
        Path to clustering result (.npz).

    output_dir : Path
        Directory for saving heatmap figures.
    """

    # --------------------------------------------------
    # Load clustering result
    # --------------------------------------------------

    data = np.load(cluster_file)

    cluster_to_class_count = data[
        "cluster_to_class_count"
    ]

    num_clusters = int(
        data["num_clusters"]
    )

    num_classes = (
        cluster_to_class_count.shape[1]
    )

    print("\n" + "=" * 80)
    print(cluster_file.name)
    print("=" * 80)

    print(
        f"Clusters : {num_clusters}"
    )

    print(
        f"Classes  : {num_classes}"
    )

    # --------------------------------------------------
    # Display cluster-to-class matrix
    # --------------------------------------------------

    df = pd.DataFrame(
        cluster_to_class_count,
        index=[
            f"cluster_{i}"
            for i in range(num_clusters)
        ],
        columns=[
            f"class_{i}"
            for i in range(num_classes)
        ],
    )

    print(
        "\nCluster-to-Class Count Matrix"
    )

    print("-" * 80)

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        None,
    ):
        print(df)

    # --------------------------------------------------
    # Row normalization
    # --------------------------------------------------

    row_sum = (
        cluster_to_class_count.sum(
            axis=1,
            keepdims=True,
        )
    )

    normalized_heatmap = (
        cluster_to_class_count
        / np.maximum(row_sum, 1)
    )

    # --------------------------------------------------
    # Display normalized matrix
    # --------------------------------------------------

    df_normalized = pd.DataFrame(
        normalized_heatmap,
        index=[
            f"cluster_{i}"
            for i in range(num_clusters)
        ],
        columns=[
            f"class_{i}"
            for i in range(num_classes)
        ],
    )

    print(
        "\nNormalized Cluster-to-Class Matrix"
    )

    print("-" * 80)

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.float_format",
        "{:.3f}".format,
    ):
        print(df_normalized)

    # --------------------------------------------------
    # Create output directory
    # --------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================
    # Heatmap 1: Absolute Counts
    # ==================================================

    plt.figure(figsize=(14, 6))

    plt.imshow(
        cluster_to_class_count,
        aspect="auto",
    )

    plt.colorbar(
        label="Sample Count"
    )

    plt.xlabel("Class ID")
    plt.ylabel("Cluster ID")

    plt.title(
        f"Cluster-to-Class Count (G={num_clusters})"
    )

    plt.tight_layout()

    save_path = (
        output_dir
        / f"heatmap_count_G{num_clusters}.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nSaved: {save_path}"
    )

    # ==================================================
    # Heatmap 2: Normalized Counts
    # ==================================================

    plt.figure(figsize=(14, 6))

    plt.imshow(
        normalized_heatmap,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )

    plt.colorbar(
        label="Class Proportion"
    )

    plt.xlabel("Class ID")
    plt.ylabel("Cluster ID")

    plt.title(
        f"Normalized Cluster-to-Class Heatmap (G={num_clusters})"
    )

    plt.tight_layout()

    save_path = (
        output_dir
        / f"heatmap_normalized_G{num_clusters}.png"
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
    """
    Generate cluster heatmaps for all
    clustering outputs.

    Input
    -----
    clustering_results/
        dataset_name/
            model_type/
                backbone_name/
                    model_name/
                        seed_x/
                            clusters_kmeans_G*.npz

    Output
    ------
    cluster_analysis/
        dataset_name/
            model_type/
                backbone_name/
                    model_name/
                        seed_x/
                            heatmap_count_G*.png
                            heatmap_normalized_G*.png
    """

    # --------------------------------------------------
    # Experiment metadata
    # --------------------------------------------------

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
    # Input directory
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

    # --------------------------------------------------
    # Find cluster files
    # --------------------------------------------------

    cluster_files = sorted(
        cluster_dir.glob(
            "clusters_kmeans_G*_seed*.npz"
        )
    )

    if len(cluster_files) == 0:
        raise FileNotFoundError(
            f"No cluster files found in:\n"
            f"{cluster_dir}"
        )

    # --------------------------------------------------
    # Analyze all clustering results
    # --------------------------------------------------

    for cluster_file in cluster_files:

        analyze_cluster_heatmap(
            cluster_file=cluster_file,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
    