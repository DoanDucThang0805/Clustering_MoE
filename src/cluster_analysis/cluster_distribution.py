# src/cluster_analysis/cluster_distribution.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyze_cluster_distribution(
    cluster_file: Path,
    output_dir: Path,
) -> None:
    """
    Analyze sample distribution across clusters.

    This function:
        1. Loads a clustering result (.npz).
        2. Computes the number of samples in each cluster.
        3. Prints cluster statistics.
        4. Saves a bar-chart visualization.

    Parameters
    ----------
    cluster_file : Path
        Path to clustering result file.

    output_dir : Path
        Directory used to save analysis figures.
    """

    # --------------------------------------------------
    # Load clustering result
    # --------------------------------------------------

    data = np.load(cluster_file)

    cluster_assignments = data["cluster_assignments"]
    num_clusters = int(data["num_clusters"])

    # --------------------------------------------------
    # Count samples per cluster
    # --------------------------------------------------

    counts = np.bincount(
        cluster_assignments,
        minlength=num_clusters,
    )

    total_samples = len(cluster_assignments)

    # --------------------------------------------------
    # Print statistics
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print(cluster_file.name)
    print("=" * 60)

    for cluster_id, count in enumerate(counts):

        percentage = (
            count / total_samples
        ) * 100

        print(
            f"Cluster {cluster_id}: "
            f"{count:6d} samples "
            f"({percentage:6.2f}%)"
        )

    print(f"Total samples: {total_samples}")

    # --------------------------------------------------
    # Create output directory
    # --------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Plot cluster distribution
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.bar(
        np.arange(num_clusters),
        counts,
    )

    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Samples")

    plt.title(
        f"Cluster Distribution (G={num_clusters})"
    )

    plt.tight_layout()

    # --------------------------------------------------
    # Save figure
    # --------------------------------------------------

    save_path = (
        output_dir
        / f"distribution_G{num_clusters}.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {save_path}")


def main() -> None:
    """
    Run cluster distribution analysis for all
    clustering outputs generated in Checkpoint 3.

    Expected input structure
    ------------------------
    clusters/
        dataset_name/
            model_type/
                backbone_name/
                    model_name/
                        seed_42/
                            clusters_kmeans_G2_seed42.npz
                            clusters_kmeans_G3_seed42.npz
                            ...

    Generated outputs
    -----------------
    cluster_analysis/
        dataset_name/
            model_type/
                backbone_name/
                    model_name/
                        seed_42/
                            distribution_G2.png
                            distribution_G3.png
                            ...
    """

    # --------------------------------------------------
    # Experiment metadata
    # --------------------------------------------------

    dataset_name = "plantdoc"

    model_type = "non_pretrain_backbone"

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
    # Find all clustering result files
    # --------------------------------------------------

    cluster_files = sorted(
        cluster_dir.glob(
            "clusters_kmeans_G*_seed*.npz"
        )
    )

    if len(cluster_files) == 0:
        raise FileNotFoundError(
            f"No cluster files found in: "
            f"{cluster_dir}"
        )

    # --------------------------------------------------
    # Analyze each clustering result
    # --------------------------------------------------

    for cluster_file in cluster_files:

        analyze_cluster_distribution(
            cluster_file=cluster_file,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()