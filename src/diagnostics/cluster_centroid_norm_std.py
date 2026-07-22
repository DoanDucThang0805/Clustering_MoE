"""Bảng std(||mu_g||_2) giữa 4 cluster (G=4), theo seed.

mu_g = centroid K-means (cosine, fit trên feature vector u_i đã L2-normalize),
lấy từ clustering_results/.../kmeans/cosine/seed_*/clusters_kmeans_G4_seed*.npz
(key "centroids", shape (G, D)). Chỉ đọc lại centroid đã có sẵn, không tính lại
K-means hay đụng tới feature vector u_i gốc.

||mu_g||_2 = L2 norm chuẩn. std tính trên 4 giá trị (G=4) trong cùng 1 seed
(sample std).

Chạy từ src/:
    python -m diagnostics.cluster_centroid_norm_std
"""
from pathlib import Path
import statistics
import numpy as np
import csv

ROOT = Path(__file__).resolve().parents[2]
CENTROID_DIR = (ROOT / "clustering_results" / "plantdoc" / "pretrain_backbone"
                 / "mobilenetv3small_torchvision_backbone" / "kmeans" / "cosine")
OUT = ROOT / "paper_results" / "tables" / "cluster_centroid_norm_std.csv"
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
G = 4


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


SUB = {1: "₁", 2: "₂", 3: "₃", 4: "₄"}
MU_COL = [f"‖μ{SUB[g+1]}‖₂" for g in range(G)]
MEAN_COL = "mean(‖μ_g‖₂)"
STD_COL = "std(‖μ_g‖₂)"

fields = ["seed"] + MU_COL + [MEAN_COL, STD_COL]

rows = []
for seed in SEEDS:
    path = CENTROID_DIR / f"seed_{seed}" / f"clusters_kmeans_G{G}_seed{seed}.npz"
    centroids = np.load(path)["centroids"]          # (G, D)
    norms = np.linalg.norm(centroids, axis=1)        # ||mu_g||_2, g=1..G

    row = {"seed": seed}
    for g in range(G):
        row[MU_COL[g]] = round(float(norms[g]), 4)
    row[MEAN_COL] = round(float(norms.mean()), 4)
    row[STD_COL] = round(_std(list(norms)), 4)
    rows.append(row)

# Dòng tổng kết: mean +/- std của std(‖μ_g‖₂) qua 10 seed (1 con số đại diện chung).
std_values = [r[STD_COL] for r in rows]
summary_row = {"seed": "mean_across_seeds"}
for col in MU_COL:
    summary_row[col] = ""
summary_row[MEAN_COL] = ""
summary_row[STD_COL] = f"{_mean(std_values):.4f} +/- {_std(std_values):.4f}"
rows.append(summary_row)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"Saved {len(rows)} rows -> {OUT}")
