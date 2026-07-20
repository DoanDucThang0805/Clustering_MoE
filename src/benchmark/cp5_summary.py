"""Xuất bảng tổng hợp CP5 cho bài báo từ cross_dataset_results.csv (per-seed).

Gộp theo (dataset, model): mean +/- sample std của accuracy/macro_f1/weighted_f1
trên các seed. params_m/flops_g/latency_ms không đổi theo seed nên giữ nguyên
giá trị (không phải mean/std).

Chạy từ src/:
    python -m benchmark.cp5_summary
"""
from pathlib import Path
import csv
import statistics
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper_results" / "tables" / "cross_dataset_results.csv"
OUT = ROOT / "paper_results" / "tables" / "cp5_cross_dataset_summary.csv"

MODEL_ORDER = ["dense", "learned_gate_moe", "cluster_moe"]
DATASET_ORDER = ["plantdoc", "plantvillage"]


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


rows = list(csv.DictReader(open(SRC)))
by_group = defaultdict(list)
for r in rows:
    by_group[(r["dataset"], r["model"])].append(r)

fields = [
    "dataset", "model", "routing",
    "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std",
    "weighted_f1_mean", "weighted_f1_std", "params_m", "flops_g", "latency_ms",
]

out_rows = []
for ds in DATASET_ORDER:
    for m in MODEL_ORDER:
        rs = by_group[(ds, m)]
        if not rs:
            continue
        row = {"dataset": ds, "model": m, "routing": rs[0]["routing"]}
        for metric in ["accuracy", "macro_f1", "weighted_f1"]:
            xs = [float(r[metric]) for r in rs]
            row[f"{metric}_mean"] = f"{_mean(xs):.4f}"
            row[f"{metric}_std"] = f"{_std(xs):.4f}"
        row["params_m"] = rs[0]["params_m"]
        row["flops_g"] = rs[0]["flops_g"]
        row["latency_ms"] = rs[0]["latency_ms"]
        out_rows.append(row)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)
print(f"Saved {len(out_rows)} rows -> {OUT}")
