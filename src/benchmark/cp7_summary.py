"""Xuất bảng tổng hợp CP7 cho bài báo từ soft_moe_baseline.csv (per-seed).

Gộp theo model: mean +/- sample std của accuracy/macro_f1/weighted_f1 trên
các seed. params_m/flops_g không đổi theo seed nên giữ nguyên giá trị.

Chạy từ src/:
    python -m benchmark.cp7_summary
"""
from pathlib import Path
import csv
import statistics
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper_results" / "tables" / "soft_moe_baseline.csv"
OUT = ROOT / "paper_results" / "tables" / "cp7_soft_moe_summary.csv"

MODEL_ORDER = ["soft_moe", "learned_gate_moe", "cluster_moe"]


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


rows = list(csv.DictReader(open(SRC)))
by_model = defaultdict(list)
for r in rows:
    by_model[r["model"]].append(r)

fields = [
    "model", "routing_type",
    "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std",
    "weighted_f1_mean", "weighted_f1_std", "params_m", "params_total_m", "flops_g",
]

out_rows = []
for m in MODEL_ORDER:
    rs = by_model[m]
    if not rs:
        continue
    row = {"model": m, "routing_type": rs[0]["routing_type"]}
    for metric in ["accuracy", "macro_f1", "weighted_f1"]:
        xs = [float(r[metric]) for r in rs]
        row[f"{metric}_mean"] = f"{_mean(xs):.4f}"
        row[f"{metric}_std"] = f"{_std(xs):.4f}"
    row["params_m"] = rs[0]["params_m"]
    row["params_total_m"] = rs[0]["params_total_m"]
    row["flops_g"] = rs[0]["flops_g"]
    out_rows.append(row)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)
print(f"Saved {len(out_rows)} rows -> {OUT}")
