"""Xuất bảng tổng hợp CP3 cho bài báo từ expert_masking_ablation.csv (per-seed).

Gộp theo mask_expert (mean +/- sample std trên các seed) để dùng trực tiếp
làm bảng Results và thảo luận structured routing behavior.

Chạy từ src/:
    python -m diagnostics.cp3_expert_masking_summary
"""
from pathlib import Path
import csv
import statistics
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper_results" / "tables" / "expert_masking_ablation.csv"
OUT = ROOT / "paper_results" / "tables" / "cp3_expert_masking_summary.csv"


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


rows = list(csv.DictReader(open(SRC)))
by_mask = defaultdict(list)
for r in rows:
    by_mask[r["mask_expert"]].append(r)


def _fmt_mean_std(xs) -> str:
    return f"{_mean(xs):.4f} ± {_std(xs):.4f}"


# Cấu hình (model=cluster_moe, routing=cosine, G=4, top_k=2, tau=0.5, n_seeds=10)
# là hằng số trên mọi hàng -> không đưa vào cột, ghi ở caption bảng khi đưa vào bài.
fields = ["mask_expert", "macro_f1_mean_std", "delta_macro_f1_mean_std"]

out_rows = []
for mask in ["none", "0", "1", "2", "3"]:
    rs = by_mask[mask]
    row = {"mask_expert": mask}
    macro_f1 = [float(r["macro_f1"]) for r in rs]
    row["macro_f1_mean_std"] = _fmt_mean_std(macro_f1)

    if mask == "none":
        row["delta_macro_f1_mean_std"] = "—"
    else:
        delta_macro_f1 = [float(r["delta_macro_f1"]) for r in rs]
        row["delta_macro_f1_mean_std"] = _fmt_mean_std(delta_macro_f1)

    out_rows.append(row)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)
print(f"Saved {len(out_rows)} rows -> {OUT}")
