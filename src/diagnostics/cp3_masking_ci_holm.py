"""Table VIII: Expert masking, paired CI95% và Holm-adjusted p (n=10 seed).

Tính paired t-test (mask vs unmasked, cùng seed) cho Macro-F1 của 4 expert,
CI95% theo t-phân phối (df=n-1), Holm step-down trên 4 phép so sánh.
Tái dùng holm_bonferroni từ paired_checkpoint_test.py (đã verify công thức đúng).

Nguồn: paper_results/tables/expert_masking_ablation.csv (raw per-seed, CP3).

Chạy từ src/:
    python -m diagnostics.cp3_masking_ci_holm
"""
from pathlib import Path
import csv
import math
import statistics
from collections import defaultdict

from scipy.stats import t as tdist

from statistical_test.paired_checkpoint_test import holm_bonferroni

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper_results" / "tables" / "expert_masking_ablation.csv"
OUT = ROOT / "paper_results" / "tables" / "table8_expert_masking_ci_holm.csv"
ALPHA = 0.05


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


rows = list(csv.DictReader(open(SRC)))
by_mask = defaultdict(list)
for r in rows:
    by_mask[r["mask_expert"]].append(r)

n = len(by_mask["none"])
df = n - 1
t_crit = float(tdist.ppf(1 - ALPHA / 2, df))

base_macro_f1 = _mean([float(r["macro_f1"]) for r in by_mask["none"]])
base_std = _std([float(r["macro_f1"]) for r in by_mask["none"]])

masks = ["0", "1", "2", "3"]
deltas = {m: [float(r["delta_macro_f1"]) for r in by_mask[m]] for m in masks}
macro_f1s = {m: [float(r["macro_f1"]) for r in by_mask[m]] for m in masks}

raw_p = []
for m in masks:
    d = deltas[m]
    se = _std(d) / math.sqrt(n)
    tstat = _mean(d) / se
    p = 2 * (1 - tdist.cdf(abs(tstat), df))
    raw_p.append(p)

holm_p = holm_bonferroni(raw_p)

fields = ["masked_expert", "macro_f1_mean_std", "delta_macro_f1_mean_std",
          "ci95_low", "ci95_high", "holm_p"]
out_rows = [{
    "masked_expert": "None", "macro_f1_mean_std": f"{base_macro_f1:.4f} ± {base_std:.4f}",
    "delta_macro_f1_mean_std": "-", "ci95_low": "-", "ci95_high": "-", "holm_p": "-",
}]
for m, hp in zip(masks, holm_p):
    d = deltas[m]
    mf1 = macro_f1s[m]
    mean_d, std_d = _mean(d), _std(d)
    se = std_d / math.sqrt(n)
    out_rows.append({
        "masked_expert": f"Expert {m}",
        "macro_f1_mean_std": f"{_mean(mf1):.4f} ± {_std(mf1):.4f}",
        "delta_macro_f1_mean_std": f"{mean_d:.4f} ± {std_d:.4f}",
        "ci95_low": round(mean_d - t_crit * se, 4),
        "ci95_high": round(mean_d + t_crit * se, 4),
        "holm_p": round(hp, 4),
    })

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)
print(f"Saved {len(out_rows)} rows -> {OUT}")
