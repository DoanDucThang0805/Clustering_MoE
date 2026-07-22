"""Tái tạo 3 bảng RAW của CP4 (seedwise_main_results.csv, paired_statistics_extended.csv,
power_analysis.csv) — bằng chứng thô cho reviewer, khớp với cp4_paired_tests_summary.csv
hiện hành (dense = retrain2 khoá, learned_gate_moe/cluster_moe = checkpoint gốc khoá).

Tái dùng eval_dense/eval_moe/eval_cluster từ cp1_direct_inference.py (đã verify khớp
cp1_backbone_init_summary.csv) và paired_tests_free/holm_bonferroni/benjamini_hochberg
từ paired_checkpoint_test.py (đã verify công thức đúng).

Chạy từ src/:
    python -m benchmark.cp4_raw_seedwise
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm, t as tdist

from benchmark.cp1_direct_inference import EVALUATORS, SEEDS, PARAMS_M, FLOPS_G
from statistical_test.paired_checkpoint_test import (
    holm_bonferroni, benjamini_hochberg, paired_tests_free,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_results" / "tables"

MODELS = ("dense", "learned_gate_moe", "cluster_moe")
ROUTING = {"dense": "none", "learned_gate_moe": "learned", "cluster_moe": "cosine"}
PAIRS = (("cluster_moe", "dense"), ("cluster_moe", "learned_gate_moe"),
         ("learned_gate_moe", "dense"))
METRICS = ("accuracy", "macro_f1")
ALPHA, POWER = 0.05, 0.80


def main() -> None:
    data: dict[str, dict[int, dict[str, float]]] = {m: {} for m in MODELS}
    for model in MODELS:
        print(f"\n[{model}]")
        for seed in SEEDS:
            m = EVALUATORS[model](seed)
            data[model][seed] = m
            print(f"  seed={seed}: acc={m['accuracy']:.4f} mF1={m['macro_f1']:.4f}")

    # ── seedwise_main_results.csv ──
    seedwise_fields = ["seed", "dataset", "initialization", "backbone", "model", "routing",
                        "G", "top_k", "tau", "accuracy", "macro_f1", "weighted_f1",
                        "params_m", "flops_g", "latency_ms"]
    seedwise_rows = []
    for model in MODELS:
        for seed in SEEDS:
            d = data[model][seed]
            seedwise_rows.append({
                "seed": seed, "dataset": "plantdoc", "initialization": "imagenet_pretrained",
                "backbone": "mobilenetv3small", "model": model, "routing": ROUTING[model],
                "G": 4 if model != "dense" else "", "top_k": 2 if model != "dense" else "",
                "tau": 0.5 if model != "dense" else "",
                "accuracy": round(d["accuracy"], 4), "macro_f1": round(d["macro_f1"], 4),
                "weighted_f1": round(d["weighted_f1"], 4),
                "params_m": PARAMS_M[model], "flops_g": FLOPS_G[model], "latency_ms": "",
            })
    with open(OUT_DIR / "seedwise_main_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=seedwise_fields)
        w.writeheader()
        w.writerows(seedwise_rows)
    print(f"\nSaved {len(seedwise_rows)} rows -> {OUT_DIR / 'seedwise_main_results.csv'}")

    # ── paired stats (Holm/BH trên toàn bộ 6 test) ──
    n = len(SEEDS)
    t_crit = float(tdist.ppf(1 - ALPHA / 2, df=n - 1))
    z_beta = float(norm.ppf(POWER))

    recs = []
    for metric in METRICS:
        for a, b in PAIRS:
            xa = np.array([data[a][s][metric] for s in SEEDS])
            xb = np.array([data[b][s][metric] for s in SEEDS])
            mean_delta, t_p, w_p = paired_tests_free(xa, xb)
            std_diff = float((xa - xb).std(ddof=1))
            dz = mean_delta / std_diff if std_diff > 0 else float("inf")
            recs.append(dict(metric=metric, a=a, b=b, xa=xa, xb=xb, mean_diff=mean_delta,
                              std_diff=std_diff, t_p=t_p, w_p=w_p, dz=dz))
    holm = holm_bonferroni([r["t_p"] for r in recs])
    bh = benjamini_hochberg([r["t_p"] for r in recs])

    ps_fields = ["metric", "model_a", "model_b", "n_seeds", "mean_a", "mean_b",
                 "mean_diff", "std_diff", "ci95_low", "ci95_high", "paired_t_p",
                 "wilcoxon_p", "holm_p", "bh_p", "effect_size_dz"]
    pw_fields = ["comparison", "metric", "alpha", "power_target", "pilot_mean_diff",
                 "pilot_std_diff", "effect_size_dz", "required_n_uncorrected",
                 "required_n_holm", "current_n", "decision"]
    ps_rows, pw_rows = [], []
    for r, hp, bp in zip(recs, holm, bh):
        se = r["std_diff"] / math.sqrt(n)
        ps_rows.append({
            "metric": r["metric"], "model_a": r["a"], "model_b": r["b"], "n_seeds": n,
            "mean_a": round(float(r["xa"].mean()), 4), "mean_b": round(float(r["xb"].mean()), 4),
            "mean_diff": round(r["mean_diff"], 4), "std_diff": round(r["std_diff"], 4),
            "ci95_low": round(r["mean_diff"] - t_crit * se, 4),
            "ci95_high": round(r["mean_diff"] + t_crit * se, 4),
            "paired_t_p": round(r["t_p"], 4), "wilcoxon_p": round(r["w_p"], 4),
            "holm_p": round(hp, 4), "bh_p": round(bp, 4), "effect_size_dz": round(r["dz"], 4),
        })
        dz_abs = abs(r["dz"])

        def req_n(alpha):
            z_a = float(norm.ppf(1 - alpha / 2))
            return int(math.ceil(((z_a + z_beta) / dz_abs) ** 2)) if dz_abs > 0 else 10 ** 9

        req_holm = req_n(ALPHA / len(PAIRS))
        decision = "sufficient" if n >= req_holm else "insufficient - report measured gain under protocol"
        pw_rows.append({
            "comparison": f"{r['a']}_vs_{r['b']}", "metric": r["metric"],
            "alpha": ALPHA, "power_target": POWER,
            "pilot_mean_diff": round(r["mean_diff"], 4), "pilot_std_diff": round(r["std_diff"], 4),
            "effect_size_dz": round(r["dz"], 4), "required_n_uncorrected": req_n(ALPHA),
            "required_n_holm": req_holm, "current_n": n, "decision": decision,
        })

    for path, fields, rows in [(OUT_DIR / "paired_statistics_extended.csv", ps_fields, ps_rows),
                                (OUT_DIR / "power_analysis.csv", pw_fields, pw_rows)]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
