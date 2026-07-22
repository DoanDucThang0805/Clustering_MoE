"""Tái tạo pretrained_backbone_results.csv (CP1, RAW per-seed) — bảng bằng
chứng thô cho reviewer (mục "release raw seed-wise CSV files").

from_scratch rows: không đổi từ bản đã khoá trước đây (không bị ảnh hưởng bởi
retrain2, giữ nguyên).
imagenet_pretrained rows: tính lại bằng đúng eval_dense/eval_moe/eval_cluster
của cp1_direct_inference.py (dense = retrain2 khoá, moe/cluster = checkpoint
gốc khoá) để khớp 100% với cp1_backbone_init_summary.csv hiện hành.

Chạy từ src/:
    python -m benchmark.cp1_raw_seedwise
"""
from __future__ import annotations

import csv
from pathlib import Path

from benchmark.cp1_direct_inference import (
    EVALUATORS, MODEL_ORDER, SEEDS, PARAMS_M, FLOPS_G,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_results" / "tables" / "pretrained_backbone_results.csv"

FIELDS = ["seed", "initialization", "backbone", "model", "routing", "cluster_method",
          "G", "top_k", "tau", "accuracy", "macro_f1", "weighted_f1",
          "params_m", "flops_g", "model_size_mb", "peak_cpu_memory_mb", "cpu_latency_ms"]

ROUTING = {"dense": "none", "learned_gate_moe": "learned", "cluster_moe": "cosine"}
CLUSTER_METHOD = {"dense": "none", "learned_gate_moe": "none", "cluster_moe": "kmeans"}

# from_scratch rows giữ nguyên từ bản đã khoá trước đây (không đổi, không bị
# ảnh hưởng bởi retrain2 — retrain2 chỉ áp dụng cho dense pretrained).
FROM_SCRATCH_ROWS = [
    (42, "dense", 0.7544, 0.7102, 0.7521), (43, "dense", 0.7825, 0.7531, 0.7806),
    (44, "dense", 0.7509, 0.7208, 0.7527), (45, "dense", 0.7649, 0.7272, 0.7652),
    (46, "dense", 0.7754, 0.7419, 0.7733),
    (42, "learned_gate_moe", 0.786, 0.7463, 0.7839), (43, "learned_gate_moe", 0.7825, 0.7537, 0.7793),
    (44, "learned_gate_moe", 0.7754, 0.7348, 0.7739), (45, "learned_gate_moe", 0.7965, 0.768, 0.7955),
    (46, "learned_gate_moe", 0.7789, 0.7463, 0.7747),
    (42, "cluster_moe", 0.8035, 0.7729, 0.8044), (43, "cluster_moe", 0.7895, 0.7492, 0.7895),
    (44, "cluster_moe", 0.814, 0.7874, 0.8126), (45, "cluster_moe", 0.8351, 0.8044, 0.8326),
    (46, "cluster_moe", 0.7719, 0.7204, 0.7683),
]


def _row(seed, init, model, acc, mf1, wf1):
    return {
        "seed": seed, "initialization": init, "backbone": "mobilenetv3small",
        "model": model, "routing": ROUTING[model], "cluster_method": CLUSTER_METHOD[model],
        "G": 4 if model != "dense" else "", "top_k": 2 if model != "dense" else "",
        "tau": 0.5 if model != "dense" else "",
        "accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1,
        "params_m": PARAMS_M[model], "flops_g": FLOPS_G[model],
        "model_size_mb": "", "peak_cpu_memory_mb": "", "cpu_latency_ms": "",
    }


def main() -> None:
    rows = [_row(seed, "from_scratch", model, acc, mf1, wf1)
            for seed, model, acc, mf1, wf1 in FROM_SCRATCH_ROWS]

    for model in MODEL_ORDER:
        print(f"\n[{model}]")
        for seed in SEEDS:
            m = EVALUATORS[model](seed)
            print(f"  seed={seed}: acc={m['accuracy']:.4f} mF1={m['macro_f1']:.4f} wF1={m['weighted_f1']:.4f}")
            rows.append(_row(seed, "imagenet_pretrained", model,
                              round(m["accuracy"], 4), round(m["macro_f1"], 4), round(m["weighted_f1"], 4)))

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
