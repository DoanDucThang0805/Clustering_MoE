"""Xuất bảng và vẽ hình CP2 theo routing temperature.

Mỗi điểm trên hình là mean trên các seed có trong
``routing_entropy_tau_sweep.csv``. Hình dùng hai panel dọc chia sẻ trục τ,
tránh dual-axis. Script đồng thời xuất bảng entropy tổng hợp (mean và sample
standard deviation) để dùng trực tiếp trong phần kết quả của bài báo.
"""
from pathlib import Path
import csv
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "mean_acc_mF1_results" / "routing_entropy_tau_sweep.csv"
OUT = (
    ROOT / "paper_results" / "images"
    / "cp2_macro_f1_normalized_entropy_vs_temperature.png"
)
TABLE = (
    ROOT / "paper_results" / "tables"
    / "cp2_routing_entropy_temperature_sweep_summary.csv"
)

# Categorical slots 1-2 từ palette dataviz đã validated
STYLE = {
    "cluster_moe": dict(color="#2a78d6", marker="o", ls="-",  label="Cluster-MoE (cosine)"),
    "moe":         dict(color="#1baf7a", marker="s", ls="--", label="Learned-gate MoE"),
}
INK, MUTED = "#0b0b0b", "#52514e"

# ── Đọc dữ liệu: gộp nhiều seed → trung bình theo (model, tau) ──
from collections import defaultdict
acc_hn = defaultdict(list)   # (model, tau) -> [H_norm,...]
acc_hb = defaultdict(list)   # (model, tau) -> [H_bar,...]
acc_mf = defaultdict(list)   # (model, tau) -> [Macro-F1,...]
with open(CSV) as f:
    for r in csv.DictReader(f):
        key = (r["model"], float(r["tau"]))
        acc_hb[key].append(float(r["mean_entropy"]))
        acc_hn[key].append(float(r["normalized_entropy"]))
        acc_mf[key].append(float(r["macro_f1"]))

def _mean(xs):
    return sum(xs) / len(xs)

def _std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0

data = {"cluster_moe": {}, "moe": {}}
for (model, tau), hn in acc_hn.items():
    mf = acc_mf[(model, tau)]
    data[model][tau] = {
        "hbar_mean": _mean(acc_hb[(model, tau)]),
        "hbar_std": _std(acc_hb[(model, tau)]),
        "hnorm_mean": _mean(hn),
        "hnorm_std": _std(hn),
        "macro_f1_mean": _mean(mf),
        "macro_f1_std": _std(mf),
        "n_seeds": len(hn),
    }

taus = sorted({t for m in data.values() for t in m})
n_seeds = {m: max(len(acc_hn[(m, t)]) for t in taus) for m in data}
print("Số seed mỗi model (mean):", n_seeds)

# ── Bảng entropy cho bài báo ──────────────────────────────────────────────
TABLE.parent.mkdir(parents=True, exist_ok=True)
fields = [
    "model", "routing", "G", "top_k", "tau", "split", "n_seeds",
    "mean_entropy_mean", "mean_entropy_std",
    "normalized_entropy_mean", "normalized_entropy_std",
    "macro_f1_mean", "macro_f1_std",
]
with open(TABLE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for model in ("cluster_moe", "moe"):
        for tau in taus:
            d = data[model][tau]
            writer.writerow({
                "model": model,
                "routing": "cosine" if model == "cluster_moe" else "learned",
                "G": 4,
                "top_k": 2,
                "tau": f"{tau:.1f}",
                "split": "test",
                "n_seeds": d["n_seeds"],
                "mean_entropy_mean": f"{d['hbar_mean']:.4f}",
                "mean_entropy_std": f"{d['hbar_std']:.4f}",
                "normalized_entropy_mean": f"{d['hnorm_mean']:.4f}",
                "normalized_entropy_std": f"{d['hnorm_std']:.4f}",
                "macro_f1_mean": f"{d['macro_f1_mean']:.4f}",
                "macro_f1_std": f"{d['macro_f1_std']:.4f}",
            })
print(f"Saved: {TABLE}")

fig, (ax_h, ax_f) = plt.subplots(
    2, 1, figsize=(6.2, 5.8), sharex=True, layout="constrained",
    gridspec_kw=dict(height_ratios=[1, 1]),
)

for model, st in STYLE.items():
    xs = taus
    hn = [data[model][t]["hnorm_mean"] for t in xs]
    mf = [data[model][t]["macro_f1_mean"] for t in xs]
    ax_h.plot(
        xs, hn, color=st["color"], marker=st["marker"], ls=st["ls"],
        lw=2, ms=7, label=st["label"],
    )
    ax_f.plot(
        xs, mf, color=st["color"], marker=st["marker"], ls=st["ls"],
        lw=2, ms=7, label=st["label"],
    )

# Vạch τ=0.5 (điểm cân bằng)
for ax in (ax_h, ax_f):
    ax.axvline(0.5, color=MUTED, ls=":", lw=1.2, zorder=0)
    ax.grid(True, color="#e6e6e3", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)

ax_h.annotate(
    r"$\tau=0.5$", xy=(0.5, 0.04), xycoords=("data", "axes fraction"),
    xytext=(6, 0), textcoords="offset points", color=MUTED, fontsize=9,
)

ax_h.set_ylabel(r"Normalized entropy $H_{\mathrm{norm}}$", color=INK)
ax_h.set_ylim(0, 1.0)
ax_f.set_ylabel("Macro-F1", color=INK)
ax_f.set_xlabel(r"Routing temperature $\tau$", color=INK)
ax_f.set_xticks(taus)

ax_h.legend(frameon=False, loc="center right", labelcolor=INK, fontsize=9)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
