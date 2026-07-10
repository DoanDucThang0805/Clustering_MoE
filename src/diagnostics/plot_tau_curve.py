"""Vẽ hình CP2: đường cong H_norm và Macro-F1 theo τ cho 2 model (seed 42).

2 panel dọc chia sẻ trục τ (KHÔNG dual-axis — anti-pattern #1).
Series phân biệt bằng màu + marker + line-style (an toàn cả khi in grayscale/CVD).
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "mean_acc_mF1_results" / "routing_entropy_tau_sweep.csv"
OUT = ROOT / "figures" / "cp2_tau_curve.png"

# Categorical slots 1-2 từ palette dataviz đã validated
STYLE = {
    "cluster_moe": dict(color="#2a78d6", marker="o", ls="-",  label="Cluster-MoE (cosine)"),
    "moe":         dict(color="#1baf7a", marker="s", ls="--", label="Learned-gate MoE"),
}
INK, MUTED = "#0b0b0b", "#52514e"

# ── Đọc dữ liệu: gộp nhiều seed → trung bình theo (model, tau) ──
from collections import defaultdict
acc_hn = defaultdict(list)   # (model, tau) -> [H_norm,...]
acc_mf = defaultdict(list)   # (model, tau) -> [Macro-F1,...]
with open(CSV) as f:
    for r in csv.DictReader(f):
        key = (r["model"], float(r["tau"]))
        acc_hn[key].append(float(r["normalized_entropy"]))
        acc_mf[key].append(float(r["macro_f1"]))

def _mean(xs):
    return sum(xs) / len(xs)

data = {"cluster_moe": {}, "moe": {}}
for (model, tau), hn in acc_hn.items():
    data[model][tau] = (_mean(hn), _mean(acc_mf[(model, tau)]))

taus = sorted({t for m in data.values() for t in m})
n_seeds = {m: max(len(acc_hn[(m, t)]) for t in taus) for m in data}
print("Số seed mỗi model (mean):", n_seeds)

fig, (ax_h, ax_f) = plt.subplots(
    2, 1, figsize=(6.2, 6.4), sharex=True,
    gridspec_kw=dict(height_ratios=[1, 1], hspace=0.12),
)

for model, st in STYLE.items():
    xs = taus
    hn = [data[model][t][0] for t in xs]
    mf = [data[model][t][1] for t in xs]
    ax_h.plot(xs, hn, color=st["color"], marker=st["marker"], ls=st["ls"],
              lw=2, ms=8, label=st["label"])
    ax_f.plot(xs, mf, color=st["color"], marker=st["marker"], ls=st["ls"],
              lw=2, ms=8, label=st["label"])

# Vạch τ=0.5 (điểm cân bằng)
for ax in (ax_h, ax_f):
    ax.axvline(0.5, color=MUTED, ls=":", lw=1.2, zorder=0)
    ax.grid(True, color="#e6e6e3", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)

ax_h.annotate("τ = 0.5", xy=(0.5, ax_h.get_ylim()[0]),
              xytext=(0.5, 0.30), ha="center", color=MUTED, fontsize=9)

ax_h.set_ylabel("Normalized entropy  $H_{norm}$", color=INK)
ax_h.set_ylim(0, 1.0)
ax_f.set_ylabel("Macro-F1", color=INK)
ax_f.set_xlabel("Routing temperature  τ", color=INK)
ax_f.set_xticks(taus)

ax_h.legend(frameon=False, loc="center right", labelcolor=INK, fontsize=9)
ax_h.set_title("Routing sharpness và Macro-F1 theo τ (mean 5 seed 42–46, G=4, top-k=2)",
               color=INK, fontsize=11, pad=8)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
