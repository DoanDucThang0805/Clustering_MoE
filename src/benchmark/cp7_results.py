"""CP7 — Soft MoE baseline → `soft_moe_baseline.csv` (PDF mục X).

Eval `soft_moe` (5 seed, PlantDoc test) + gộp hàng `learned_gate_moe` và `cluster_moe`
(top-2) từ CP1 để bảng tự chứa, so được chuỗi:

    soft_moe (learned + SOFT all-4)  ->  learned_gate_moe (learned + top-2)
                                     ->  cluster_moe (prototype + top-2)

CỘT PARAMS — điểm quan trọng:
  * `params_m`       = **ACTIVE params** (thop.profile chỉ đếm module ĐƯỢC GỌI).
                       Giữ đúng tên theo schema PDF; chọn *active* vì PDF nguyên tắc II:
                       "báo cáo active computational cost, không phải tổng tham số".
  * `params_total_m` = cột PHỤ = sum(numel) TRỪ classifier ImageNet chết trong backbone
                       wrapper (~1.62 M, forward không bao giờ gọi) → chứng minh các model
                       CAPACITY-MATCHED.

Chạy từ src/:
    python -m benchmark.cp7_results
"""
from __future__ import annotations

import argparse
import csv
import glob
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch
from thop import profile
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 43, 44, 45, 46]
BACKBONE = "mobilenetv3small_torchvision"

ROUTING_TYPE = {
    "soft_moe": "learned_soft",
    "learned_gate_moe": "learned_topk",
    "cluster_moe": "cosine_topk",
}
# latency (Pi, ONNX) — soft_moe chưa benchmark trên edge -> để trống
LATENCY_MS = {"learned_gate_moe": 8.2498, "cluster_moe": 7.9588, "soft_moe": ""}

FIELDS = [
    "seed", "dataset", "backbone", "model", "num_experts", "routing_type",
    "accuracy", "macro_f1", "weighted_f1",
    "params_m", "flops_g", "latency_ms",
    "params_total_m",          # cột PHỤ (thêm cuối, không phá schema PDF)
]


# ─────────────────────────────────────────────────────────────
# Params / FLOPs
# ─────────────────────────────────────────────────────────────
def _dead_classifier_params(model) -> int:
    """Params của classifier ImageNet trong backbone wrapper — forward KHÔNG gọi."""
    for attr in ("feature_extractor", "backbone"):
        bb = getattr(model, attr, None)
        inner = getattr(bb, "model", None) if bb is not None else None
        clf = getattr(inner, "classifier", None) if inner is not None else None
        if clf is not None:
            return sum(p.numel() for p in clf.parameters())
    return 0


def complexity(model, inputs) -> tuple[float, float, float]:
    """(params_active_m, flops_g, params_total_m)."""
    macs, thop_params = profile(model, inputs=inputs, verbose=False)
    total_used = sum(p.numel() for p in model.parameters()) - _dead_classifier_params(model)
    return (
        round(thop_params / 1e6, 4),        # ACTIVE (chỉ module được gọi)
        round(2.0 * macs / 1e9, 4),         # FLOPs = 2 x MACs (giữ quy ước cũ)
        round(total_used / 1e6, 4),         # TOTAL thực dùng (đã trừ params chết)
    )


# ─────────────────────────────────────────────────────────────
# Soft MoE — eval 5 seed
# ─────────────────────────────────────────────────────────────
def _best_run_by_val(seed: int, dataset: str, num_experts: int) -> tuple[Path, float, int]:
    """Chọn run có VALIDATION accuracy cao nhất trong các restart của seed này.

    QUAN TRỌNG: chọn theo VAL, TUYỆT ĐỐI KHÔNG theo test — test chỉ 285 ảnh
    (1 ảnh = 0.35 điểm), chọn theo test = chọn nhiễu / overfit test set.
    """
    seed_dir = (ROOT / "checkpoints" / dataset / "soft_moe"
                / f"{BACKBONE}_softmoe" / f"{num_experts}_experts" / f"seed_{seed}")
    cands = sorted(glob.glob(str(seed_dir / "run_*" / "best_checkpoint.pth")))
    if not cands:
        raise FileNotFoundError(f"No soft_moe checkpoint under {seed_dir}")

    best_path, best_val = None, -float("inf")
    for p in cands:
        ck = torch.load(p, map_location="cpu")
        val = max(ck.get("val_acc_history", [-float("inf")]))
        if val > best_val:
            best_path, best_val = Path(p), val
    return best_path, best_val, len(cands)


def eval_soft_moe(seed: int, dataset: str, num_experts: int):
    from models.soft_moe.model import build_soft_moe_from_checkpoint
    from datasets.registry import get_moe_build

    ckpt_path, best_val, n_runs = _best_run_by_val(seed, dataset, num_experts)
    print(f"    seed {seed}: chọn {ckpt_path.parent.name} "
          f"(best-of-{n_runs} theo VAL={best_val:.2f}%)")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = build_soft_moe_from_checkpoint(ckpt).to(DEVICE).eval()

    _, _, test_ds = get_moe_build(dataset)(use_context=True)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    labs, preds = [], []
    with torch.inference_mode():
        for images, labels, context in loader:
            logits, _ = model(images.to(DEVICE), context.to(DEVICE))
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
            labs.extend(labels.tolist())

    metrics = dict(
        accuracy=round(float(accuracy_score(labs, preds)), 4),
        macro_f1=round(float(f1_score(labs, preds, average="macro")), 4),
        weighted_f1=round(float(f1_score(labs, preds, average="weighted")), 4),
    )
    return metrics, model, int(ckpt["num_classes"]), int(ckpt["context_dim"])


# ─────────────────────────────────────────────────────────────
# CP1 rows (learned_gate_moe, cluster_moe) — gộp để bảng tự chứa
# ─────────────────────────────────────────────────────────────
def cp1_complexity(num_classes: int, context_dim: int):
    """Tính lại params/flops cho MoE (top-2) và Cluster-MoE (top-2) theo CÙNG chuẩn."""
    from models.moe.model import MoEModel
    from models.clustering_moe.model import ClusteringMoEModel

    img = torch.randn(1, 3, 224, 224)
    ctx = torch.randn(1, context_dim)

    moe = MoEModel(context_dim=context_dim, num_classes=num_classes, num_experts=4, top_k=2,
                   router_mode="context_aware", backbone_name=BACKBONE,
                   pretrain_backbone=False, temperature=0.5).eval()
    clu = ClusteringMoEModel(num_classes=num_classes, centroids=torch.randn(4, 576), top_k=2,
                             backbone_name=BACKBONE, metric="cosine",
                             pretrain_backbone=False, temperature=0.5).eval()
    return {
        "learned_gate_moe": complexity(moe, (img, ctx)),
        "cluster_moe": complexity(clu, (img,)),
    }


def cp1_rows(cp1_csv: Path, dataset: str, comp: dict) -> list[dict]:
    rows = []
    if not cp1_csv.exists():
        print(f"[WARN] Không thấy {cp1_csv} — bỏ qua hàng CP1.")
        return rows
    for r in csv.DictReader(open(cp1_csv)):
        m = r["model"]
        if r["initialization"] != "imagenet_pretrained" or m not in comp:
            continue
        if int(r["seed"]) not in SEEDS:      # chỉ lấy 5 seed để so cùng n
            continue
        p_act, fl, p_tot = comp[m]
        rows.append({
            "seed": int(r["seed"]), "dataset": dataset, "backbone": BACKBONE, "model": m,
            "num_experts": 4, "routing_type": ROUTING_TYPE[m],
            "accuracy": float(r["accuracy"]), "macro_f1": float(r["macro_f1"]),
            "weighted_f1": float(r["weighted_f1"]),
            "params_m": p_act, "flops_g": fl, "latency_ms": LATENCY_MS[m],
            "params_total_m": p_tot,
        })
    return rows


# ─────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="plantdoc")
    ap.add_argument("--num_experts", type=int, default=4)
    ap.add_argument("--seeds", type=str, default="42 43 44 45 46")
    ap.add_argument("--cp1_csv", type=Path,
                    default=ROOT / "paper_results/tables/pretrained_backbone_results.csv")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "paper_results/tables/soft_moe_baseline.csv")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split()]

    rows: list[dict] = []
    soft_comp = None
    num_classes = context_dim = None

    for s in seeds:
        try:
            metrics, model, num_classes, context_dim = eval_soft_moe(
                s, args.dataset, args.num_experts
            )
        except FileNotFoundError as e:
            print(f"[SKIP] soft_moe seed {s}: {e}")
            continue
        if soft_comp is None:
            img = torch.randn(1, 3, 224, 224).to(DEVICE)
            ctx = torch.randn(1, context_dim).to(DEVICE)
            soft_comp = complexity(model, (img, ctx))
        p_act, fl, p_tot = soft_comp
        rows.append({
            "seed": s, "dataset": args.dataset, "backbone": BACKBONE, "model": "soft_moe",
            "num_experts": args.num_experts, "routing_type": ROUTING_TYPE["soft_moe"],
            **metrics,
            "params_m": p_act, "flops_g": fl, "latency_ms": LATENCY_MS["soft_moe"],
            "params_total_m": p_tot,
        })
        print(f"[soft_moe seed {s}] acc={metrics['accuracy']} "
              f"mF1={metrics['macro_f1']} wF1={metrics['weighted_f1']}")

    if num_classes is None:
        raise SystemExit("Không có checkpoint soft_moe nào — chưa train xong?")

    rows += cp1_rows(args.cp1_csv, args.dataset, cp1_complexity(num_classes, context_dim))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {args.out}")

    # ── mean ± std ──
    agg = defaultdict(lambda: defaultdict(list))
    meta = {}
    for r in rows:
        for k in ("accuracy", "macro_f1", "weighted_f1"):
            agg[r["model"]][k].append(float(r[k]))
        meta[r["model"]] = (r["params_m"], r["params_total_m"], r["flops_g"])

    print(f"\n{'model':<18}{'n':>3}  {'accuracy':>15} {'macro_f1':>15} {'weighted_f1':>15}"
          f"  {'act.P(M)':>9} {'tot.P(M)':>9} {'FLOPs(G)':>9}")
    print("-" * 104)
    for m in ("soft_moe", "learned_gate_moe", "cluster_moe"):
        if m not in agg:
            continue
        v = agg[m]
        n = len(v["accuracy"])
        pa, pt, fl = meta[m]
        cells = "  ".join(
            f"{st.mean(v[k]):.4f}±{st.pstdev(v[k]):.4f}"
            for k in ("accuracy", "macro_f1", "weighted_f1")
        )
        print(f"{m:<18}{n:>3}  {cells}  {pa:>9} {pt:>9} {fl:>9}")


if __name__ == "__main__":
    main()
