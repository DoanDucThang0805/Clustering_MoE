# CP2 (mở rộng) — Routing Entropy + Đường cong τ (pretrained, G=4, top-k=2)

## Mục tiêu
Hoàn tất CP2: ngoài bảng entropy tại τ=0.5 (đã có `routing_entropy.csv`, 20 dòng), bổ sung **hình đường cong τ** cho PDF mục V.C — để chứng minh **τ=0.5 là điểm cân bằng** giữa routing quá sắc và quá mềm.

**Deliverable chính = 1 HÌNH gồm 2 đường cong** (Cluster-MoE + learned-gate MoE), trục hoành τ, trục tung normalized entropy H_norm (thêm trục tung phụ Macro-F1 theo PDF). **Mỗi đường 4 điểm** ứng với τ ∈ {0.3, 0.5, 0.7, 1.0}, **mỗi điểm = 1 seed (42)**.

**Phạm vi (user chốt):** backbone **pretrained**, **G=4, top-k=2**, **1 seed = 42**, **τ ∈ {0.3, 0.5, 0.7, 1.0}**. 1 seed/điểm → không error band; hợp lệ vì H_norm biến thiên <0.01 across 10 seed tại τ=0.5 (đã đo ở CP2 chính) → 1 seed đại diện tốt.

## Hiện trạng — đã có τ=0.5, chỉ thiếu τ=0.3/0.7/1.0
| Model | τ=0.5 (đã có, seed 42) | τ=0.3/0.7/1.0 (cần train seed 42) |
|---|---|---|
| Cluster-MoE | `dense_aligned_pretrain_backbone` champion s42 | ❌ train thêm |
| Learned-gate MoE | `moe_temperature_0.5_pretrain_backbone` s42 | ❌ train thêm |

→ Cần train **6 lượt** = 3 τ × 2 model × 1 seed (42). τ=0.5 tái dùng checkpoint có sẵn.

## Nguyên tắc nhất quán (để đường cong chỉ khác nhau ở τ)
Trong **mỗi seed**, mọi τ phải dùng **cùng backbone + cùng centroid + cùng recipe**, chỉ đổi τ. Recipe khớp champion τ=0.5 của seed đó.

### A. Cluster-MoE seed 42 — recipe champion (lr **3e-4**, wd 1e-2, ls 0.05)
Dùng lại đúng backbone + centroid của champion τ=0.5 seed 42 (đã verify tồn tại):
- backbone_checkpoint: `checkpoints/plantdoc/pretrain_baseline/mobilenetv3small_torchvision/seed_42/run_20260629-101639/best_checkpoint.pth`
- centroid: `clustering_results/plantdoc/pretrain_backbone/mobilenetv3small_torchvision_backbone/kmeans/cosine/seed_42/clusters_kmeans_G4_seed42.npz`

Lệnh (từ `src/`, với τ ∈ {0.3,0.7,1.0}) — gọi python trực tiếp để đổi τ (`cluster_moe_train.sh` hardcode τ=0.5):
```
python -m training.clustering_moe \
  --seed 42 --num_experts 4 --top_k 2 --distance_metric cosine --temperature <TAU> \
  --backbone_checkpoint <ROOT>/checkpoints/plantdoc/pretrain_baseline/mobilenetv3small_torchvision/seed_42/run_20260629-101639/best_checkpoint.pth \
  --lr 3e-4 --weight_decay 1e-2 --label_smoothing 0.05 \
  --num_epochs 400 --batch_size 32 --dataset_name plantdoc \
  --backbone_type dense_aligned_pretrain_backbone \
  --centroid_backbone_type pretrain_backbone \
  --backbone_name mobilenetv3small_torchvision --model_clustering_name kmeans --pretrain_backbone
```
Verify mỗi run: log có `Loaded 240 backbone tensors from .../seed_42/run_20260629-101639/best_checkpoint.pth` + `Loaded centroids (4,576)`. Checkpoint lưu vào `temperature_<TAU>/G4_cosine_top2/seed_42`.

### B. Learned-gate MoE seed 42 — recipe CP1 (ImageNet pretrained, context_aware, lr 1e-3)
```
bash moe_train.sh --seed 42 --temperature <TAU> --type_model moe_temperature_<TAU>_pretrain_backbone
```
(moe_train.sh mặc định pretrained=True, G4 top2, context_aware — chỉ cần đổi `--temperature` + `--type_model`.) 1 lần, τ ∈ {0.3,0.7,1.0}.

## Bước 3 — Tính entropy cho toàn bộ sweep (seed 42, 4 τ, 2 model)
Sửa nhẹ `src/diagnostics/routing_entropy.py` để nhận `--tau` (Cluster-MoE: đổi path `temperature_<TAU>`) và `--type_model` (learned-gate: `moe_temperature_<TAU>_pretrain_backbone`); giữ nguyên logic lấy α (weights / re-run gating). Chạy cho **8 điểm** = 2 model × 4 τ (seed 42) → file riêng:
```
mean_acc_mF1_results/routing_entropy_tau_sweep.csv   # schema y hệt routing_entropy.csv
```
τ=0.5 tái dùng checkpoint seed 42 đã có, không train lại.

## Bước 4 — Vẽ HÌNH đường cong τ (deliverable chính)
Từ `routing_entropy_tau_sweep.csv` (8 điểm) vẽ 1 hình:
- Trục hoành: τ = 0.3/0.5/0.7/1.0.
- Trục tung trái: **normalized_entropy** — 2 đường (Cluster-MoE, learned-gate MoE), mỗi đường 4 điểm nối liền.
- Trục tung phải (tùy chọn, cùng trục hoành): **Macro-F1** — 2 đường nét đứt, để chỉ ra τ=0.5 gần đỉnh Macro-F1.
- Đánh dấu điểm τ=0.5 (vd đường gạch dọc) làm "điểm cân bằng".
- Lưu `figures/cp2_tau_curve.png` (+ CSV bảng số kèm theo).

Kèm bảng số (model × τ): `normalized_entropy`, `mean_entropy`, `macro_f1`, `accuracy`, `usage_cv` (1 giá trị/điểm, seed 42).

## Sanity check
- Mỗi (model,τ): normalized_entropy ∈ [0,1]; mean_entropy ≤ log2≈0.693; Σ expert_usage = 2.
- **Kỳ vọng đường cong Cluster-MoE:** normalized_entropy TĂNG theo τ (τ nhỏ → routing sắc → H thấp; τ lớn → mềm → H cao). Nếu ngược lại phải xem lại.
- Macro-F1 theo τ: kỳ vọng đỉnh/gãy quanh τ=0.5 để hỗ trợ claim "điểm cân bằng". Nếu τ=0.5 không phải đỉnh → điều chỉnh claim theo bằng chứng (PDF mục XIII).

## Caveat cần ghi khi viết bài
- **1 seed (42) → không có error bar**, đường cong chỉ minh họa xu hướng, không kết luận thống kê.
- τ=0.5 là **best-of-restart** (champion), các τ khác **1 lần train** → điểm accuracy τ=0.5 có thể nhỉnh do lựa chọn, không thuần do τ. Đường cong **entropy**-vs-τ ít bị ảnh hưởng. Nếu cần công bằng tuyệt đối cho accuracy: train lại τ=0.5 1-lần cùng recipe.
- So sánh normalized_entropy **giữa** 2 model ở cùng τ khập khiễng: Cluster-MoE softmax trên cosine-sim (chặn [-1,1]), learned-gate trên logit không chặn → thang đầu vào khác. Diễn giải đường cong **trong từng model** là chính; so chéo cần caveat.

## Ước lượng thời gian
- 6 lượt train (3 Cluster-MoE ~15-20'/run + 3 learned-gate ~9'/run) + entropy (<5') ≈ ~1-1.5 giờ. Chạy nền, báo từng run.

## Ràng buộc
- KHÔNG đụng champion τ=0.5 đã commit (bảng kết quả chính) — τ-sweep là namespace/CSV riêng.
- Checkpoint `.pth` không lên git; commit CSV + script + **hình `figures/cp2_tau_curve.png`**.

## Tóm tắt deliverable
1. 6 checkpoint mới (seed 42, τ=0.3/0.7/1.0, 2 model) — trên đĩa, không lên git.
2. `mean_acc_mF1_results/routing_entropy_tau_sweep.csv` — 8 dòng (2 model × 4 τ).
3. **`figures/cp2_tau_curve.png`** — hình 2 đường cong H_norm-vs-τ (+ Macro-F1) → HÌNH cho bài báo.
4. `routing_entropy.py` cập nhật (thêm `--tau`/`--type_model`).
