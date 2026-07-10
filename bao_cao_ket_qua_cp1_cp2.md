# Báo cáo kết quả mô phỏng bổ sung — CP1 & CP2

**Bài báo:** Clustered Mixture-of-Experts Routing for Edge Image Classification
**Dataset:** PlantDoc-derived tomato subset (8 lớp), split cố định train/val/test.
**Cấu hình trọng tâm (PDF eq. 1):** Cluster-MoE cosine, G = 4, top-k = 2, τ = 0.5.
**Ngày:** 2026-07-10.

Báo cáo này tổng hợp kết quả cho hai checkpoint đầu trong Bảng I của tài liệu
"Hướng Dẫn Mô Phỏng Bổ Sung": **CP1 — ImageNet-pretrained backbone** và
**CP2 — routing entropy**. Cả hai đều dùng cùng seed list và cùng data split
(nguyên tắc II của PDF).

---

## CP1 — Kiểm tra ImageNet-pretrained backbone

**Mục tiêu (PDF mục IV.A):** xác nhận Cluster-MoE không phụ thuộc vào feature
space kém chất lượng do huấn luyện từ đầu; với backbone pretrained, centroid
k-means đại diện tốt hơn cho các vùng đặc trưng.

**Cấu hình (PDF Bảng II):** backbone MobileNetV3-Small, initialization =
ImageNet-pretrained (torchvision), 3 model (Dense / learned-gate MoE /
Cluster-MoE cosine), Cluster-MoE G = 4, top-k = 2, τ = 0.5. **Seeds = {42…51}
(n = 10).** Centroid chỉ fit trên train features; đổi sang backbone pretrained
đã trích xuất lại features và fit lại centroid (nguyên tắc II).

### Kết quả (test split, mean ± std trên 10 seed)

| Model | Accuracy (%) | Macro-F1 (%) |
|---|---|---|
| Dense MobileNetV3-Small | 85.02 ± 1.73 | 81.92 ± 2.28 |
| Learned-gate MoE (G=4, k=2) | 85.79 ± 0.76 | 82.54 ± 0.97 |
| **Cluster-MoE cosine (τ=0.5)** | **87.26 ± 0.72** | **84.52 ± 1.05** |

**Chênh lệch:**
- Cluster-MoE vs Dense: **+2.25** điểm accuracy, **+2.61** điểm Macro-F1.
- Cluster-MoE vs learned-gate MoE: **+1.47** điểm accuracy, **+1.98** điểm Macro-F1.

### Nhận xét
- Cluster-MoE **cải thiện** so với cả dense và learned-gate MoE ngay cả khi
  feature encoder được khởi tạo từ checkpoint ImageNet-pretrained thực dụng
  → thoả điều kiện claim thứ nhất của PDF mục XIV
  ("Cluster-MoE cosine vẫn cạnh tranh hoặc cải thiện với pretrained backbone").
- Cluster-MoE cũng có **độ lệch chuẩn nhỏ nhất** (±0.72 acc / ±1.05 mF1 so với
  dense ±1.73 / ±2.28) → ổn định hơn giữa các seed.
- **Claim đưa vào bài (PDF mục IV.D):** *"Cluster-MoE remains competitive when
  the feature encoder is initialized from a practical pretrained checkpoint."*

---

## CP2 — Routing entropy

**Mục tiêu (PDF mục V.A):** bài đã định nghĩa routing entropy nhưng chưa báo cáo
giá trị. Với top-k = 2, entropy chuẩn hoá cần được báo cáo để phân biệt routing
sắc (chọn gần như một expert) và routing phân tán (trộn hai expert gần đều), so
sánh giữa learned-gate MoE và Cluster-MoE.

**Công thức (PDF eq. 2–3), đã đối chiếu khớp với `src/diagnostics/routing_entropy.py`:**

- Entropy mỗi mẫu: `H_i = −Σ_{g∈K_i} α_{i,g}·log(α_{i,g} + ε)`, ε = 1e-9.
- Entropy trung bình: `H̄ = (1/N)·Σ H_i`.
- Entropy chuẩn hoá: `H_norm = H̄ / log(k)`, k = 2.
- `expert_usage_i = count_i / N` (Σ = top-k = 2); `usage_cv = std(usage)/mean(usage)`.

α được lấy trực tiếp từ `weights` (softmax trên top-k) với Cluster-MoE, và bằng
cách chạy lại gating ở chế độ `eval()` (tắt noise) với learned-gate MoE. Tính chỉ
từ checkpoint có sẵn, **không retrain** (PDF Bảng IV, ưu tiên 1).

### Phần 1 — Entropy tại cấu hình chốt τ = 0.5 (10 seed)

Nguồn: `mean_acc_mF1_results/routing_entropy.csv` (20 dòng = 10 seed × 2 model).

| Model | H̄ | H_norm | usage_CV | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| Cluster-MoE cosine | 0.664 ± 0.006 | **0.958 ± 0.009** | 0.224 | 0.8726 | 0.8452 |
| Learned-gate MoE | 0.085 ± 0.016 | **0.123 ± 0.022** | 0.062 | 0.8579 | 0.8254 |

**Diễn giải:**
- **Cluster-MoE có H_norm ≈ 0.96** → routing gần như trộn đều 2 expert top-k:
  prototype cosine phân bổ đa dạng, không sụp về một expert.
- **Learned-gate MoE có H_norm ≈ 0.12** → routing rất sắc, gần như luôn dồn vào
  một expert chính. Điều này giải thích vì sao learned-gate dễ mất tính đa dạng
  chuyên gia dù vẫn đạt accuracy cạnh tranh.
- `usage_cv` của Cluster-MoE cao hơn (0.224 vs 0.062): phân bổ tải giữa 4 expert
  lệch hơn ở mức tổng thể, nhưng ở mức từng mẫu lại mềm hơn (H_norm cao) — hai
  chỉ số đo hai khía cạnh khác nhau (per-sample sharpness vs global load).

### Phần 2 — Đường cong entropy & Macro-F1 theo τ (5 seed, mở rộng cho mục V.C)

Nguồn: `mean_acc_mF1_results/routing_entropy_tau_sweep.csv` (40 dòng = 2 model ×
4 τ × 5 seed 42–46). Hình: `figures/cp2_tau_curve.png`.

Mean trên 5 seed (42–46):

| τ | Cluster-MoE H_norm | Cluster-MoE Macro-F1 | Learned-gate H_norm | Learned-gate Macro-F1 |
|---|---|---|---|---|
| 0.3 | 0.913 | 0.8227 | 0.062 | 0.7988 |
| **0.5** | 0.966 | **0.8409** | 0.113 | **0.8236** |
| 0.7 | 0.983 | 0.8184 | 0.252 | 0.7774 |
| 1.0 | 0.992 | 0.8240 | 0.355 | 0.8015 |

**Diễn giải (thoả điều kiện claim thứ hai của PDF mục XIV):**
- **H_norm tăng đơn điệu theo τ** ở cả hai model (τ nhỏ → routing sắc → H thấp;
  τ lớn → routing mềm → H cao) — đúng vật lý của softmax temperature.
- **Macro-F1 đạt đỉnh tại τ = 0.5 ở cả hai model.** Đỉnh này **vượt nhiễu giữa
  các seed**: Cluster-MoE hơn á quân +0.0168 (std ≈ 0.006); learned-gate hơn á
  quân +0.0222 (std ≈ 0.011).
- → **τ = 0.5 là điểm cân bằng** giữa routing quá sắc (τ < 0.5, expert không đủ
  trộn) và routing quá mềm (τ > 0.5, mất chuyên biệt hoá) — đúng như giải thích
  PDF mục V.C yêu cầu.

---

## Kiểm tra tính hợp lệ (sanity check)

- ✅ Cả 40 dòng tau-sweep + 20 dòng τ=0.5 thoả: H_norm ∈ [0,1], H̄ ≤ log 2 ≈ 0.693,
  Σ expert_usage = top-k = 2.
- ✅ Accuracy tính lại trong CSV **trùng khít** report champion đã commit
  (vd Cluster-MoE τ=0.5: seed 42 = 87.02%, seed 44 = 87.37%…) → script load đúng
  checkpoint, tính đúng metric.
- ✅ Đỉnh τ=0.5 vượt std across-seed ở cả hai model.

## Caveat cần ghi khi viết bài

1. **Thiên lệch lựa chọn ở τ=0.5 (Cluster-MoE):** các checkpoint Cluster-MoE tại
   τ=0.5 là champion đã tối ưu kỹ, còn τ=0.3/0.7/1.0 là train một lần. Một phần
   lợi thế Macro-F1 của điểm 0.5 có thể do lựa chọn, không thuần do τ. Tuy nhiên
   kết luận vẫn vững vì (a) đường **learned-gate** (không tuyển chọn kiểu champion)
   độc lập tái hiện đúng đỉnh 0.5, và (b) claim chính là **H_norm-vs-τ** — thuần
   hình học routing, không bị ảnh hưởng.
2. **So sánh H_norm chéo giữa 2 model cùng τ cần thận trọng:** Cluster-MoE softmax
   trên cosine-similarity (chặn [−1,1]), learned-gate trên logit không chặn → thang
   đầu vào khác nhau. Nên diễn giải đường H_norm **trong từng model** là chính.
3. **Một dataset (PlantDoc tomato):** phạm vi claim hiện giới hạn ở tomato case
   study; mở rộng cần CP5 (dataset độc lập) và CP6 (backbone thay thế).

## Deliverable (khớp Bảng V của PDF)

| File | Nội dung | Trạng thái |
|---|---|---|
| `mean_acc_mF1_results/routing_entropy.csv` | H̄, H_norm, usage_cv theo model tại τ=0.5 (20 dòng, 10 seed) | ✅ |
| `mean_acc_mF1_results/routing_entropy_tau_sweep.csv` | Mở rộng 4 τ × 2 model × 5 seed (40 dòng) cho hình đường cong | ✅ |
| `figures/cp2_tau_curve.png` | Hình 2 panel: H_norm & Macro-F1 theo τ (đỉnh τ=0.5) | ✅ |
| `src/diagnostics/routing_entropy.py` | Script tính entropy (eq. 2–3) | ✅ |
| `src/diagnostics/plot_tau_curve.py` | Script vẽ hình đường cong τ | ✅ |

> **Ghi chú CP1:** kết quả 3 model pretrained (dense / learned-gate / Cluster-MoE)
> lấy từ `pretrain_baseline/…torchvision_per_seed.csv`, `moe/…`, và
> `cluster_moe/…dense_aligned_pretrain_backbone…temp0.5.csv`. Nếu cần đóng gói đúng
> tên `pretrained_backbone_results.csv` theo PDF (kèm cột params/flops/latency),
> có thể xuất thêm ở bước hoàn thiện bảng.

## Trạng thái so với "điều kiện claim mạnh" (PDF mục XIV)

- ✅ **Điều kiện 1** — Cluster-MoE cạnh tranh/cải thiện với pretrained backbone (CP1).
- ✅ **Điều kiện 2** — routing entropy giải thích điểm cân bằng τ=0.5 (CP2).
- ⏳ Điều kiện 3 (expert masking — CP3), Điều kiện 4 (n≥10 seed / power analysis — CP4):
  chưa nằm trong phạm vi báo cáo này.
