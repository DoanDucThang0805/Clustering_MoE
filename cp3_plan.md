# CP3 — Expert masking ablation → `expert_masking_ablation.csv` + `conditional_confusion_summary.csv`

## Mục tiêu
Can thiệp tại inference (KHÔNG retrain, KHÔNG sửa kiến trúc): tắt từng expert g, renormalize trọng số còn lại (tài liệu eq. 4):
- nếu g ∈ K_i: α̃_{i,j} = α_{i,j}·1(j≠g) / (Σ_{ℓ∈K_i, ℓ≠g} α_{i,ℓ} + ε)
Đo suy giảm Accuracy/Macro-F1/Weighted-F1 + per-class F1 → bằng chứng can thiệp cho expert specialization. Kèm conditional confusion matrix theo primary expert.

## Hiện trạng repo
- Đối tượng chính: Cluster-MoE champion 10 seed (dense_aligned, G4 cosine top2 τ0.5) — đường dẫn xem cp2_plan.md.
- `MoELayer.forward` của Cluster-MoE ([src/models/clustering_moe/model.py](src/models/clustering_moe/model.py)) nhận `weights, top_indices` từ gating rồi loop expert → điểm can thiệp sạch nhất là **giữa gating và expert computation**.
- Có thể làm tương tự cho learned-gate MoE (pretrained, sau CP1) để đối chiếu "structured routing" giữa 2 router.

## Việc cần làm
1. Viết `src/diagnostics/expert_masking.py` — TUYỆT ĐỐI không sửa file model; can thiệp trong script:
   - Load model champion (copy pattern từ `src/inference/cluster_moe_model/inference.py`).
   - Chạy test set, với mỗi batch: gọi `model.backbone(x)` → `model.moe_layer.gating(embedding)` lấy `(weights, top_indices, scores)` → áp mask expert g: zero hóa weight tại vị trí `top_indices == g`, renormalize theo eq. 4 (ε=1e-9; nếu cả k expert đều bị mask — không xảy ra với 1 mask — thì giữ 0) → tự loop expert như trong `MoELayer.forward` (copy nguyên logic 15 dòng) → residual + norm + classifier như `ClusteringMoEModel.forward`.
   - `mask_expert ∈ {none, 0, 1, 2, 3}`; `none` = baseline không mask (phải khớp 100% kết quả champion — dùng làm self-check).
   - Metrics mỗi cấu hình: accuracy, macro_f1, weighted_f1, per-class F1 (8 lớp), delta so với `none`.
2. Conditional confusion theo primary expert (cùng script hoặc `--mode confusion`):
   - primary_expert của mẫu i = `top_indices[i, 0]` (expert trọng số lớn nhất, không mask).
   - Gom mẫu test theo primary expert → mỗi nhóm: num_samples, accuracy, macro_f1, per-class recall, cặp lớp nhầm nhiều nhất (`most_confused_pair` dạng "true→pred", `most_confused_count`).
3. Xuất 2 CSV đúng schema tài liệu (đủ cột như PDF trang 4):
   - `mean_acc_mF1_results/expert_masking_ablation.csv`
   - `mean_acc_mF1_results/conditional_confusion_summary.csv`
   - Chạy cho cả 10 seed (mỗi seed 5 hàng masking: none + 4 expert).

## Tiêu chí hoàn thành / sanity check
- Hàng `mask_expert=none` khớp chính xác acc/mF1 champion từng seed (nếu lệch → lỗi tái dựng forward).
- Tổng num_samples của 4 nhóm primary expert = 285.
- Δ khi mask expert lớn (cụm nhiều mẫu — xem `expert_usage.png`/heatmap có sẵn trong run dir) phải ≥ Δ khi mask expert nhỏ về trung bình.

## Điều kiện claim (ghi vào bài đúng như tài liệu)
- Mask 1 expert làm giảm mạnh Macro-F1 hoặc F1 một nhóm lớp → được viết "expert đóng góp lớn cho vùng đặc trưng đó".
- Không suy giảm rõ → CHỈ viết "structured routing behavior", không viết "specialization".

## Phụ thuộc: không (checkpoint có sẵn). Thời gian: ~1 buổi code + ~15 phút chạy 10 seed × 5 mask.
