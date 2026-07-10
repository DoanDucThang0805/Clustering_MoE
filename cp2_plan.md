# CP2 — Routing entropy → `routing_entropy.csv`

## Mục tiêu
Tính routing entropy H̄, normalized entropy H_norm = H̄/log(k), expert-usage và usage CV trên **test split**, cho learned-gate MoE và Cluster-MoE, từ checkpoint có sẵn (KHÔNG retrain). Công thức (tài liệu, eq. 2–3):
- H_i = −Σ_{g∈K_i} α_{i,g} · log(α_{i,g} + ε), ε = 1e-9
- H̄ = mean(H_i), H_norm = H̄ / log(top_k)   (top_k=2 → log 2)

## Hiện trạng repo
- **Cluster-MoE**: `ClusteringMoEModel.forward` trả `(logits, weights, top_indices, scores)` — `weights [B, top_k]` chính là α sau softmax/τ → lấy trực tiếp.
- **Learned-gate MoE**: `MoEModel.forward` trả `(logits, clean_router_logits, top_k_indices)` — **KHÔNG trả α**. Phải gọi thêm `model.moe_layer.gating(feature_norm[, context])` → trả `(combined_weights, top_k_indices, clean_logits)`. Lưu ý eval mode để tắt noise.
- Checkpoint nguồn:
  - Cluster-MoE champion 10 seed: `checkpoints/plantdoc/clustering_moe/dense_aligned_pretrain_backbone/mobilenetv3small_torchvision_backbone/kmeans/temperature_0.5/G4_cosine_top2/seed_{42..51}/run_*/best_checkpoint.pth` (1 run/seed).
  - Learned-gate MoE pretrained: `checkpoints/plantdoc/moe_temperature_0.5_pretrain_backbone/mobilenetv3small_torchvision_moe/4_experts/top_2/seed_{42..51}/` (CP1 đang chạy — đợi xong).
  - (Tùy chọn đối chiếu) MoE non-pretrain cũ: `checkpoints/plantdoc/moe_temperature_0.5/...` seed 42–46.
- Script diagnostics tham khảo pattern load model + loop test: `src/diagnostics/moe_cluster_routing_diagnostic.py`, `src/diagnostics/moe_baseline_routing_diagnostics.py`.

## Việc cần làm
1. Viết `src/diagnostics/routing_entropy.py`:
   - Args: `--model_type {cluster_moe,moe}`, `--seeds 42..51`, `--split {test,validation}`, các arg namespace giống inference script tương ứng, `--output_csv`.
   - Với mỗi seed: load checkpoint (pattern load y hệt `src/inference/cluster_moe_model/inference.py` / `src/inference/moe_model/inference.py` — copy phần dựng model + load_state_dict, đừng viết mới).
   - Loop test set (batch 32, không shuffle):
     - Cluster-MoE: lấy `weights` từ forward.
     - MoE: forward backbone → pre_moe_norm → gọi `model.moe_layer.gating(...)` lấy `combined_weights`; model.eval() bắt buộc (tắt noisy).
     - H_i trên α của top-k; tích lũy usage count mỗi expert (đếm xuất hiện trong top_indices).
   - Tính: mean_entropy, std_entropy, normalized_entropy, expert_usage_1..4 (tỉ lệ, tổng = top_k), usage_cv = std(usage)/mean(usage), + accuracy/macro_f1 của chính lần chạy đó (sklearn).
2. Xuất CSV đúng schema tài liệu:
   ```
   seed,model,routing,G,top_k,tau,split,
   mean_entropy,std_entropy,normalized_entropy,
   expert_usage_1,expert_usage_2,expert_usage_3,expert_usage_4,
   usage_cv,accuracy,macro_f1
   ```
   → lưu `mean_acc_mF1_results/routing_entropy.csv`.
3. (Cho hình entropy-vs-τ trong bài) Chạy thêm chính script này trên các checkpoint temp sweep **non-pretrain** có sẵn (namespace `clustering_moe/non_pretrain_backbone`, temp 0.3/0.5/0.7/1.0, seed 42–46) → cùng CSV, phân biệt bằng cột tau. KHÔNG train thêm τ mới cho dense_aligned.

## Thứ tự & phụ thuộc
- Phần Cluster-MoE + temp sweep: chạy được NGAY (checkpoint có sẵn).
- Phần learned-gate MoE pretrained: đợi CP1 xong.

## Tiêu chí hoàn thành / sanity check
- H_norm ∈ [0,1]; top-k=2 → mean_entropy ≤ log 2 ≈ 0.693.
- Cluster-MoE τ=0.5 kỳ vọng H_norm thấp hơn τ=1.0 (routing sắc hơn khi τ nhỏ) — nếu ngược lại phải xem lại code.
- Σ expert_usage_i = top_k (mỗi mẫu chọn k expert).
- accuracy in CSV khớp report champion đã có (±0 — cùng checkpoint cùng test set).

## Thời gian: ~1 buổi code + <10 phút chạy (chỉ inference).
