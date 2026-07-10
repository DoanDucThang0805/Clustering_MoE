# CP7 — Soft MoE baseline → `soft_moe_baseline.csv` (HOẶC hạ xuống Related Work)

## Quyết định cần chốt trước tiên
Tài liệu cho 2 phương án:
- **A (tốt nhất)**: implement Soft MoE classifier-side, cùng backbone / số expert / training budget.
- **B (rẻ)**: KHÔNG đưa Soft MoE vào bảng baseline chính; chỉ thảo luận Related Work, ghi rõ khác biệt (soft assignment vs cluster-prototype top-k routing).
→ Khuyến nghị: làm B trước để bài tự hoàn chỉnh; A chỉ khi còn thời gian sau CP2–CP6. Dưới đây là plan cho A.

## Phương án A — Implement Soft MoE classifier-side
### Thiết kế (baseline MỚI, không đụng model hiện có)
- File mới `src/models/soft_moe/model.py` — `SoftMoEModel`:
  - Backbone: `BACKBONE_REGISTRY` dùng chung (mobilenetv3small_torchvision, pretrained=True).
  - Soft routing classifier-side theo tinh thần Soft MoE (Puigcerver et al.): thay top-k dispatch bằng **soft assignment qua slot**:
    - dispatch: `D = softmax(X W, axis=slots)`, slot input = D^T X; mỗi expert xử lý slot của nó; combine: `C = softmax(X W, axis=experts·slots)`, output = C · expert_outputs.
    - Cấu hình tối thiểu: 4 expert × 1 slot/expert (so sánh công bằng E=4), expert MLP y hệt expert hiện tại (Linear 576→1024→576, LN, GELU, Dropout 0.1).
  - Residual + LayerNorm + classifier head y hệt `MoEModel` (copy) — chỉ khác cơ chế routing.
- Trainer: dùng lại `MoETrainer` được KHÔNG? — Không thẳng được (loss aux + chữ ký forward khác). Viết `src/utils/soft_moe_trainer.py` copy `moe_trainer.py`, bỏ auxiliary loss (Soft MoE không cần load-balance loss vì mọi expert đều nhận gradient), criterion = CrossEntropy weighted như dense.
- Training budget GIỐNG learned-gate MoE: 400 epochs, batch 64, lr 1e-3, wd 1e-3, warmup 10, cosine, patience 50, cùng augmentation, cùng seed list.

### Chạy
- Seeds {42,43,44,45,46} (5 seed đủ cho một hàng baseline; 10 nếu rảnh GPU).
- Script `soft_moe_train.sh` copy pattern `moe_train.sh` (loop seed + is_done).
- Inference + report: copy `src/inference/moe_model/inference.py` chỉnh forward.
- Đo params/FLOPs (`param_flops.py`) — LƯU Ý claim: Soft MoE kích hoạt TẤT CẢ expert mỗi mẫu → active FLOPs cao hơn Cluster-MoE top-2; đây chính là điểm bài muốn nhấn (active computational cost).

### CSV
```
seed,dataset,backbone,model,num_experts,routing_type,
accuracy,macro_f1,weighted_f1,params_m,flops_g,latency_ms
```
→ `mean_acc_mF1_results/soft_moe_baseline.csv`.

## Phương án B — Hạ xuống Related Work (nếu chọn)
- Không code. Tạo ghi chú `soft_moe_decision.md`: "Soft MoE not implemented as baseline; discussed in Related Work. Difference: Soft MoE uses dense soft slot assignment (all experts active per sample), our work studies cluster-prototype top-k routing for classifier-side expert selection with bounded active cost."
- Gỡ Soft MoE khỏi bảng baseline chính trong bản thảo nếu đang có.

## Tiêu chí hoàn thành
- A: 5 seed × SoftMoE có report; bảng so sánh Soft MoE vs learned-gate MoE vs Cluster-MoE cùng backbone pretrained; câu chuyện active-cost rõ ràng.
- B: ghi chú quyết định + bản thảo đã sửa wording.

## Phụ thuộc: cuối cùng (ưu tiên 7/7 theo tài liệu). Thời gian A: 1–2 buổi code + ~half-day GPU.
