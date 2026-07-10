# CP4 — 10 seed cho 3 model chính + paired statistics + power analysis

## Mục tiêu
`seedwise_main_results.csv`, `paired_statistics_extended.csv`, `power_analysis.csv` cho đúng 3 cấu hình (KHÔNG mở rộng grid):
1. Dense MobileNetV3-Small (pretrained)
2. Learned-gate MoE, E=4, top-k=2 (pretrained)
3. Cluster-MoE cosine, G=4, top-k=2, τ=0.5 (dense-aligned)

## Hiện trạng repo — phần lớn ĐÃ XONG
| Cấu hình | 10 seed 42–51 | Nguồn |
|---|---|---|
| Dense pretrained | ✅ có (champion per-seed) | `checkpoints/plantdoc/pretrain_baseline/.../seed_*/run_*` — LƯU Ý: `mobilenetv3small_torchvision_per_seed.csv` đang trỏ run CŨ cho seed 47–51 (champion đã đổi), phải regenerate |
| Learned-gate MoE pretrained | 🔄 CP1 đang chạy | namespace `moe_temperature_0.5_pretrain_backbone` |
| Cluster-MoE dense-aligned | ✅ xong, mean 87.26%/0.8452 | champion 1 run/seed + `cluster_moe_dense_aligned_..._temp0.5.csv` |
- Đã có sẵn `src/statistical_test/paired_checkpoint_test.py` (xuất CSV + variable description) — đọc kỹ hàm `main(output_csv)` xem nó đang so cặp model nào, sửa input trỏ vào 3 cấu hình trên thay vì viết mới.

## Việc cần làm
1. **Đợi CP1 xong** → chạy inference 10 seed MoE pretrained (`moe_inference.sh --type_model moe_temperature_0.5_pretrain_backbone --seed N --runtime <run>` từng seed; script chọn run mới nhất nếu để trống runtime nhưng nên truyền tường minh).
2. **Regenerate per-seed dense**: chạy lại `pretrain_inference.sh --seed N --run_time <champion>` cho seed nào CSV per-seed đang lệch (47–51); champion dense theo seed:
   - 47: `run_20260703-172432` · 48: `run_20260704-205832` · 49: `run_20260705-132128` · 50: `run_20260705-163207` · 51: `run_20260705-184851` · 42–46: run duy nhất/gốc còn lại trong thư mục.
3. **Build `seedwise_main_results.csv`** — script nhỏ `src/benchmark/build_seedwise_main.py` đọc 3 nguồn trên, schema tài liệu:
   ```
   seed,dataset,initialization,backbone,model,routing,G,top_k,tau,
   accuracy,macro_f1,weighted_f1,params_m,flops_g,latency_ms
   ```
   (params/flops/latency lấy từ `params_flops_results/params_flops.csv` + `edge_benchmark_results/...pi.csv`, cùng giá trị cho mọi seed của 1 model; weighted_f1 phải tính lại lúc inference — thêm vào bước 1–2 nếu report hiện chỉ có macro.)
4. **Paired statistics** (3 cặp so sánh: ClusterMoE−Dense, ClusterMoE−MoE, MoE−Dense; metric: accuracy & macro_f1):
   - mean_diff, std_diff, CI95 (t-dist, df=9), paired t-test p, Wilcoxon signed-rank p, Holm p (m=3), BH p, effect size d_z = mean_diff/std_diff.
   - Dùng/mở rộng `paired_checkpoint_test.py`; xuất `paired_statistics_extended.csv` đúng schema tài liệu trang 5.
5. **Power analysis** — `power_analysis.csv`:
   - n_min = ((z_{1−α/2} + z_{1−β}) / (Δ/σ_d))², α=0.05, power 0.8; Holm: α′=α/3.
   - Cột: comparison,metric,alpha,power_target,pilot_mean_diff,pilot_std_diff,effect_size_dz,required_n_uncorrected,required_n_holm,current_n,decision.
   - `decision`: "sufficient" nếu current_n=10 ≥ required, ngược lại "insufficient — report measured gain under protocol".

## Nguyên tắc phải giữ (tài liệu mục II)
- Cùng seed list {42..51}, cùng split cho cả 3 model.
- Không dùng test để chọn checkpoint trong phân tích này — dùng champion đã khóa, ghi rõ protocol chọn trong caption.

## Tiêu chí hoàn thành
- 30 hàng seedwise (3 model × 10 seed), không hàng nào thiếu metric chính.
- Nếu paired test không significant sau Holm → bài viết dùng wording "measured gain under the reported protocol" (tài liệu mục XIII).

## Phụ thuộc: CP1. Thời gian sau CP1: ~nửa buổi.
