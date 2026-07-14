# Session notes 2026-07-13→14 — Retrain baselines + trạng thái toàn dự án

> File chống mất context. Nếu session/máy chết: đọc file này là tiếp tục được ngay.

## 1. TRẠNG THÁI TOÀN DỰ ÁN (CP1–CP7)

| CP | Nội dung | Deliverable (paper_results/tables/) | Git |
|---|---|---|---|
| CP1 | Pretrained backbone (3 model × 10 seed) | `pretrained_backbone_results.csv`, `cp1_backbone_init_summary.csv` | ✅ committed |
| CP2 | Routing entropy + τ-curve | `routing_entropy*.csv`, `figures/cp2_tau_curve.png` | ✅ |
| CP3 | Expert masking + conditional confusion | `expert_masking_ablation.csv`, `conditional_confusion_summary.csv` | ✅ |
| CP4 | 10-seed paired stats + power | `seedwise_main_results.csv`, `paired_statistics_extended.csv`, `power_analysis.csv` | ✅ |
| CP5 | PlantVillage (dataset độc lập) | `cross_dataset_results.csv` | ✅ |
| CP6 | EfficientNet-B0 tv+timm | `backbone_generalization.csv` | ✅ (commit d9febef) |
| CP7 | Soft MoE baseline (làm lại, pooled) | `soft_moe_baseline.csv` | ✅ (commit 86e4cb9) |

Push cuối: `81b5299` (origin/main). Working tree lúc ghi file này: chỉ còn 5 file code sửa CHƯA commit (mục 4).

## 2. SỐ LIỆU CHÍNH (PlantDoc test, đã chốt)

### CP1 (pretrained, n=10)
- Dense (torchvision): **85.19 ± 1.61** / mF1 82.11
- Learned-gate MoE: **85.79 ± 0.76** / mF1 82.54 (vs dense: +0.60, p=0.36 **ns**)
- **Cluster-MoE: 87.27 ± 0.72** / mF1 84.52 (vs dense: +2.08, **Holm p=0.020 SIG**)

### CP6 (n=5)
| backbone | dense | learned MoE | cluster |
|---|---|---|---|
| MNv3 | 85.19 | 85.79 | **87.27** |
| B0-torchvision | 86.81 | 84.63 (−2.2 vs dense!) | **86.88** |
| B0-timm | 85.89 | 85.05 | **86.25** |

→ Cluster #1 cả 3 backbone; learned MoE TỤT dưới dense trên B0 → luận điểm: learned gating không bền vững, cluster routing là thành phần duy nhất nhất quán.

### CP7 (n=5, best-of-3 theo VAL, seed 42 khóa run gốc)
- soft_moe: **85.05 ± 1.62** ≈ learned-gate 85.61 (p=0.68) nhưng **active params +68%** (5.85M vs 3.48M)
- cluster_moe: 86.88 ± 0.36 — tốt nhất ở active thấp nhất
- Params chuẩn: `params_m`=ACTIVE (thop), `params_total_m`=phụ (đã trừ 1.6158M params chết của classifier ImageNet trong backbone wrapper). Cả 3 model total ≈ 5.85M (capacity-matched).

## 3. QUYẾT ĐỊNH TRUNG THỰC ĐÃ CHỐT (không đảo ngược)
1. **KHÔNG** thay dense torchvision (85.19) bằng lamb1k (82.2) trong CP1 — sandbagging + mâu thuẫn CP4/CP5/CP6 CSV đã commit.
2. **KHÔNG** ghi nhãn chung "mobilenetv3small" cho số lamb1k — misrepresentation.
3. **KHÔNG** under-train dense (100 epoch) — mọi model đều early-stop (dense best epoch 34–91, cap 200 chưa bao giờ chạm). Câu cho bài: *"trained to convergence, identical early stopping (patience 50); no run reached its epoch cap."*
4. **Chọn run theo VALIDATION, tuyệt đối không theo test** (test 285 ảnh, 1 ảnh = 0.35 điểm). Protocol: *"best-of-N restarts selected by validation accuracy"*.
5. Dense ≈ learned MoE **là điểm mạnh** của bài (gain đến từ routing, không phải capacity): MoE relative-error-reduction giảm nửa khi pretrained (7.8%→4.1%) còn cluster ổn định (15.9%→14.0%).

## 4. CODE SỬA TRONG SESSION NÀY (CHƯA COMMIT)
| File | Thay đổi |
|---|---|
| `src/training/mobilenetv3small.py` | +`--model_dir_name` (default canonical) → chuyển hướng namespace checkpoint |
| `src/models/pretrain_baseline/model_registry.py` | +alias `mobilenetv3small_torchvision_retrain2` |
| `src/training/moe.py` | +`--restart_id` (init_seed = seed + 1000*restart_id; không đổi data split) |
| `src/training/efficientnetb0_timm.py` | +`--restart_id` (tương tự) |
| `src/benchmark/cp6_results.py` | `_latest()` → chọn run **best VAL** (`val_acc_history` trong ckpt); seed 1-run giữ hành vi cũ |

→ Sau khi đo xong: commit 5 file này + 2 CSV kết quả mới.

## 5. JOBS ĐANG CHẠY (phóng ~00:00–01:00 ngày 14/7)

### Job A — MNv3 torchvision **retrain2** (10 seed SONG SONG, 10 process)
- Lệnh mỗi seed: `python -m training.mobilenetv3small --seed S --dataset_name plantdoc --model_dir_name mobilenetv3small_torchvision_retrain2`
- Checkpoint: `checkpoints/plantdoc/pretrain_baseline/mobilenetv3small_torchvision_retrain2/seed_{42..51}/run_*`
- **Namespace baseline gốc `mobilenetv3small_torchvision` KHÔNG bị đụng** (đã verify 0 run mới).
- Logs: `$SCRATCH/retrain2_seed{42..51}.log` · Watcher: `bcijh8boh`
- Mục đích: bộ dense độc lập thứ 2, báo cáo riêng nhãn retrain2, KHÔNG tự thay CP1.

### Job B — B0-timm **best-of-3 theo VAL** (2 lane tuần tự song song nhau)
- Lane dense: `training.efficientnetb0_timm --seed S --restart_id R` (R=1,2 × S=42–46)
- Lane MoE: `training.moe --seed S --restart_id R --backbone_name efficientnetb0_timm --type_model moe_temperature_0.5_pretrain_backbone --num_experts 4 --top_k 2 --num_epochs 400 --batch_size 64 --lr 0.001 --weight_decay 0.001 --moe_alpha 0.05 --temperature 0.5 --router_mode context_aware --context_dim 6` (KHÔNG truyền `--pretrain_backbone` — flag là store_false!)
- Checkpoint đổ THÊM run vào namespace B0-timm hiện có → `cp6_results.py` tự chọn best-val.
- Logs: `$SCRATCH/b0timm_dense.log`, `$SCRATCH/b0timm_moe.log` · Watcher: `bi4mojhnt`
- Protocol ĐỐI XỨNG: dense + MoE đều best-of-3; cluster giữ 1 run (conservative). **MỘT vòng duy nhất** — kết quả val chọn ra sao ghi vậy, kể cả MoE vẫn ≤ dense.

`$SCRATCH = /tmp/claude-1000/-media-data-minhht-clustering-moe/2876f2e7-1045-4b59-88e9-39bc2b41f77c/scratchpad`
⚠️ Scratchpad có thể bị wipe theo session — nếu mất log, kiểm tra trực tiếp bằng đếm checkpoint (mục 6).

## 6. VIỆC TỰ ĐỘNG LÀM KHI JOB XONG (hoặc làm tay sáng mai nếu watcher chết)

### Khi Job A xong (đủ 10 best_checkpoint trong namespace retrain2):
```bash
cd /media/data/minhht/clustering_moe && source venv/bin/activate && cd src
python -m benchmark.get_acc_mF1_pretrain_baseline \
    --dataset_name plantdoc --type_model pretrain_baseline \
    --model_name mobilenetv3small_torchvision_retrain2 \
    --csv_store_dir ../mean_acc_mF1_results/pretrain_baseline \
    --csv_filename mobilenetv3small_torchvision_retrain2.csv --export_csv
```
→ Báo bảng per-seed + mean±std cạnh bộ gốc (85.19 ± 1.61).

### Khi Job B xong (mỗi seed B0-timm dense & moe có 3 run):
```bash
cd src && python -m benchmark.cp6_results   # đã sửa: tự chọn best-of-3 theo VAL
```
→ Ghi đè `paper_results/tables/backbone_generalization.csv` → báo bảng B0-timm mới (trung thực).

### Kiểm tra tiến độ khi không còn watcher:
```bash
ps aux | grep -E 'training\.' | grep -v grep | wc -l          # số process còn chạy
find checkpoints/plantdoc/pretrain_baseline/mobilenetv3small_torchvision_retrain2 -name best_checkpoint.pth | wc -l   # cần 10
for s in 42 43 44 45 46; do ls -d checkpoints/plantdoc/pretrain_baseline/efficientnetb0_timm/seed_$s/run_* | wc -l; done   # cần 3/seed
for s in 42 43 44 45 46; do ls -d checkpoints/plantdoc/moe_temperature_0.5_pretrain_backbone/efficientnetb0_timm_moe/4_experts/top_2/seed_$s/run_* | wc -l; done   # cần 3/seed
```
Job dở dang → chạy lại đúng lệnh ở mục 5 cho phần thiếu (restart_id/seed nào thiếu run thì chạy cái đó; run dở không có best_checkpoint hoàn chỉnh thì xóa run dir đó trước).

## 7. VIỆC TREO / BƯỚC TIẾP
- [ ] Đo Job A + Job B (mục 6) → báo user → **quyết định dùng số nào cho bài** (retrain2 chỉ thay CP1 nếu nhất quán hóa được TOÀN BỘ CSV liên quan: CP1/CP4/CP5/CP6 + stats).
- [ ] Commit 5 file code (mục 4) + CSV kết quả mới sau khi đo.
- [ ] Đề nghị còn mở: đo routing entropy trên B0 MoE (cơ chế giải thích learned-gate tụt trên B0 — cần param hóa path trong `routing_entropy.py` đang hardcode MNv3).
- [ ] CP6 params cột đang TRỘN chuẩn (MNv3=active 3.48, B0=total 16.2/14.9) — cần thống nhất theo chuẩn CP7 (active + total 2 cột) trước khi đưa bảng vào bài. User đã quyết "để riêng", chưa làm.
- [ ] Ghi chú viết bài: các câu chốt nằm trong lịch sử chat + mục 3 file này.

## 8. BỐI CẢNH KỸ THUẬT NHANH (cho session mới)
- Champion Cluster-MoE: `checkpoints/plantdoc/clustering_moe/dense_aligned_pretrain_backbone/mobilenetv3small_torchvision_backbone/kmeans/temperature_0.5/G4_cosine_top2/seed_*/`
- Soft MoE mới: `checkpoints/plantdoc/soft_moe/mobilenetv3small_torchvision_softmoe/4_experts/seed_*` (seed 42 khóa run `run_20260713-152737`, 43–46 best-of-3).
- Mọi training deterministic theo seed; restart dùng `--restart_id` (init_seed = seed + 1000·R).
- Data split cố định `random_state=42` trong `LoadDataset` — không phụ thuộc seed CLI.
- `--pretrain_backbone` trong `training/moe.py` là **store_false** (mặc định pretrained=True; truyền flag = TẮT pretrained).
- Test PlantDoc = 285 ảnh (1 ảnh = 0.35 điểm acc).
