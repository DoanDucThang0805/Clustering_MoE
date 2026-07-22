# Trả lời 7 câu hỏi làm rõ phương pháp luận

Tài liệu này trả lời từng câu hỏi dựa trên việc đọc trực tiếp code, checkpoint
metadata, và trạng thái hiện tại của repo (`clustering_moe`) — không suy đoán.
Những chỗ dữ liệu KHÔNG có sẵn được ghi rõ là "chưa có / cần bổ sung", không
bịa số.

---

## 1. Training implementation

Ba nhóm model (dense, learned-gate MoE, Cluster-MoE) dùng chung augmentation
pipeline nhưng khác optimizer/scheduler/batch size. Nguồn: `src/datasets/plantdoc_dataset.py`,
`src/training/mobilenetv3small.py`, `moe_train.sh` → `src/training/moe.py`,
`cluster_moe_train.sh` → `src/training/clustering_moe.py`, đối chiếu với
checkpoint metadata thực tế (`learning_rate`, `weight_decay`, `label_smoothing`,
`val_acc_history`, `lr_history` lưu trong mỗi `best_checkpoint.pth`).

| | Dense | Learned-gate MoE | Cluster-MoE |
|---|---|---|---|
| Input resolution | 224×224 (train: resize 256 → random-crop 224) | như dense | như dense |
| Normalization | ImageNet mean/std: `(0.485,0.456,0.406)` / `(0.229,0.224,0.225)` | như dense | như dense |
| Augmentation (train) | Resize(256)→RandomCrop(224), HorizontalFlip(p=0.5), VerticalFlip(p=0.3), RandomRotate90(p=0.5), ShiftScaleRotate(shift=0.05, scale=0.1, rotate=30°, p=0.5) | như dense | như dense |
| Optimizer | AdamW | AdamW | AdamW |
| Initial LR (peak sau warmup) | 1e-3 | 1e-3 (`moe_train.sh`) | 3e-4 (checkpoint metadata: `learning_rate=0.0003`) |
| Weight decay | 1e-3 | 1e-3 | 1e-2 (checkpoint metadata: `weight_decay=0.01`) |
| Label smoothing | không dùng (CrossEntropyLoss có `class_weight='balanced'`) | không dùng label smoothing (có `moe_alpha=0.05` cho auxiliary load-balance loss) | 0.05 (checkpoint metadata: `label_smoothing=0.05`) |
| LR scheduler | `ReduceLROnPlateau` (mode='max' trên val-acc, patience=10) | `SequentialLR` = `LinearLR` warmup 10 epoch (start_factor 0.1→1.0) + `CosineAnnealingLR` (T_max = num_epochs−10, eta_min=1e-6) | như Learned-gate MoE |
| Batch size | 32 | 64 (`moe_train.sh`) | 32 (`cluster_moe_train.sh`) |
| Số epoch tối đa | 200 | 400 | 400 |
| Early-stopping rule | Dừng nếu val-acc không cải thiện > 1e-5 trong 50 epoch liên tiếp | như dense | như dense |
| Checkpoint-selection metric | Validation accuracy cao nhất (`best_checkpoint.pth` ghi đè mỗi khi val-acc cải thiện) | như dense | như dense |

Nguồn kiểm chứng cụ thể: `src/utils/baseline_trainer.py` (dense), `src/utils/moe_trainer.py`
(Learned-gate MoE), `src/utils/cluster_moe_trainer.py` (Cluster-MoE) — cả 3 dùng
chung logic early-stopping/checkpoint-selection (`val_acc_threshold=1e-5`,
`early_stopping_patience=50`), chỉ khác scheduler.

---

## 2. Prototype timing — encoder có bị đóng băng sau khi fit centroid không?

**KHÔNG bị đóng băng.** Đây là điểm cần làm rõ minh bạch trong manuscript vì
hiện chỉ nêu "verified data-separation rule" (centroid chỉ fit trên train
features) mà chưa nói rõ điều này.

Quy trình thực tế (đọc từ `src/training/clustering_moe.py`,
`src/utils/cluster_moe_trainer.py`, `src/models/clustering_moe/cluster_gating.py`):

1. Backbone dense được train trước (độc lập).
2. Feature embedding được trích xuất **một lần** từ backbone dense đó (snapshot
   cố định tại thời điểm trích xuất).
3. K-means fit **một lần** trên các embedding train này → centroid `μ_g`.
4. Centroid được nạp vào `ClusterPrototypeGating` dưới dạng
   `register_buffer("centroids", ...)` — **non-trainable, không nhận gradient**.
5. Backbone của `ClusteringMoEModel` được khởi tạo lại từ checkpoint dense đó
   (`load_dense_checkpoint`), sau đó **toàn bộ model (bao gồm backbone) tiếp
   tục được huấn luyện** — `optimizer = optim.AdamW(model.parameters(), ...)`
   không loại trừ backbone khỏi optimizer, không có dòng code nào set
   `requires_grad_(False)` cho backbone trong `ClusteringMoEModel`/`ClusterMoETrainer`.

**Hệ quả cần nêu rõ trong bài**: centroid được fit tại một thời điểm cố định
(state của encoder trước khi fine-tune Cluster-MoE), nhưng encoder **tiếp tục
trôi (drift)** trong suốt quá trình huấn luyện Cluster-MoE sau đó. Centroid
**không được refit/refresh** theo encoder đã cập nhật. Đây là một giả định
ngầm của phương pháp (centroid ban đầu đủ tốt để làm điểm khởi đầu routing,
và routing dựa trên khoảng cách cosine có thể chấp nhận được độ trôi nhỏ của
encoder) — nên được phát biểu tường minh là một hạn chế/thiết kế có chủ đích,
không phải lỗi.

---

## 3. PlantVillage — chi tiết dataset

Nguồn: `src/datasets/plantvillage_dataset.py` (chạy trực tiếp để lấy số liệu
thật, không suy đoán) + `src/utils/load_dataset.py` (logic chia split).

**10 lớp (Tomato subset)**, `class_to_idx`:
```
Tomato_Bacterial_spot                          : 0
Tomato_Early_blight                            : 1
Tomato_Late_blight                             : 2
Tomato_Leaf_Mold                               : 3
Tomato_Septoria_leaf_spot                      : 4
Tomato_Spider_mites_Two_spotted_spider_mite    : 5
Tomato__Target_Spot                            : 6
Tomato__Tomato_YellowLeaf__Curl_Virus          : 7
Tomato__Tomato_mosaic_virus                    : 8
Tomato_healthy                                 : 9
```

**Số lượng mẫu theo split** (đếm trực tiếp từ dataset đã load):

| split | tổng | class 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 12808 | 1701 | 800 | 1527 | 762 | 1417 | 1341 | 1123 | 2566 | 298 | 1273 |
| validation | 1601 | 213 | 100 | 191 | 95 | 177 | 167 | 140 | 321 | 38 | 159 |
| test | 1602 | 213 | 100 | 191 | 95 | 177 | 168 | 141 | 321 | 37 | 159 |

**Quy tắc chia split**: stratified 2-bước bằng `sklearn.train_test_split`
(`src/utils/load_dataset.py`): bước 1 tách 80% train / 20% tạm; bước 2 tách
20% tạm đó làm đôi (10%/10%) thành validation/test. `random_state=42` cố định
cho cả 2 bước → **split hoàn toàn độc lập với seed huấn luyện** (đổi seed
model không đổi tập test), stratify theo nhãn lớp để giữ tỉ lệ lớp đồng đều
giữa 3 tập.

---

## 4. EfficientNet-B0 và Soft MoE — cùng training budget/seed list?

**Soft MoE vs Learned-gate MoE (cùng backbone mobilenetv3small_torchvision) —
CÙNG budget, có xác nhận rõ trong code**: docstring của
`src/training/soft_moe.py` ghi trực tiếp *"Training budget giữ Y HỆT
`moe_train.sh` (lr 1e-3, wd 1e-3, 400 epoch, batch 64, τ=0.5, context_dim=6)"*
— đối chiếu default trong `get_args()` của file này khớp 100% với
`moe_train.sh`. Seed list: `SEEDS = [42, 43, 44, 45, 46]` (5 seed) trong
`src/benchmark/cp7_results.py`.

**EfficientNet-B0 (torchvision + timm) — KHÔNG cùng seed list với thí nghiệm
chính.** Kiểm tra trực tiếp checkpoint directory:
`checkpoints/plantdoc/pretrain_baseline/efficientnetb0_torchvision/` và
`efficientnetb0_timm/` chỉ có `seed_42` đến `seed_46` (**5 seed**), trong khi
thí nghiệm chính (mobilenetv3small_torchvision, dùng cho CP1/CP4 paired
statistics) dùng **10 seed (42–51)**. Training budget (epoch cap 400, batch
size, optimizer, scheduler) dùng chung code path (`training/moe.py`,
`training/clustering_moe.py` với `--backbone_name efficientnetb0_*`) nên về
mặt hyperparameter là giống hệt mobilenetv3small — chỉ khác **số seed** (5 so
với 10).

→ **Cần nêu rõ trong Methods**: so sánh backbone-generalization (CP6) và
Soft-MoE baseline (CP7) dùng **n=5 seed**, tách biệt với thí nghiệm chính
n=10 seed dùng cho paired statistical tests. Đây không phải sai sót — chỉ là
phạm vi thí nghiệm nhỏ hơn cho các ablation phụ — nhưng manuscript cần nói rõ
để reviewer không hiểu nhầm là cùng một protocol thống kê.

---

## 5. Raspberry Pi 5 benchmark — chi tiết protocol

Nguồn: `src/benchmark/edge_benchmark.py`, `edge_benchmark_results/edge_benchmark_onnx_results_on_pi.csv`.

**Đã xác nhận được từ code:**
- Framework đo: **ONNX Runtime** (`onnxruntime==1.26.0` trong venv hiện tại;
  không dùng TFLite, không dùng PyTorch trực tiếp trên Pi).
- Execution provider: `CPUExecutionProvider` only (không dùng GPU/NPU).
- Batch size: **1** (cố định, `create_dummy_input(batch_size=1)`).
- Warm-up: **10 lần** chạy trước khi đo (`num_warmup=10`).
- Số lần đo lấy trung bình: **100 lần** (`num_runs=100`), lấy
  `avg_time_ms = tổng_thời_gian / 100`.
- Thread count: **KHÔNG được set tường minh trong code** (không có
  `SetIntraOpNumThreads`/`SetInterOpNumThreads`) → dùng mặc định tự động của
  ONNX Runtime (thường bằng số core vật lý khả dụng trên máy chạy).
- Model size đo bằng `.onnx` + `.onnx.data` (external data) cộng lại.

**CHƯA có trong repo (cần bổ sung thủ công, không thể suy ra từ code):**
- CPU model/clock cụ thể của Raspberry Pi 5 đã dùng (repo chỉ có
  `README.md` ghi yêu cầu phần cứng cho **máy train** — GPU, 8+ core, 16GB+
  RAM — đây KHÔNG phải specs của Pi 5, không nên nhầm lẫn hai thứ).
- Dung lượng RAM thực tế của Pi 5 đã dùng.
- Số thread thực tế mà ONNX Runtime tự động chọn khi chạy (phụ thuộc số core
  của thiết bị, cần đo lại trên máy thật hoặc log lại `session.get_provider_options()`/
  `os.cpu_count()` tại thời điểm benchmark).

→ Đây là chỗ **cần tác giả cung cấp trực tiếp** (thông tin phần cứng vật lý,
không nằm trong code), tài liệu này không thể tự suy ra.

---

## 6. Raw seed-wise CSV cho các bảng 10-seed / entropy / masking / PlantVillage / B0 / Soft MoE

**Hiện trạng repo tại thời điểm viết tài liệu này**: commit `4d72c77 "Curate
final CP1-CP7 paper results"` đã **xoá toàn bộ CSV per-seed thô** khỏi
`paper_results/tables/` và chỉ giữ lại các bảng tổng hợp (mean±std) đã curate:

| Bảng tổng hợp còn lại (đã commit) | Bảng thô (per-seed) tương ứng — hiện KHÔNG còn trong repo |
|---|---|
| `cp1_backbone_init_summary.csv` | `pretrained_backbone_results.csv` (per-seed, 10 seed × 3 model × 2 initialization) |
| `cp2_routing_entropy_temperature_sweep_summary.csv` | `routing_entropy.csv`, `routing_entropy_tau_sweep.csv` |
| `cp3_expert_masking_summary.csv` | `expert_masking_ablation.csv` (per-seed × mask_expert) |
| `cp4_paired_tests_summary.csv` | `seedwise_main_results.csv`, `paired_statistics_extended.csv`, `power_analysis.csv` |
| `cp5_cross_dataset_summary.csv` | `cross_dataset_results.csv` (per-seed, PlantDoc + PlantVillage) |
| `cp6_efficientnetb0_timm_summary.csv` | `backbone_generalization.csv` (per-seed, 3 backbone × 3 model) |
| `cp7_soft_moe_summary.csv` | `soft_moe_baseline.csv` (per-seed) |
| `conditional_confusion_summary.csv` | (bảng này **vẫn còn** — đã là per-seed × primary_expert, không bị xoá) |

**Xác nhận: đúng như reviewer quan sát** — archive hiện tại chỉ có aggregate,
thiếu raw rows.

**Tin tốt: dữ liệu thô KHÔNG bị mất**, có 2 đường khôi phục:
1. **Từ git history** — các file trên vẫn còn nguyên trong commit trước khi
   bị xoá (`a856660`, `d9febef`, `86e4cb9`, v.v.), khôi phục bằng
   `git show <commit>:paper_results/tables/<file>.csv > <file>.csv`.
2. **Tái tạo trực tiếp từ checkpoint** — mọi script sinh ra các bảng này vẫn
   còn nguyên trong `src/` (`src/benchmark/cp5_results.py`,
   `src/benchmark/cp6_results.py`, `src/benchmark/cp7_results.py`,
   `src/diagnostics/expert_masking.py`, `src/diagnostics/plot_tau_curve.py`,
   `src/statistical_test/paired_checkpoint_test.py` mode `cp4`), chạy lại sẽ
   ra đúng số liệu per-seed vì checkpoint gốc (`checkpoints/`) vẫn còn nguyên
   trên máy.

**Việc cần làm**: khôi phục 9 file CSV per-seed ở trên (từ git history hoặc
chạy lại script) và đính kèm cùng bản nộp cho reviewer/kho lưu trữ, để đúng
chuẩn "release raw seed-wise data" — không chỉ báo cáo aggregate.

---

## 7. UMAP và confusion-matrix — có đúng protocol 10-seed ImageNet-pretrained không?

### Confusion-matrix: **ĐÚNG protocol.**
Confusion matrix được sinh tự động bởi pipeline inference chuẩn
(`src/inference/cluster_moe_model/inference.py`, gọi qua `cluster_moe_inference.sh`)
mỗi khi đánh giá một checkpoint. Đã kiểm tra trực tiếp: tồn tại
`confusion_matrix.png` cho đủ cả 10 seed (42–51) dưới đúng namespace của cấu
hình chốt —
`reports/plantdoc/clustering_moe_model/dense_aligned_pretrain_backbone/mobilenetv3small_torchvision/kmeans/G4_cosine_top2/temperature_0.5/seed_{42..51}/run_.../confusion_matrix.png`
— tức cùng checkpoint đã dùng để tính accuracy/macro-F1 trong CP1/CP4. Nếu
manuscript đang dùng confusion matrix từ namespace khác (vd. `non_pretrain_backbone`),
cần đổi sang đúng thư mục trên.

### UMAP: **SAI protocol — cần tái tạo lại.**
File UMAP hiện có duy nhất nằm tại:
```
cluster_analysis/plantdoc/non_pretrain_backbone/mobilenetv3small_torchvision_backbone/
kmeans/cosine/seed_42/umap_cluster_G4.png (+ umap_class_G4.png, và các G khác)
```
Hai vấn đề:
1. **Backbone type = `non_pretrain_backbone`** (từ-đầu/from-scratch), không
   phải `pretrain_backbone` — tức KHÔNG phải cấu hình chính (ImageNet-pretrained)
   đang dùng cho CP1/CP4/kết luận chính của bài.
2. **Chỉ có seed 42**, không đại diện cho 10 seed đã báo cáo.

→ **Cần regenerate UMAP** bằng `src/cluster_analysis/umap_visualization.py`,
trỏ vào embedding/centroid của backbone **pretrained** (namespace
`pretrain_backbone`, `mobilenetv3small_torchvision_backbone`, G=4, cosine),
tại tối thiểu 1 seed đại diện thuộc dải 10 seed đã báo cáo (ví dụ seed 42,
hoặc seed có kết quả gần mean nhất để mang tính đại diện thống kê thay vì
chọn tuỳ ý).

---

## Tóm tắt việc cần làm trước khi phản hồi reviewer

1. Điền bảng hyperparameter ở mục 1 vào Experimental Setup / Appendix.
2. Thêm 1 đoạn Methods/Limitations giải thích rõ encoder KHÔNG đóng băng sau
   khi fit centroid (mục 2).
3. Thêm bảng dataset PlantVillage (mục 3) vào phần Dataset.
4. Ghi rõ n=5 seed cho CP6 (backbone generalization) và CP7 (Soft MoE), tách
   biệt với n=10 seed của thí nghiệm chính (mục 4).
5. Bổ sung thông tin phần cứng Pi 5 thật (CPU/RAM/thread) — không thể tự suy
   ra từ code (mục 5).
6. Khôi phục và đính kèm 9 file CSV per-seed đã bị curate khỏi repo (mục 6).
7. Regenerate UMAP đúng backbone pretrained + seed đại diện của protocol
   chính (mục 7); confusion-matrix hiện đã đúng, không cần đổi.
