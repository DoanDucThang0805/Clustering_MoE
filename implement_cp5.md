# CP5 — Dataset độc lập (PlantVillage tomato): những thay đổi đã triển khai

Tài liệu ghi lại phần **code + hạ tầng** đã sửa để chạy CP5 (mục VIII PDF "Bổ sung
mô phỏng"): so sánh 3 model chính trên **PlantVillage tomato** (dataset độc lập,
ngoài PlantDoc) → `cross_dataset_results.csv`.

## Quyết định thiết kế (user chốt)
- **Dense baseline (báo cáo)** dùng backbone **lamb1k** (`mobilenetv3small_timm_lamb1k`,
  timm `mobilenetv3_small_100.lamb_in1k`) — mạnh hơn trên PV.
- **Learned-gate MoE & Cluster-MoE** dùng **torchvision** (BACKBONE_REGISTRY của
  MoE/Cluster **chỉ có** `torchvision`/`timm`, KHÔNG có lamb1k).
- **Cluster-MoE dense-aligned**: vẫn train một **dense torchvision** riêng trên PV
  (train → extract embedding → kmeans → cluster_moe backbone_checkpoint), đúng recipe
  champion PlantDoc CP1. Dense torchvision này chỉ phục vụ nhánh cluster, không phải
  baseline báo cáo.
- Recipe Cluster-MoE: **lr 3e-4**, wd 1e-2, ls 0.05 (champion). **5 seed 42–46** (pilot).

## Dữ liệu
`data/tomato-plantvillage-mod/` — 10 lớp, 16,011 ảnh, folder-per-class `Tomato*`.
`LoadDataset` stratified 80/10/10 (random_state=42) → **train 12808 / val 1601 /
test 1602** (đã verify). num_classes suy runtime = 10.

## Thay đổi code

### 1. Module dataset PlantVillage mới
- `src/datasets/plantvillage_dataset.py` — copy `plantdoc_dataset.py`, đổi
  `cropped_data_path → data/tomato-plantvillage-mod`. Augmentation giữ nguyên.
- `src/datasets/plantvillage_dataset_moe.py` — copy `plantdoc_dataset_moe.py`, đổi path.
  (Context extractor `extract_context_features` dataset-agnostic → dùng lại.)

### 2. Registry chọn dataset — `src/datasets/registry.py` (MỚI)
Gom if/else theo `dataset_name` một chỗ, import lazy:
- `get_train_val(dataset_name)` → `(train_dataset, validation_dataset)`
- `get_test(dataset_name)` → `test_dataset`
- `get_moe_build(dataset_name)` → hàm `build_datasets`
Map: `plantdoc→plantdoc_dataset[_moe]`, `plantvillage→plantvillage_dataset[_moe]`.

### 3. Training scripts — chọn dataset + rebuild head + namespace theo `--dataset_name`
| File | Thay đổi |
|---|---|
| `src/training/mobilenetv3small.py` (dense **torchvision**, cho cluster align) | +arg `--dataset_name` (default plantdoc); import dataset qua `registry.get_train_val` **sau** parse args; **rebuild head** `model.classifier[-1]=nn.Linear(in,num_classes)`; checkpoint namespace `plantdoc`→`args.dataset_name`, model dir về canonical `mobilenetv3small_torchvision` (bỏ hậu tố `_retrain1`) |
| `src/training/mobilenetv3smallv2.py` (dense **lamb1k**, baseline báo cáo) | tương tự; head qua **`model.reset_classifier(num_classes)`** (timm); model dir canonical `mobilenetv3small_timm_lamb1k` (bỏ `_retrain2`) |
| `src/training/moe.py` (learned-gate MoE) | import `registry.get_moe_build`; +arg `--dataset_name`; namespace `plantdoc`→`args.dataset_name`; num_classes đã tự suy |
| `src/training/clustering_moe.py` (Cluster-MoE) | import dòng 99 → `registry.get_train_val(args.dataset_name)`; đã có `--dataset_name` cho namespace/centroid, num_classes tự suy |

**Lưu ý:** model dir dense đưa về tên canonical (bỏ `_retrainN`) để khớp default
downstream (`pretrain_inference.sh`, `extract_embedding.sh` dùng
`mobilenetv3small_torchvision`). Với `--dataset_name plantdoc` (mặc định) hành vi
PlantDoc không đổi ngoài việc dense ghi về tên canonical thay vì `_retrainN`.

### 4. Shell wrapper — `moe_train.sh`
+ biến `DATASET_NAME="plantdoc"` + arg `--dataset_name` + truyền `--dataset_name`
vào `python -m training.moe`. Các script `extract_embedding.sh`, `train_kmeans.sh`,
`cluster_moe_train.sh`, `pretrain_inference.sh`, `moe_inference.sh` **đã** nhận
`--dataset_name` từ trước → chỉ cần truyền `plantvillage`.

## Kiểm tra đã chạy (đều PASS)
- `plantvillage_dataset` import: 10 lớp, 12808/1601/1602 = 16011, stratified.
- `registry`: plantdoc→8 lớp, plantvillage→10 lớp; `get_moe_build(plantvillage)`
  trỏ đúng `plantvillage_dataset_moe`.
- Head rebuild: torchvision `classifier[-1]` và lamb1k `reset_classifier(10)` đều
  cho output `(B, 10)`.
- Không còn import dataset plantdoc cứng / namespace hardcode trong 4 script training
  (chỉ còn default arg `plantdoc`).

## Pipeline chạy mỗi seed N ∈ {42,43,44,45,46} (nền, tuần tự)
1. **Dense lamb1k (baseline)**: `python -m training.mobilenetv3smallv2 --seed N --dataset_name plantvillage`
2. **Dense torchvision (cho cluster align)**: `python -m training.mobilenetv3small --seed N --dataset_name plantvillage`
3. **Learned-gate MoE**: `bash moe_train.sh --seed N --dataset_name plantvillage --type_model moe_temperature_0.5_pretrain_backbone`
4. **Cluster-MoE** (dùng dense torchvision seed N):
   - `bash extract_embedding.sh --dataset_name plantvillage --type_backbone pretrain_backbone --seed N --run_time <dense-tv run>` (train/validation/test)
   - `bash train_kmeans.sh --dataset_name plantvillage` (backbone_type pretrain_backbone, seed N, G4 cosine)
   - `python -m training.clustering_moe --seed N --dataset_name plantvillage --num_experts 4 --top_k 2 --distance_metric cosine --temperature 0.5 --lr 3e-4 --weight_decay 1e-2 --label_smoothing 0.05 --backbone_type dense_aligned_pretrain_backbone --centroid_backbone_type pretrain_backbone --backbone_checkpoint <dense-tv champion seed N> --pretrain_backbone`
5. Thu metric: `python -m benchmark.cp5_results` → `cross_dataset_results.csv`.

## Output CSV (PDF VIII.C)
`paper_results/tables/cross_dataset_results.csv`:
```
seed,dataset,num_classes,num_train,num_val,num_test,
backbone,initialization,model,routing,G,top_k,tau,
accuracy,macro_f1,weighted_f1,params_m,flops_g,latency_ms
```
Gồm hàng PlantVillage (num_classes=10) + hàng PlantDoc (num_classes=8, từ CP1) để
so sánh 2 dataset cạnh nhau. dense=lamb1k, moe/cluster=torchvision.

## Ràng buộc
- Checkpoint PV nằm dưới `checkpoints/plantvillage/...` — KHÔNG đụng PlantDoc.
- Centroid fit CHỈ trên train embeddings (nguyên tắc II PDF).
- Checkpoint `.pth` không lên git; commit: 2 module dataset + registry + 4 training
  script sửa + `moe_train.sh` + CSV + `cp5_results.py`.
