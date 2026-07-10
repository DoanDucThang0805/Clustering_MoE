# CP5 — Dataset độc lập → `cross_dataset_results.csv`

## Mục tiêu
Chạy 3 model chính (dense / learned-gate MoE / Cluster-MoE cosine G4 k2 τ0.5) trên ≥1 dataset ngoài PlantDoc-tomato để kiểm tra tổng quát. Ưu tiên tài liệu: **PlantVillage tomato subset** (1) > Cassava (2) > CIFAR (chỉ sanity).

## Hiện trạng repo — các điểm code PHỤ THUỘC CỨNG vào PlantDoc (phải xử lý, không đổi kiến trúc)
1. `src/datasets/plantdoc_dataset.py` hard-code `data/tomato-plantdoc-mod` và bị import cứng trong:
   - `src/training/clustering_moe.py` (dòng ~99: `from datasets.plantdoc_dataset import train_dataset, validation_dataset`)
   - `src/training/mobilenetv3small.py` (tương tự)
   - `src/datasets/plantdoc_dataset_moe.py` (build_datasets cho MoE)
   - các inference/embedding module
2. `LoadDataset(root_dir, split, train_ratio=0.8, transform)` — generic theo folder class; PlantVillage tomato cùng cấu trúc folder-per-class là dùng lại được NGUYÊN VẸN.
3. Số lớp suy ra runtime từ labels → không phải sửa model (PlantVillage tomato có 10 lớp thay vì 8 — classifier tự ra 10).

## Việc cần làm
### B1. Chuẩn bị dữ liệu (~30 phút)
- Tải PlantVillage tomato (Kaggle), đặt `data/plantvillage-tomato/` cấu trúc folder-per-class y hệt `tomato-plantdoc-mod`.
- Kiểm tra phân bố lớp, kích thước ảnh; log num_train/num_val/num_test (cần cho CSV).

### B2. Tối thiểu hóa sửa code (chỉ chỗ trỏ dataset, KHÔNG đụng model)
- Tạo `src/datasets/plantvillage_dataset.py` và `plantvillage_dataset_moe.py`: copy nguyên 2 file plantdoc tương ứng, đổi MỖI `cropped_data_path`. (Augmentation giữ nguyên để so sánh công bằng.)
- Trong 3 file training + embedding + inference: đổi import cứng thành chọn theo `--dataset_name` (if/else 2 nhánh — thay đổi nhỏ nhất có thể, các arg `--dataset_name` đã tồn tại sẵn trong CLI và namespace path).

### B3. Pipeline cho mỗi seed (dùng seed {42,43,44,45,46} — 5 seed pilot là đủ theo tài liệu)
Theo đúng nguyên tắc mục II tài liệu — đổi dataset ⇒ TRAIN LẠI baseline, TRÍCH LẠI feature, FIT LẠI centroid (không tái dùng bất kỳ centroid/checkpoint PlantDoc nào):
1. Dense pretrained: `python -m training.mobilenetv3small --seed N` (nhánh plantvillage) → inference.
2. Learned-gate MoE pretrained: `moe_train.sh --seed N` (type_model thêm hậu tố dataset, vd `moe_temperature_0.5_pretrain_backbone_plantvillage`).
3. Cluster-MoE: extract embedding từ checkpoint dense (namespace `pretrain_backbone`, dataset_name plantvillage) → kmeans G4 cosine seed N → train `clustering_moe` lr **7e-5**, wd 1e-2, ls 0.05 (recipe đã chốt) → inference.
- Centroid fit CHỈ trên train features (script kmean hiện tại đã đúng như vậy — xác nhận lại trước khi chạy).

### B4. Xuất CSV
```
seed,dataset,num_classes,num_train,num_val,num_test,
backbone,initialization,model,routing,G,top_k,tau,
accuracy,macro_f1,weighted_f1,params_m,flops_g,latency_ms
```
→ `mean_acc_mF1_results/cross_dataset_results.csv` (gộp cả hàng PlantDoc tương ứng để bảng trong bài so sánh 2 dataset cạnh nhau).

## Tiêu chí hoàn thành
- 3 model × 5 seed × ≥1 dataset mới, không hàng thiếu.
- PlantVillage sạch hơn PlantDoc → acc kỳ vọng CAO hơn hẳn (90%+); nếu thấp bất thường → nghi lỗi split/leak, dừng kiểm tra.
- Claim: Cluster-MoE giữ thứ hạng tương đối so với dense/MoE trên dataset mới.

## Rủi ro
- PlantVillage rất dễ (ảnh lab) → cả 3 model bão hòa, khó phân biệt — nếu vậy cân nhắc thêm Cassava làm dataset thứ 3 (nặng hơn nhiều, để sau).
- Thời gian GPU: ~3 model × 5 seed, dataset lớn hơn PlantDoc (~10-18k ảnh tomato PV) → ước 1–2 ngày GPU. Chạy tuần tự nền như quy trình seed 47–51.

## Phụ thuộc: không phụ thuộc CP khác (nhưng nên sau CP4 để ưu tiên GPU).
