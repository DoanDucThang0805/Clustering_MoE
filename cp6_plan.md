# CP6 — Backbone thay thế (MobileNetV2) → `backbone_generalization.csv`

## Mục tiêu
Chứng minh Cluster-MoE không phải artifact của MobileNetV3-Small. Tối thiểu 1 backbone thay thế: **MobileNetV2** (edge baseline gần nhất — tài liệu ưu tiên); EfficientNet-B0 tùy thời gian. Chạy đủ 3 model: dense, learned-gate MoE, Cluster-MoE.

## Hiện trạng repo — các registry phải MỞ RỘNG (thêm entry, không sửa kiến trúc hiện có)
1. `src/models/pretrain_baseline/model_registry.py` — thêm `mobilenetv2_torchvision` (file mới `mobilenetv2.py` copy pattern `mobilenetv3small.py`: torchvision `mobilenet_v2(weights=IMAGENET1K_V1)`, thay classifier cuối cho 8 lớp).
2. `src/models/moe/backbone_registry.py` + `backbone.py` — thêm class `Mobilenetv2BackboneTorchvision(pretrained)`:
   - forward: `features → avgpool/adaptive_avg_pool2d → flatten`; **output_dim = 1280** (khác 576 của V3-Small!).
   - CẢNH BÁO bug cũ: KHÔNG để trailing comma sau `models.mobilenet_v2(...)` (bug từng làm self.model thành tuple).
3. `src/models/clustering_moe/backbone_registry.py` + `backbone.py` — tương tự; đồng thời backbone cần method `load_dense_checkpoint` nếu muốn dùng dense-aligned recipe (copy logic từ class V3-Small hiện có, đổi prefix keys cho khớp state_dict MobileNetV2).
4. `src/embedding/pretrain_backbone/image_embedding.py` — thêm vào `_BACKBONE_MAP` + `choices` của `--model_name`.
5. Type hint `Literal["mobilenetv3small_timm","mobilenetv3small_torchvision"]` ở `ClusteringMoEModel`/`MoEModel` — thêm literal mới (chỉ là type hint, không đổi logic).
6. Expert/gating/centroid dim tự theo `centroids.shape[1]`/`output_dim` → model tự thích ứng 1280, không sửa gì.

## Việc cần làm (pipeline giống seed 47–51, backbone_name=mobilenetv2_torchvision)
Seed: {42,43,44,45,46} (5 seed pilot).
1. Dense: train `training.mobilenetv3small` biến thể... — LƯU Ý: script này hard-code model import; cách nhỏ nhất: thêm arg `--model_name` đọc từ `MODEL_REGISTRY` (đổi 2 dòng import).
2. Learned-gate MoE: `moe_train.sh --backbone_name mobilenetv2_torchvision --type_model moe_temperature_0.5_pretrain_backbone_mnv2`.
3. Cluster-MoE: extract embedding (namespace `pretrain_backbone`, backbone_name mới → thư mục `mobilenetv2_torchvision_backbone` tự tách) → kmeans G4 cosine (centroid 1280-d) → train clustering_moe lr 7e-5 với `--backbone_name mobilenetv2_torchvision --backbone_checkpoint <dense seed đó>`.
4. Inference 3 model → per-seed metrics; đo params/FLOPs bằng `src/benchmark/param_flops.py` cho 3 model MobileNetV2.

## CSV xuất
```
seed,dataset,backbone,initialization,model,routing,G,top_k,tau,
accuracy,macro_f1,weighted_f1,params_m,flops_g,
model_size_mb,peak_cpu_memory_mb,cpu_latency_ms
```
→ `mean_acc_mF1_results/backbone_generalization.csv` (kèm hàng MobileNetV3-Small hiện có để so).

## Tiêu chí hoàn thành
- 3 model × 5 seed MobileNetV2, thứ hạng tương đối Cluster-MoE vs dense/MoE giữ nguyên chiều với V3-Small.
- Smoke test bắt buộc trước khi chạy batch: build backbone pretrained=True, forward (2,3,224,224) → (2,1280); load_dense_checkpoint đếm đủ tensor.

## Rủi ro
- MobileNetV2 classifier structure khác V3 → `load_dense_checkpoint` prefix `features.`/`classifier.` cần kiểm bằng in state_dict keys thật trước khi viết.
- Thời gian GPU ~1 ngày (15 run dense/MoE/ClusterMoE + retrains nếu dùng recipe săn backbone; đề nghị mức tối thiểu: 1 run/seed, không săn).

## Phụ thuộc: nên sau CP4 (giải phóng GPU); code registry có thể viết trước song song CP1.
