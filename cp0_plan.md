# CP0 — Khóa kết quả pilot hiện có → `pilot_summary.csv`

## Mục tiêu
Đóng băng toàn bộ kết quả pilot đã có (dense, learned-gate MoE, Cluster-MoE cosine/Euclidean, grid G/k, temperature sweep, Pi 5 runtime) vào MỘT file `pilot_summary.csv` làm mốc đối chiếu, trước khi các CP sau ghi đè/cập nhật bài báo.

## Hiện trạng repo (đã có sẵn — KHÔNG cần train gì)
| Mảnh pilot | Nguồn dữ liệu có sẵn |
|---|---|
| Dense non-pretrain | `mean_acc_mF1_results/non_pretrain_baseline/mobilenetv3small_torchvision.csv` (+ timm) |
| Dense pretrained | `mean_acc_mF1_results/pretrain_baseline/mobilenetv3small_torchvision.csv` + `_per_seed.csv` |
| Learned-gate MoE (non-pretrain) | `mean_acc_mF1_results/moe/linear_moe.csv` (G4/top2: 78.39%/0.7498) |
| Cluster-MoE grid G,k (non-pretrain, temp 0.5) | `mean_acc_mF1_results/cluster_moe/cluster_moe_non_pretrain_backbone_..._temp0.5.csv` (full grid G∈{2,3,4,5,6,8} × cosine/euclidean × top-k) |
| Temperature sweep | các file `..._temp0.3.csv`, `_temp0.7.csv`, `_temp1.0.csv` cùng thư mục |
| Pi 5 runtime | `edge_benchmark_results/edge_benchmark_onnx_results_on_pi.csv` |
| Params/FLOPs | `params_flops_results/params_flops.csv` |

## Việc cần làm
1. Viết script gộp `src/benchmark/build_pilot_summary.py` (~100 dòng, chỉ đọc CSV, KHÔNG chạy model):
   - Đọc từng CSV nguồn ở trên, chuẩn hóa về schema chung:
     `config_group,model,routing,backbone,initialization,G,top_k,tau,metric,accuracy_mean,accuracy_std,macro_f1_mean,macro_f1_std,params_m,flops_g,pi5_latency_ms,source_file`
   - Với hàng thiếu trường (vd temperature sweep không có latency) → để trống.
   - Join params/FLOPs và Pi-5 latency theo tên model nếu khớp được, không khớp thì để trống (đừng đoán).
2. Xuất `mean_acc_mF1_results/pilot_summary.csv`.
3. Kiểm tra chéo tay 3 giá trị bất kỳ với file nguồn (vd G4 cosine top2 temp0.5 non-pretrain = 0.8028/0.7669).

## Tiêu chí hoàn thành
- `pilot_summary.csv` tồn tại, mỗi hàng truy được về file nguồn qua cột `source_file`.
- Không sửa/ghi đè bất kỳ CSV nguồn nào.

## Lưu ý
- KHÔNG đưa kết quả dense_aligned (mới, 87.26%) vào CP0 — CP0 chỉ khóa pilot cũ. Kết quả mới thuộc CP1/CP4.
- Thời gian: ~30 phút (thuần xử lý CSV).
