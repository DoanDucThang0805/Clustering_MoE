II. Checkpoint 2: Trích xuất feature
Feature extraction phải được thực hiện sau khi có checkpoint ổn định. Với mỗi ảnh xi , backbone
tạo feature vector:
ui = fθ (xi ), ui ∈ Rd .
(1)
Phương trình này xác định không gian đặc trưng dùng cho clustering và routing. Clustering không
được thực hiện trên ảnh thô, mà trên feature vector đã được học bởi backbone.
Chỉ dùng train features để học cluster centers. Validation và test features chỉ dùng để đánh giá
routing behavior và visualization.
Output cần có:
features/
features_train_seed42.npz
features_val_seed42.npz
features_test_seed42.npz
features_train_seed43.npz
...
Mỗi file .npz phải chứa:
features: float array [N, d]
labels: int array [N]
image_paths: string array [N]
split: train/val/test
seed: int
class_names: list[str]
Điều kiện đạt checkpoint:
• Số feature bằng số ảnh trong split tương ứng.
• Dimension d thống nhất giữa train/val/test.
• Không dùng test split để fit clustering.
• Có thể load lại .npz và tái tạo labels đúng thứ tự.