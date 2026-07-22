1. Ảnh đầu vào được resize về kích thước 224x224x3 và sau đó được chuẩn hóa theo ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) trước khi huấn luyện mô hình. Các kĩ thuật tăng cường dữ liệu là các kĩ thuật HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate được áp dụng ngẫu nhiên trong suốt quá trình huấn luyện. Các siêu tham số huấn luyện bao gồm learning rate = 0.001; weight decay = 0.001; batch size = 32, số lượng epochs = 400 và early stopping là sau 50 epoch không cải thiện acc; thuật toán tối ưu hóa là AdamW. Kết thúc quá trình huấn luyện best_checkpoint được lưu lại và được tiến hành suy luận để đánh giá hiệu quả dự đoán.


2. Encoder không bị đóng băng sau khi fit centroid. Centroid $\mu_g$ chỉ được tính một lần duy nhất, bằng K-means trên embedding trích xuất từ một checkpoint dense backbone cố định (chỉ dùng tập train), sau đó lưu lại dưới dạng buffer không huấn luyện được (non-trainable). Trong quá trình huấn luyện Cluster-MoE, encoder tiếp tục được cập nhật bằng gradient descent — centroid không được tính lại hay làm mới theo encoder đã cập nhật. Quy tắc tách dữ liệu chỉ fit trên tập train chỉ áp dụng cho bước fit centroid một lần đó; ở cả huấn luyện lẫn đánh giá, cùng một bộ centroid cố định được dùng song song với encoder đã tiếp tục cập nhật (và cuối cùng được fine-tune).


3. PlantVillage dùng subset 10 lớp cà chua (tomato): Tomato_Bacterial_spot, Tomato_Early_blight, Tomato_Late_blight, Tomato_Leaf_Mold, Tomato_Septoria_leaf_spot, Tomato_Spider_mites_Two_spotted_spider_mite, Tomato__Target_Spot, Tomato__Tomato_YellowLeaf__Curl_Virus, Tomato__Tomato_mosaic_virus, Tomato_healthy. Tổng số mẫu: train 12.808 / validation 1.601 / test 1.602 (tỉ lệ ~80/10/10).


4. EfficientNetB0 (dense, moe learned, moe clustering) và Soft MoE đều được huấn luyện trên tập dữ liệu plantdoc với 5 seed khác nhau ( từ 42 - 46) và với cùng các siêu tham số huấn luyện (learning rate, weight decay, batchsize, num of epochs, early stopping, Adam optimization)

5. Đánh giá hiệu năng trên thiết bị biên được thực hiện trên Raspberry Pi 5, sử dụng CPU Broadcom BCM2712 quad-core Arm Cortex-A76 với xung nhịp tối đa 2.4 GHz và 2GB RAM. Quá trình đo được thực hiện hoàn toàn trên CPU thông qua thư viện ONNX Runtime, với kích thước batch cố định bằng 1. Trước khi đo, mô hình được chạy khởi động (warm-up) 10 lần để loại bỏ độ trễ khởi tạo, sau đó thời gian suy luận được đo qua 100 lần lặp và lấy giá trị trung bình làm kết quả báo cáo.


6. Các file csv results được công bố tại https://github.com/DoanDucThang0805/Clustering_MoE/tree/main/paper_results


7. Các hình Umap và ma trận nhầm lẫn được lấy từ kết quả của phần khởi tạo non pretrain Imagenet backbone. Cả pretrain và non pretrain chỉ khác nhau về việc khởi tạo trọng số chứ không khác nhau về giao thức. Cụ thể hình Umap được lấy ra từ kết quả phân cụm và hình ma trận nhầm lẫn được lấy ra khi inference xong trên tập test của tập dữ liệu plantdoc. 