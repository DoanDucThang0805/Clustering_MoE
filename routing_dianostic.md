# Routing Diagnostics Analysis

![Expert Utilization](../reports/moe_routing_diagnostics/moe_baseline_expert_usage.png)

Để đánh giá mức độ hiệu quả của cơ chế định tuyến (routing mechanism) trong mô hình, chúng tôi tiến hành phân tích tỷ lệ kích hoạt của các chuyên gia (experts) trên tập dữ liệu kiểm thử (test set) với cấu hình kích hoạt Top-K = 2. Biểu đồ trên biểu diễn tỷ lệ phần trăm (%) mỗi chuyên gia được lựa chọn dựa trên tổng số lượt phân công định tuyến của toàn bộ hệ thống. 

Về mặt lý thuyết, một hệ thống cân bằng tải (load balancing) lý tưởng với 4 chuyên gia sẽ duy trì tỷ lệ kích hoạt tiệm cận 25% cho mỗi node, nhằm tối ưu hóa không gian tham số và khả năng tính toán song song. Tuy nhiên, kết quả thực nghiệm cho thấy bộ định tuyến Baseline hiện vẫn đang tồn tại sự mất cân bằng tải cục bộ. Cụ thể, Expert 4 đang phải xử lý khối lượng công việc lớn nhất, chiếm khoảng 31% tổng lượt kích hoạt, theo sát bởi Expert 3 (~28%) và Expert 2 (~26%). Đáng chú ý, Expert 1 hoạt động với hiệu suất thấp nhất khi chỉ nhận được xấp xỉ 15% tổng lượt phân công—tức chưa bằng một nửa tải trọng của Expert 4. Mặc dù mạng lưới không gặp phải hiện tượng sụp đổ định tuyến hoàn toàn (routing collapse - nơi mạng chỉ sử dụng 1-2 chuyên gia), sự phân bổ thiên vị này cho thấy cơ chế định tuyến của Baseline chưa thể khai thác đồng đều và triệt để năng lực biểu diễn của tất cả các nhánh chuyên gia.
