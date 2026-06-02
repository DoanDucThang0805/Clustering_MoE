## Cluster-to-Class Heatmap Analysis

### Overview

K-Means clustering was performed on the MobileNetV3 feature embeddings using different numbers of clusters:

```text
G ∈ {2, 3, 4, 5, 6, 8}
```

The resulting cluster-to-class matrices reveal that the feature space is already strongly class-separable. As the number of clusters increases, K-Means tends to create highly class-specific clusters rather than discovering balanced semantic groups.

---

### G = 2

The clustering result is highly imbalanced.

| Cluster | Dominant Class | Purity |
|----------|----------|----------|
| Cluster 0 | Class 6 | 99.1% |
| Cluster 1 | Mixed classes | - |

Cluster 0 almost exclusively contains samples from Class 6, while Cluster 1 contains nearly all remaining classes.

This indicates that Class 6 forms a highly compact and separable region in the embedding space.

---

### G = 3

Two highly pure clusters emerge:

| Cluster | Dominant Class | Purity |
|----------|----------|----------|
| Cluster 0 | Class 1 | 89.3% |
| Cluster 2 | Class 6 | 99.6% |

The remaining cluster contains a mixture of all other classes.

This suggests that Classes 1 and 6 are significantly easier to separate than the rest.

---

### G = 4

The clustering becomes even more class-oriented:

| Cluster | Dominant Class | Purity |
|----------|----------|----------|
| Cluster 0 | Class 6 | 99.8% |
| Cluster 1 | Class 2 | 96.3% |
| Cluster 2 | Class 1 | 94.7% |
| Cluster 3 | Mixed | - |

Three clusters become almost class-exclusive, while one cluster absorbs all remaining classes.

---

### G = 5

The same trend continues:

| Cluster | Dominant Class | Purity |
|----------|----------|----------|
| Cluster 0 | Class 2 | 97.8% |
| Cluster 3 | Class 6 | 100.0% |
| Cluster 4 | Class 1 | 96.7% |

Only two clusters are responsible for representing all remaining classes.

Although the cluster-size distribution is relatively balanced, the semantic distribution is highly uneven.

---

### G = 6

Most clusters become nearly equivalent to individual classes.

| Cluster | Dominant Class | Purity |
|----------|----------|----------|
| Cluster 0 | Class 6 | 100.0% |
| Cluster 2 | Class 0 | 88.6% |
| Cluster 3 | Class 2 | 98.4% |
| Cluster 4 | Class 5 | 84.9% |
| Cluster 5 | Class 1 | 98.1% |

At this stage, K-Means is effectively performing an unsupervised reconstruction of class labels.

---

### G = 8

The strongest class separation is observed when the number of clusters equals the number of classes.

| Cluster | Dominant Class | Purity |
|----------|----------|----------|
| Cluster 1 | Class 3 | 95.9% |
| Cluster 3 | Class 2 | 97.5% |
| Cluster 4 | Class 1 | 99.4% |
| Cluster 5 | Class 6 | 100.0% |
| Cluster 6 | Class 4 | 98.3% |
| Cluster 7 | Class 0 | 90.6% |

The clustering result closely resembles the ground-truth class partitioning.

This suggests that the extracted embeddings already contain strong class-discriminative information.

---

## Key Findings

### Strong Class Separability

The MobileNetV3 embedding space exhibits strong class-level separation.

Several classes consistently form highly pure clusters across different values of G, particularly:

- Class 1
- Class 2
- Class 6

These classes appear to occupy compact and isolated regions in feature space.

### Cluster Purity Increases with G

As the number of clusters increases:

- Cluster purity increases significantly.
- Many clusters become nearly class-exclusive.
- K-Means increasingly behaves like an unsupervised class classifier.

### Implications for Mixture of Experts

The clustering results suggest that the routing mechanism may learn class-specific experts rather than feature-specialized experts.

Instead of routing based on semantic disease characteristics, the experts may naturally specialize in particular classes:

```text
Expert 1 → Class 1
Expert 2 → Class 2
Expert 3 → Class 6
...
```

This behavior differs from the original goal of discovering shared disease characteristics or visual patterns across classes.

---

## Conclusion

The cluster-to-class heatmaps demonstrate that MobileNetV3 features are highly discriminative and already contain strong class-level structure.

K-Means successfully identifies compact regions in the embedding space; however, the discovered clusters are largely aligned with class labels rather than broader semantic groupings.

Further analysis using UMAP and t-SNE visualization is required to determine whether the observed clusters correspond to meaningful feature-space structures or simply reflect class separation.



# Phân tích Heatmap Cluster–Class

## Tổng quan

Thực nghiệm phân cụm K-Means được thực hiện trên các vector đặc trưng (feature embeddings) được trích xuất từ MobileNetV3 với số lượng cụm:

```text
G ∈ {2, 3, 4, 5, 6, 8}
```

Kết quả heatmap Cluster-to-Class cho thấy không gian đặc trưng đã có khả năng phân tách lớp rất tốt. Khi tăng số lượng cụm, K-Means có xu hướng tạo ra các cụm gần như tương ứng trực tiếp với từng lớp thay vì phát hiện các nhóm ngữ nghĩa cân bằng hơn trong không gian đặc trưng.

---

## Kết quả với G = 2

Kết quả phân cụm ở mức G = 2 cho thấy sự mất cân bằng đáng kể.

| Cluster | Lớp chiếm ưu thế | Độ thuần khiết |
|----------|----------|----------|
| Cluster 0 | Class 6 | 99.1% |
| Cluster 1 | Hỗn hợp nhiều lớp | - |

Cluster 0 gần như chỉ chứa các mẫu thuộc Class 6, trong khi Cluster 1 chứa hầu hết các mẫu còn lại thuộc nhiều lớp khác nhau.

Điều này cho thấy Class 6 tạo thành một vùng đặc trưng rất cô đặc và dễ tách biệt trong không gian embedding.

---

## Kết quả với G = 3

Khi tăng lên 3 cụm, xuất hiện thêm một cụm có độ thuần khiết cao:

| Cluster | Lớp chiếm ưu thế | Độ thuần khiết |
|----------|----------|----------|
| Cluster 0 | Class 1 | 89.3% |
| Cluster 2 | Class 6 | 99.6% |

Cluster 1 vẫn là cụm hỗn hợp bao gồm phần lớn các lớp còn lại.

Kết quả này cho thấy Class 1 và Class 6 có đặc trưng riêng biệt hơn so với các lớp khác.

---

## Kết quả với G = 4

Ở mức G = 4, K-Means bắt đầu hình thành các cụm gần như tương ứng với từng lớp:

| Cluster | Lớp chiếm ưu thế | Độ thuần khiết |
|----------|----------|----------|
| Cluster 0 | Class 6 | 99.8% |
| Cluster 1 | Class 2 | 96.3% |
| Cluster 2 | Class 1 | 94.7% |
| Cluster 3 | Hỗn hợp | - |

Ba cụm đầu gần như đã trở thành các cụm chuyên biệt cho từng lớp riêng biệt.

---

## Kết quả với G = 5

Xu hướng phân tách theo lớp tiếp tục được thể hiện rõ:

| Cluster | Lớp chiếm ưu thế | Độ thuần khiết |
|----------|----------|----------|
| Cluster 0 | Class 2 | 97.8% |
| Cluster 3 | Class 6 | 100.0% |
| Cluster 4 | Class 1 | 96.7% |

Hai cụm còn lại phải gánh phần lớn các lớp còn lại trong tập dữ liệu.

Mặc dù phân bố số lượng mẫu giữa các cụm tương đối cân bằng, nhưng về mặt ngữ nghĩa các cụm vẫn bị chi phối mạnh bởi một số lớp cụ thể.

---

## Kết quả với G = 6

Ở mức G = 6, hầu hết các cụm đã trở thành các cụm gần tương ứng với từng lớp:

| Cluster | Lớp chiếm ưu thế | Độ thuần khiết |
|----------|----------|----------|
| Cluster 0 | Class 6 | 100.0% |
| Cluster 2 | Class 0 | 88.6% |
| Cluster 3 | Class 2 | 98.4% |
| Cluster 4 | Class 5 | 84.9% |
| Cluster 5 | Class 1 | 98.1% |

Điều này cho thấy K-Means đang dần tái tạo lại cấu trúc nhãn lớp chỉ dựa trên thông tin embedding.

---

## Kết quả với G = 8

Khi số cụm bằng số lớp, hiện tượng này trở nên rõ ràng nhất.

| Cluster | Lớp chiếm ưu thế | Độ thuần khiết |
|----------|----------|----------|
| Cluster 1 | Class 3 | 95.9% |
| Cluster 3 | Class 2 | 97.5% |
| Cluster 4 | Class 1 | 99.4% |
| Cluster 5 | Class 6 | 100.0% |
| Cluster 6 | Class 4 | 98.3% |
| Cluster 7 | Class 0 | 90.6% |

Kết quả phân cụm lúc này gần như tương đương với việc khôi phục lại nhãn lớp ban đầu bằng phương pháp không giám sát.

---

# Nhận xét chính

## Khả năng phân tách lớp mạnh

Các vector đặc trưng được trích xuất từ MobileNetV3 đã mang thông tin phân biệt lớp rất rõ ràng.

Một số lớp luôn tạo thành các cụm có độ thuần khiết rất cao ở hầu hết các giá trị G, đặc biệt là:

- Class 1
- Class 2
- Class 6

Điều này cho thấy các lớp này chiếm những vùng tương đối độc lập trong không gian đặc trưng.

## Độ thuần khiết của cụm tăng khi tăng G

Khi số lượng cụm tăng:

- Độ thuần khiết của các cụm tăng lên rõ rệt.
- Nhiều cụm trở nên gần như chỉ chứa một lớp duy nhất.
- K-Means hoạt động gần giống như một bộ phân loại không giám sát.

## Ý nghĩa đối với Mixture of Experts

Kết quả này cho thấy cơ chế routing dựa trên K-Means có xu hướng tạo ra các expert chuyên biệt theo lớp thay vì chuyên biệt theo đặc trưng ngữ nghĩa.

Thay vì:

```text
Expert 1 → bệnh nấm
Expert 2 → bệnh vi khuẩn
Expert 3 → bệnh sinh lý
```

mô hình có xu hướng học:

```text
Expert 1 → Class 1
Expert 2 → Class 2
Expert 3 → Class 6
...
```

Do đó, các expert được hình thành chủ yếu dựa trên ranh giới lớp đã có sẵn trong embedding space.

---

# Kết luận

Kết quả heatmap Cluster-to-Class cho thấy MobileNetV3 đã học được không gian đặc trưng có khả năng phân biệt lớp rất mạnh. K-Means có thể xác định các cụm ổn định và có độ thuần khiết cao, tuy nhiên phần lớn các cụm thu được tương ứng trực tiếp với các lớp thay vì phản ánh những cấu trúc ngữ nghĩa rộng hơn trong dữ liệu.

Do đó, cần tiếp tục thực hiện phân tích bằng UMAP hoặc t-SNE để đánh giá liệu các cụm này thực sự phản ánh cấu trúc của không gian đặc trưng hay chỉ đơn thuần là sự tách biệt giữa các lớp.