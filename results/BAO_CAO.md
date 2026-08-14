# Báo cáo Lab 16 — Cloud AI Environment Setup (AWS)

**Sinh viên:** Nguyễn Tuấn Khanh — 2A202601139
**Ngày thực hiện:** 14/08/2026
**Hạ tầng:** AWS `us-east-1` — VPC riêng, Bastion Host (`t3.micro`), Compute Node (`t3.medium`, 2 vCPU / 4 GB RAM), NAT Gateway, ALB
**Bài toán:** Credit Card Fraud Detection (284,807 giao dịch, 30 features, 492 ca gian lận = 0.173%)
**Model:** LightGBM 4.7.0 (`binary`, `scale_pos_weight` ≈ 578, early stopping 50 vòng)

---

## 1. Bảng kết quả

| Metric | Kết quả |
|---|---|
| Thời gian load data | 2.316 s |
| Thời gian training | 3.040 s |
| Best iteration | 41 |
| AUC-ROC | 0.886823 |
| Accuracy | 0.956497 |
| F1-Score | 0.061364 |
| Precision | 0.031865 |
| Recall | 0.826531 |
| Inference latency (1 row) | 0.449 ms (p50 0.438 ms — p95 0.465 ms) |
| Inference throughput (1000 rows) | 1.833 ms → **545,682 rows/s** |

Confusion matrix (tập test 56,962 dòng): TN = 54,403 — FP = 2,461 — FN = 17 — TP = 81

---

## 2. Nhận xét

**Về training time.** Huấn luyện 182,276 dòng chỉ mất **3.04 giây** trên 2 vCPU, dừng sớm ở vòng lặp thứ 41. Đây là minh chứng rõ cho việc bài toán dữ liệu bảng (tabular) không cần GPU: LightGBM dùng thuật toán histogram-based nên chi phí chủ yếu là quét bộ nhớ tuần tự, thứ mà CPU xử lý rất hiệu quả. Thời gian load data (2.32 s) gần bằng thời gian train — tức nút thắt cổ chai của toàn pipeline nằm ở I/O đọc CSV chứ không phải ở tính toán.

**Về AUC-ROC.** Kết quả **0.8868** ở mức trung bình khá, nhưng thấp hơn đáng kể so với mức ~0.97 thường thấy trên bộ dữ liệu này. Nguyên nhân là early stopping cắt quá sớm ở vòng 41: khi kết hợp `scale_pos_weight` rất lớn (≈578) với learning rate thấp (0.05), chỉ số AUC trên tập validation dao động mạnh và chạm đỉnh sớm, khiến model dừng lại khi còn **underfit**.

**Về F1-Score và sự đánh đổi Precision/Recall.** F1 chỉ đạt **0.0614** dù Recall lên tới **0.8265**. Lý do: `scale_pos_weight` khuếch đại trọng số của lớp thiểu số lên ~578 lần, đẩy model sang xu hướng "báo động thừa" — tại ngưỡng mặc định 0.5, nó gắn cờ 2,542 giao dịch nhưng chỉ 81 là gian lận thật, kéo Precision xuống **3.19%**. Nói cách khác, model bắt được 81/98 ca gian lận (bỏ sót 17) nhưng phải đánh đổi bằng 2,461 báo động giả.

**Về Accuracy.** Con số **0.9565** trông đẹp nhưng gần như vô nghĩa trong bài toán này: chỉ cần đoán "mọi giao dịch đều hợp lệ" đã đạt 99.83%. Model của chúng ta thực chất *thấp hơn* baseline ngây thơ đó về Accuracy — đây là ví dụ điển hình cho thấy vì sao với dữ liệu mất cân bằng cực đoan, phải đánh giá bằng AUC-ROC và cặp Precision/Recall thay vì Accuracy.

**Về inference speed.** Dự đoán 1 dòng mất **0.449 ms**, nhưng khi xử lý theo lô 1000 dòng thì chi phí trung bình giảm còn 0.0018 ms/dòng — **nhanh hơn ~245 lần**. Khoảng chênh này là do phần overhead cố định (chuyển đổi dataframe, gọi hàm, khởi tạo bộ nhớ) được chia đều cho cả lô thay vì gánh trọn cho một dòng. Bài học thực tế: nếu triển khai API real-time thì độ trễ ~0.45 ms/request đã thừa đáp ứng, còn với xử lý theo mẻ thì nên gom batch để tận dụng tối đa thông lượng.

**Kết luận.** Một `t3.medium` giá ~$0.0416/giờ là quá đủ cho toàn bộ vòng đời của bài toán ML dạng bảng này. Chi phí thực tế của hạ tầng lại bị chi phối bởi **NAT Gateway** (~$0.045/giờ) — thành phần mạng chứ không phải thành phần tính toán — nên việc `terraform destroy` ngay sau khi thực hành xong là bắt buộc.

---

## 3. Hướng cải thiện

Nếu chạy lại, có ba điều chỉnh đáng làm:

1. **Bỏ `scale_pos_weight`, thay bằng tinh chỉnh ngưỡng quyết định.** Giữ model học phân phối gốc, sau đó chọn ngưỡng tối ưu F1 trên tập validation. Cách này thường nâng Precision lên trên 0.80 mà vẫn giữ Recall khoảng 0.75-0.80.
2. **Nới lỏng early stopping** (tăng patience lên 100-200 vòng, hoặc hạ learning rate xuống 0.02 kèm nhiều vòng hơn) để model không dừng khi còn underfit — kỳ vọng đưa AUC lên vùng 0.97+.
3. **Đổi metric early stopping sang `average_precision` (AUC-PR)** thay vì AUC-ROC. Với dữ liệu mất cân bằng 0.17%, AUC-PR phản ánh chất lượng thực tế tốt hơn nhiều.

---

## 4. Chi phí thực tế

Hạ tầng khởi tạo lúc **15:56 UTC**, thời điểm lập báo cáo **16:49 UTC** → tổng thời gian chạy ≈ **0.89 giờ**.

| Dịch vụ | Loại | Đơn giá/giờ | Ước tính |
|---|---|---|---|
| EC2 — Compute Node | `t3.medium` | ~$0.0416 | ~$0.037 |
| EC2 — Bastion | `t3.micro` | ~$0.0104 | ~$0.009 |
| NAT Gateway | 1 AZ | ~$0.045 + data | ~$0.040 |
| ALB | Application LB | ~$0.0225 | ~$0.020 |
| EBS | gp3 30 GB + 8 GB | ~$0.004 | ~$0.004 |
| **Tổng** | | **~$0.12/giờ** | **~$0.11** |

Ghi chú: NAT Gateway và ALB cộng lại (~$0.068/giờ) **đắt hơn cả hai máy EC2 cộng lại** (~$0.052/giờ), dù chúng không thực hiện bất kỳ phép tính ML nào.
