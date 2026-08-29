# Sườn kịch bản video - Bank Marketing

Thời lượng đề xuất: 12-15 phút.

## 1. Mở đầu - Thành viên 1 - 1 phút

> Nhóm 05 xây dựng Decision Tree để dự đoán khách hàng có đăng ký tiền gửi kỳ hạn trước khi ngân hàng thực hiện liên hệ. Nhóm ưu tiên F1 của lớp yes do dữ liệu mất cân bằng.

Giới thiệu luồng: dataset, preprocessing, baseline, ba cải thiện và so sánh.

## 2. Dataset và tiền xử lý - Thành viên 2 - 3 phút

- Nguồn UCI, 45.211 mẫu và 16 feature gốc.
- Target `y`: no/yes.
- `yes` chỉ chiếm 11,70%.
- Loại `duration` vì chỉ biết sau cuộc gọi.
- Loại 16 dòng trùng sau khi bỏ duration.
- Sáu numeric và chín categorical feature.
- One-Hot Encoding trong pipeline.
- Giữ `unknown` như một category.
- Chia 80/20 có stratify.

Hiển thị `class_distribution.png` và giải thích tại sao Accuracy không đủ.

## 3. Baseline - Thành viên 3 - 3 phút

- Train Accuracy 100%, Test Accuracy 82,86%.
- F1 `yes` 31,31%.
- Depth 45, 4.727 leaf: overfitting.
- Root node `poutcome_success`.
- Đọc một luật mức cao của cây.
- Hiển thị baseline tree và Confusion Matrix.

## 4. Controlled Complexity và Pruning - Thành viên 4 - 3 phút

### Controlled Complexity

- GridSearch trên criterion, depth, split và leaf.
- Test Accuracy 88,02%, F1 `yes` 36,03%.
- Số leaf giảm còn 1.344.

### Pruning

- `ccp_alpha=0.00006051`.
- Accuracy cao nhất 88,77%.
- Precision `yes` cao nhất 53,47%.
- Recall `yes` chỉ 31,29%.

## 5. Balanced Class Weights - Thành viên 5 - 2 phút

- Lớp `yes` ít hơn `no` khoảng 7,5 lần.
- Cấu hình: depth 7, split 50, leaf 10, balanced weights.
- Recall `yes` 55,95%, F1 `yes` 43,58%.
- Accuracy giảm do mô hình chấp nhận nhiều False Positive hơn.
- Đây là trade-off hợp lý theo metric nhóm đã chọn.

## 6. So sánh và kết luận - 2 phút

- Hiển thị `model_comparison.png` và `confusion_matrices.png`.
- Pruning tốt nhất nếu ưu tiên Accuracy/Precision.
- Balanced Class Weights tốt nhất nếu ưu tiên F1/Recall `yes`.
- Nhóm chọn Balanced vì metric được thống nhất trước là F1 `yes`.

Kết luận mẫu:

> Kết quả cho thấy không thể chọn mô hình chỉ bằng Accuracy. Trên dataset mất cân bằng, mô hình pruning đạt Accuracy cao nhưng bỏ sót nhiều khách hàng đăng ký. Balanced Class Weights tạo cây nhỏ hơn, tăng Recall và đạt F1 của lớp yes cao nhất.

## Checklist video

- Mỗi thành viên giải thích phần mình làm.
- Không chỉ đọc số liệu.
- Giải thích lý do bỏ `duration`.
- Nêu rõ positive class và metric ưu tiên.
- Zoom đủ để đọc cây và Confusion Matrix.
- Kiểm tra link chia sẻ trước khi nộp.

