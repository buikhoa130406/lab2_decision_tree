# Checklist kiểm tra bản base

## Trước buổi họp

- [ ] Chạy `.\run_demo.ps1` thành công.
- [ ] Có `model_comparison.csv`.
- [ ] Có bốn Confusion Matrix.
- [ ] Có cây baseline và cây của ba mô hình cải thiện.
- [ ] Có `tree_rules.txt`.
- [ ] Mở được báo cáo Markdown.
- [ ] Mọi thành viên nhận được project.

## Khi đối chiếu kết quả thành viên

- [ ] Dataset và target có giống cấu hình chung không?
- [ ] `duration` đã được loại hoàn toàn chưa?
- [ ] Target có được mã hóa `no=0`, `yes=1` không?
- [ ] Train/test split có cùng random state không?
- [ ] Có dùng `stratify` không?
- [ ] Tuning có chỉ dùng tập train không?
- [ ] Accuracy và Error Rate có khớp nhau theo `Error = 1 - Accuracy` không?
- [ ] Confusion Matrix có đúng thứ tự class không?
- [ ] Có báo cáo Train Accuracy để nhận biết overfitting không?
- [ ] Mỗi phương pháp có mô tả, kết quả và giải thích không?
- [ ] Không che giấu phương pháp cho kết quả kém hơn.
- [ ] Việc chọn mô hình có dựa trên F1 của lớp `yes` không?

## Trước khi nộp

- [ ] Điền tên và MSSV thật.
- [ ] Xác nhận đóng góp thật của từng người.
- [ ] Kiểm tra lại tất cả đường dẫn hình.
- [ ] Xuất báo cáo thành `Group05 - Report.pdf`.
- [ ] Tạo `Group05 - Code` không chứa `.venv`.
- [ ] Thêm video hoặc link video.
- [ ] Chạy trên ít nhất một máy khác.
- [ ] Đóng gói đúng tên `Group05.zip`.
