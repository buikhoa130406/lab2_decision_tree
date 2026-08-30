# Agenda họp nhóm - Bank Marketing

Thời lượng đề xuất: 60-75 phút.

## 1. Chốt mục tiêu - 10 phút

- Dự đoán khách hàng đăng ký tiền gửi trước khi liên hệ.
- Binary classification: `no=0`, `yes=1`.
- Positive class: `yes`.
- Không sử dụng `duration`.
- Metric chính: F1 của `yes`.

## 2. Chạy demo - 10 phút

```powershell
.\run_demo.ps1
```

Lệnh được chạy trong Terminal tại thư mục gốc của repository.

Mở lần lượt:

1. `outputs/metrics/dataset_metadata.json`.
2. `outputs/metrics/model_comparison.csv`.
3. `outputs/figures/class_distribution.png`.
4. `outputs/figures/confusion_matrices.png`.
5. `outputs/figures/model_comparison.png`.
6. `report/Group05_Base_Report.md`.

## 3. Điểm cần giải thích - 15 phút

- Tại sao phải bỏ `duration`.
- Tại sao `unknown` được giữ như category.
- Tại sao One-Hot Encoding nằm trong pipeline.
- Tại sao Accuracy không phải metric chính.
- Khác biệt giữa Precision, Recall và F1 của `yes`.

## 4. So sánh mô hình - 15 phút

- Baseline overfit: Train Accuracy 100%, depth 45.
- Pruning cho Accuracy và Precision cao nhất.
- Balanced Class Weights cho Recall và F1 `yes` cao nhất.
- Nhóm ưu tiên F1 nên chọn Balanced Class Weights.

## 5. Phân công và deadline - 15 phút

Sử dụng `materials/work_assignment.md`. Mỗi phần phải có code, kết quả, hình, giải thích, nội dung báo cáo và lời thuyết trình.

## Quyết định cần ghi biên bản

- Xác nhận giữ nguyên metric F1 `yes`.
- Xác nhận ba phương pháp cải thiện.
- Deadline code và report từng thành viên.
- Người tích hợp, người ghép báo cáo và ngày quay video.
