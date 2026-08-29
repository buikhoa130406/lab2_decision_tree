# Group05 Base - Bank Marketing Decision Tree

Đây là bản base hoàn chỉnh cho Lab 2 môn Cơ sở AI, sử dụng **Bank Marketing** từ UCI để dự đoán khách hàng có đăng ký tiền gửi có kỳ hạn hay không.

## Quyết định dữ liệu đã thống nhất

- File: `data/raw/bank-full.csv`.
- Target gốc: `y`.
- Mã hóa: `no=0`, `yes=1`.
- Positive class: `yes`.
- Bài toán: binary classification.
- Metric tối ưu chính: F1-score của lớp `yes`.
- Loại hoàn toàn `duration` vì giá trị này chỉ có sau cuộc gọi.
- Giữ `unknown` như một category riêng.
- One-Hot Encode categorical features bên trong pipeline.

## Pipeline đã hoàn thành

1. Đọc file bằng dấu phân cách `;` và kiểm tra schema.
2. Loại `duration` để tránh temporal leakage.
3. Loại 16 dòng trùng xuất hiện sau khi bỏ `duration`.
4. Chia train/test 80/20 bằng `stratify` và `random_state=42`.
5. Huấn luyện baseline.
6. Thử ba phương pháp cải thiện:
   - Controlled complexity.
   - Cost-complexity pruning.
   - Balanced class weights.
7. Chọn tham số bằng 5-fold CV trên train với F1 của `yes`.
8. Đánh giá test bằng Accuracy, Error Rate, Precision/Recall/F1 `yes`, Balanced Accuracy, PR-AUC và ROC-AUC.
9. Sinh model, bảng, luật cây và biểu đồ.

## Cấu trúc

```text
group05_base/
├── data/
│   ├── raw/bank-full.csv
│   └── processed/bank_marketing_processed.csv
├── src/
│   ├── config.py
│   ├── data.py
│   ├── modeling.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── main.py
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── report/Group05_Base_Report.md
├── video/Group05_Base_Video_Script.md
├── materials/
├── requirements.txt
└── run_demo.ps1
```

## Chạy bằng VS Code

Mở folder `D:\AI\pj2\group05_base`, chọn Python interpreter:

```text
C:\Users\KHOA\anaconda3\python.exe
```

Sau đó mở Terminal và chạy:

```powershell
.\run_demo.ps1
```

Hoặc:

```powershell
& 'C:\Users\KHOA\anaconda3\python.exe' .\src\main.py
```

Trên máy khác:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\src\main.py
```

Một lần chạy đầy đủ có thể mất khoảng 2-3 phút do cross-validation trên hơn 45 nghìn mẫu.

## Kết quả hiện tại

| Model | Accuracy | Error | Precision yes | Recall yes | F1 yes | Depth | Leaves |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 82,86% | 17,14% | 29,49% | 33,36% | 31,31% | 45 | 4.727 |
| Controlled complexity | 88,02% | 11,98% | 48,03% | 28,83% | 36,03% | 49 | 1.344 |
| Cost-complexity pruning | **88,77%** | **11,23%** | **53,47%** | 31,29% | 39,48% | 28 | 393 |
| Balanced class weights | 83,04% | 16,96% | 35,68% | **55,95%** | **43,58%** | **7** | **66** |

Vì nhóm ưu tiên F1 của `yes`, **Balanced class weights** là mô hình được chọn. Pruning phù hợp hơn nếu ngân hàng ưu tiên Precision và Accuracy, nhưng bỏ sót nhiều khách hàng `yes` hơn.

## Output quan trọng

- `outputs/metrics/model_comparison.csv`: kết quả tổng hợp.
- `outputs/metrics/tuning_summary.json`: tham số tốt nhất theo CV.
- `outputs/metrics/tree_rules.txt`: luật cây dạng text.
- `outputs/figures/confusion_matrices.png`: Confusion Matrix.
- `outputs/figures/model_comparison.png`: so sánh metric.
- `outputs/figures/*_tree_readable.png`: bốn tầng đầu của từng cây.
- `outputs/models/best_model.joblib`: pipeline Balanced Class Weights được chọn bằng CV.

## Lưu ý

- Accuracy của mô hình luôn đoán `no` đã khoảng 88,30%; vì vậy không dùng Accuracy làm metric tối ưu chính.
- `unknown` không phải `NaN`; project xem nó là một trạng thái thông tin chưa biết.
- Không điều chỉnh tham số sau khi nhìn test chỉ để làm đẹp kết quả.
- Cần điền tên, MSSV, đóng góp thật và diễn đạt báo cáo bằng hiểu biết của nhóm trước khi nộp.

