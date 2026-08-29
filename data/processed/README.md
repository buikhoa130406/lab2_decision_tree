# Processed data

`bank_marketing_processed.csv` được tự động tạo khi chạy project.

Các thay đổi:

- Loại `duration` để tránh temporal leakage.
- Loại 16 dòng trùng sau khi bỏ `duration`.
- Mã hóa target thành `no=0`, `yes=1`.
- Giữ 15 feature gốc; categorical feature chỉ được One-Hot Encode bên trong model pipeline để tránh rò rỉ giữa train và test.

