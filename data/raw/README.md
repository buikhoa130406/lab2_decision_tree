# Raw data

File sử dụng: `bank-full.csv` từ UCI Bank Marketing.

- Dấu phân cách: `;`.
- 45.211 dòng dữ liệu gốc.
- 17 cột: 16 feature và target `y`.
- Kích thước chính xác: `4.610.348` byte.
- SHA-256 chuẩn của file được commit trong repository: `D1513EC63B385506F7CFCE9F2C5CAA9FE99E7BA4E8C3FA264B3AAF0F849ED32D`.

Nguồn: <https://archive.ics.uci.edu/dataset/222/bank>

Kiểm tra trên Windows PowerShell, tại thư mục gốc của project:

```powershell
(Get-FileHash -Algorithm SHA256 .\data\raw\bank-full.csv).Hash
```

Kiểm tra trên macOS hoặc Linux:

```bash
sha256sum data/raw/bank-full.csv
```

Repository cấu hình file CSV là dữ liệu giữ nguyên byte để Git không tự đổi ký tự xuống dòng giữa các hệ điều hành. Nếu vừa cập nhật project và hash vẫn khác, hãy khôi phục riêng file dữ liệu bằng `git restore data/raw/bank-full.csv`, sau đó kiểm tra lại.

Không chỉnh sửa trực tiếp file này. Pipeline loại `duration` trong code; file raw vẫn giữ nguyên đầy đủ dữ liệu UCI.
