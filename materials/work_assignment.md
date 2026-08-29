# Phân công công việc - Nhóm 05

| Thành viên | Phụ trách | File/code chính | Phần báo cáo | Phần video |
|---|---|---|---|---|
| TV1 | Dataset, cấu hình, tích hợp | `config.py`, README | Nhóm, mở đầu, references | Mở đầu và kết luận |
| TV2 | EDA và preprocessing | `data.py` | Dataset Description | Dataset và tiền xử lý |
| TV3 | Baseline và phân tích cây | `evaluation.py`, baseline | Baseline, Analysis of Tree | Baseline và luật cây |
| TV4 | Controlled complexity và pruning | `modeling.py` phần tuning/pruning | Improvement 1-2 | Hyperparameter tuning và pruning |
| TV5 | Class weighting và comparison | `modeling.py`, `visualization.py` | Improvement 3, Comparison | Class imbalance và so sánh |

## Quy tắc review chéo

- TV1 review TV2.
- TV2 review TV3.
- TV3 review TV4.
- TV4 review TV5.
- TV5 review TV1.

## Definition of Done

Một phần chỉ hoàn thành khi có đủ:

1. Code chạy được từ `main.py`.
2. Không thay đổi test split chung.
3. Có metric và output lưu thành file.
4. Có ít nhất một hình hoặc bảng nếu phù hợp.
5. Có giải thích kết quả, không chỉ chép số.
6. Có nội dung báo cáo và lời trình bày.
7. Đã được review chéo.
