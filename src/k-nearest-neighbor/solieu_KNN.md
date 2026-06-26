=== BẢNG SỐ LIỆU — [KNN] ===

Train Accuracy  : 100.00%
Val Accuracy    : 81.20%
Test Accuracy   : 82.16%
Overfit gap     : 18.80% (train - val)

Macro F1        : 0.8182
Macro Precision : 0.8327
Macro Recall    : 0.8216

Training Time   : 14.4433 s
Inference Time  : 2.8354 ms/câu

Model Size      : 597.5396 MB

Top 5 confused pairs:
1. improve_credit_score → credit_score  (15 lần)
2. change_user_name → user_name  (14 lần)
3. share_location → current_location  (14 lần)
4. todo_list_update → todo_list  (8 lần)
5. shopping_list_update → shopping_list  (7 lần)

=== PHÂN TÍCH OOS DETECTION ===

Confidence trung bình:
| Loại dữ liệu | Confidence |
|---|---|
| In-scope | 0.7361 |
| OOS | 0.3893 |

Nhận xét:
- Câu thuộc intent đã học thường có confidence cao.
- Câu OOS thường nằm xa dữ liệu train → confidence thấp hơn.
- KNN có khả năng phát hiện OOS khá tự nhiên thông qua khoảng cách tới các hàng xóm gần nhất.
