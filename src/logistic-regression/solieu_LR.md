=== BẢNG SỐ LIỆU — [LogisticRegression] ===

Train Accuracy  : 98.31%
Val Accuracy    : 89.37%
Test Accuracy   : 89.62%
Overfit gap     : 8.95% (train - val)

Macro F1        : 0.8954
Macro Precision : 0.9009
Macro Recall    : 0.8962

Training Time   : 192.5506 s
Inference Time  : 0.0255 ms/câu

Model Size      : 6.1317 MB

Top 5 confused pairs:
1. improve_credit_score → credit_score  (10 lần)
2. change_user_name → change_ai_name  (7 lần)
3. calendar → calendar_update  (6 lần)
4. redeem_rewards → rewards_balance  (6 lần)
5. todo_list_update → todo_list  (5 lần)

=== PHÂN TÍCH OOS DETECTION ===

Confidence trung bình:
| Loại dữ liệu | Confidence |
|---|---|
| In-scope | 0.7753 |
| OOS | 0.2171 |

Nhận xét:
- Sự chênh lệch confidence giữa In-scope (0.7753) và OOS (0.2171) của Logistic Regression rất rõ rệt và trực quan.
- Nhờ khoảng cách phân tách rộng này, mô hình Logistic Regression rất tối ưu khi cần đặt một ngưỡng (threshold) cụ thể để lọc bỏ câu Out-Of-Scope mà ít làm ảnh hưởng đến độ chính xác của tập In-scope.
