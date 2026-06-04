# Intent Classification AI

Dự án này phục vụ mục đích học tập và nghiên cứu liên quan đến chủ đề: **So sánh hiệu quả giữa các thuật toán Machine Learning trong bài toán Phân loại ý định (Intent Classification) trong Tự nhiên Ngôn ngữ (NLP) với bộ dữ liệu CLINIC 150**.

## Cấu trúc thư mục chính
- `dataset/`: Chứa các file dữ liệu thô (.json) và danh sách stopwords.
- [`main.py`](main.py): File chạy chính của chương trình.
- `src/`: Thư mục mã nguồn chính.
  - `src/input/`: Chứa các file corpus và label (.txt) đã được xử lý chuẩn bị cho việc huấn luyện/đánh giá (train/val/test).
  - `src/preprocess/`: Modules xử lý dữ liệu như loại bỏ stopword ([`stopword_remove.py`](src/preprocess/stopword_remove.py)), pipeline tiền xử lý ([`preprocess_pipeline.py`](src/preprocess/preprocess_pipeline.py)).
  - `src/logistic-regression/`: Nơi chứa bộ phân loại sử dụng Logistic Regression ([`mrl.py`](src/logistic-regression/mrl.py)).
  - `src/model/`: Nơi lưu trữ các mô hình đã được huấn luyện (VD: `LogisticRegression_model.pkl`).

## Tính năng
- **Tiền xử lý dữ liệu**: Xử lý dữ liệu văn bản thô, làm sạch, loại bỏ stopwords thông qua module [preprocess_pipeline.py](src/preprocess/preprocess_pipeline.py).
- **Huấn luyện Mô hình**: Huấn luyện các mô hình Machine Learning phân loại văn bản (đã cài đặt Logistic Regression).
- **Đánh giá Mô hình**: Kiểm tra và đánh giá độ chính xác của mô hình trên tập dữ liệu kiểm thử (Test set).

## Yêu cầu Hệ thống
Bạn cần cài đặt Python (phiên bản >= 3.7) và các thư viện cần thiết. Thường bao gồm:
- `numpy`
- `pandas`
- `scikit-learn`
- `nltk` (nếu dùng để tách từ/xóa stopword)

## Cài đặt

1. Clone kho lưu trữ về máy cục bộ:
   ```bash
   git clone <đường_dẫn_repo_của_bạn>
   cd intent-classification-ai

2. Tạo và kích hoạt môi trường ảo
    ```bash
    python -m venv venv
    # Trên Windows:
    venv\Scripts\activate
    # Trên macOS/Linux:
    source venv/bin/activate

3. Cài đặt các thư viện Python cần thiết
    ```bash
    pip install -r requirements.txt

## Hướng dẫn Sử dụng
Sau khi cài đặt hoàn thành cài đặt. 
Người dùng có thể chọn giữa các chế độ cli như sau:
- chat: Mở chế độ tương tác liên tục qua terminal để kiểm tra mô hình. Người dùng nhập câu liên tục và mô hình sẽ trả về kết quả dự đoán. Chọn exit hoặc quit để thoát.
    ```bash
    python main.py chat

- predict: Ở chế độ này dùng nhập riêng lẻ từng câu.
    ```bash
    python main.py predict "<câu văn người dùng muốn kiểm tra>"

- eval: Chạy đánh giá độ chính xác của mô hình trên toàn bộ tập dữ liệu kiểm thử (test set) và trả về các chỉ số phân tích (Accuracy, Precision, Recall, F1-Score).
    ```bash
    python main.py eval
