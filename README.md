# Intent Classification AI

Dự án này phục vụ mục đích học tập và nghiên cứu liên quan đến chủ đề: **So sánh hiệu quả giữa các thuật toán Machine Learning trong bài toán Phân loại ý định (Intent Classification) trong Tự nhiên Ngôn ngữ (NLP) với bộ dữ liệu CLINIC 150**.

## Cấu trúc thư mục chính
- `dataset/`: Chứa các file dữ liệu thô (.json) và danh sách stopwords.
- [`main.py`](main.py): File chạy chính của chương trình.
- `src/`: Thư mục mã nguồn chính.
  - `src/input/`: Chứa các file corpus và label (.txt) đã được xử lý chuẩn bị cho việc huấn luyện/đánh giá (train/val/test).
  - `src/preprocess/`: Modules xử lý dữ liệu như loại bỏ stopword ([`stopword_remove.py`](src/preprocess/stopword_remove.py)), pipeline tiền xử lý ([`preprocess_pipeline.py`](src/preprocess/preprocess_pipeline.py)).
  - `src/logistic-regression/`: Bộ phân loại sử dụng Logistic Regression ([`mrl.py`](src/logistic-regression/mrl.py)).
  - `src/naive-bayes/`: Bộ phân loại sử dụng Multinomial Naive Bayes ([`mnb.py`](src/naive-bayes/mnb.py)).
  - `src/decision-tree/`: Bộ phân loại sử dụng Decision Tree ([`mdt.py`](src/decision-tree/mdt.py)).
  - `src/model/`: Nơi lưu trữ các mô hình đã được huấn luyện (VD: `LogisticRegression_model.pkl`, `NaiveBayes_model.pkl`, `DecisionTree_model.pkl`).

## Tính năng
- **Tiền xử lý dữ liệu**: Xử lý dữ liệu văn bản thô, làm sạch, loại bỏ stopwords thông qua module [preprocess_pipeline.py](src/preprocess/preprocess_pipeline.py).
- **Huấn luyện Mô hình**: Hỗ trợ huấn luyện và so sánh hai thuật toán Machine Learning:
  - **Logistic Regression** (`src/logistic-regression/mrl.py`)
  - **Multinomial Naive Bayes** (`src/naive-bayes/mnb.py`)
  - **Decesion Tree** (`src/decision-tree/mdt.py`)
- **Đánh giá Mô hình**: Kiểm tra và đánh giá độ chính xác của mô hình trên tập dữ liệu kiểm thử (Test set) với các chỉ số Accuracy, Precision, Recall, F1-Score.
- **Phát hiện câu ngoài phạm vi (OOS)**: Cả hai mô hình đều hỗ trợ từ chối trả lời khi câu hỏi không thuộc 150 ý định đã học.

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

### Bước 0: Huấn luyện mô hình trước khi sử dụng
Mỗi mô hình cần được huấn luyện một lần để tạo ra file `.pkl` trong `src/model/`.
```bash
# Huấn luyện Logistic Regression
python src/logistic-regression/mrl.py

# Huấn luyện Naive Bayes
python src/naive-bayes/mnb.py

# Huấn luyện Decision 
python src/decision-tree/mdt.py
```

### Bước 1: Chạy chương trình
Tất cả các lệnh đều hỗ trợ tham số `-m` (hoặc `--model`) để chọn mô hình:
- `-m lr` : Logistic Regression (mặc định)
- `-m nb` : Naive Bayes
- `-m dt` : Decision Tree

#### `chat` — Mở chế độ tương tác liên tục
Người dùng nhập câu liên tục, mô hình trả về kết quả dự đoán. Gõ `exit` hoặc `quit` để thoát.
```bash
python main.py chat           # Logistic Regression (mặc định, không cần -m)
python main.py chat -m lr     # Logistic Regression (chỉ định tường minh)
```

#### `predict` — Dự đoán riêng lẻ một câu
```bash
python main.py predict "<câu văn người dùng muốn kiểm tra>"           # Logistic Regression (mặc định)
python main.py predict "<câu văn người dùng muốn kiểm tra>" -m lr     # Logistic Regression (chỉ định tường minh)
```

#### `eval` — Đánh giá mô hình trên tập Test
Chạy đánh giá và trả về các chỉ số Accuracy, Precision, Recall, F1-Score.
```bash
python main.py eval           # Logistic Regression (mặc định, không cần -m)
python main.py eval -m lr     # Logistic Regression (chỉ định tường minh)
```
