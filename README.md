# Intent Classification AI

Dự án này phục vụ mục đích học tập và nghiên cứu liên quan đến chủ đề: **So sánh hiệu quả giữa các thuật toán Machine Learning trong bài toán Phân loại ý định (Intent Classification) trong Tự nhiên Ngôn ngữ (NLP) với bộ dữ liệu CLINIC 150**.

## Cấu trúc thư mục chính
- `dataset/`: Chứa các file dữ liệu thô (.json) và danh sách stopwords.
- [`main.py`](main.py): File chạy chính của chương trình.
- [`benchmark.py`](benchmark.py): Chạy và so sánh đầy đủ 4 thuật toán, đối chiếu với các file số liệu `solieu_*.md`.
- `src/`: Thư mục mã nguồn chính.
  - `src/input/`: Chứa các file corpus và label (.txt) đã được xử lý chuẩn bị cho việc huấn luyện/đánh giá (train/val/test).
  - `src/utils/`: Thư viện dùng chung — TF-IDF ([`tfidfcal.py`](src/utils/tfidfcal.py)), mã hóa nhãn ([`labelEncode.py`](src/utils/labelEncode.py)), I/O ([`dataio.py`](src/utils/dataio.py)).
  - `src/preprocess/`: Modules xử lý dữ liệu như loại bỏ stopword ([`stopword_remove.py`](src/preprocess/stopword_remove.py)), pipeline tiền xử lý ([`preprocess_pipeline.py`](src/preprocess/preprocess_pipeline.py)).
  - `src/logistic-regression/`: Bộ phân loại Logistic Regression ([`mrl.py`](src/logistic-regression/mrl.py)).
  - `src/naive-bayes/`: Bộ phân loại Multinomial Naive Bayes ([`mnb.py`](src/naive-bayes/mnb.py)).
  - `src/k-nearest-neighbor/`: Bộ phân loại K-Nearest Neighbors ([`mknn.py`](src/k-nearest-neighbor/mknn.py)).
  - `src/nearest-centroid/`: Bộ phân loại Nearest Centroid / Rocchio ([`mnc.py`](src/nearest-centroid/mnc.py)).
  - `src/model/`: Nơi lưu trữ các mô hình đã được huấn luyện (`LogisticRegression_model.pkl`, `NaiveBayes_model.pkl`, `KNN_model.pkl`, `NearestCentroid_model.pkl`).

## Tính năng
- **Tiền xử lý dữ liệu**: Xử lý dữ liệu văn bản thô, làm sạch, loại bỏ stopwords thông qua module [preprocess_pipeline.py](src/preprocess/preprocess_pipeline.py).
- **Huấn luyện Mô hình**: Hỗ trợ huấn luyện và so sánh bốn thuật toán Machine Learning (tự cài đặt thuần NumPy):
  - **Logistic Regression** (`src/logistic-regression/mrl.py`)
  - **Multinomial Naive Bayes** (`src/naive-bayes/mnb.py`)
  - **K-Nearest Neighbors** (`src/k-nearest-neighbor/mknn.py`)
  - **Nearest Centroid** (`src/nearest-centroid/mnc.py`)
- **Đánh giá Mô hình**: Kiểm tra và đánh giá độ chính xác của mô hình trên tập dữ liệu kiểm thử (Test set) với các chỉ số Accuracy, Precision, Recall, F1-Score.
- **Phát hiện câu ngoài phạm vi (OOS)**: Cả bốn mô hình đều hỗ trợ từ chối trả lời khi câu hỏi không thuộc 150 ý định đã học.

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

# Huấn luyện K-Nearest Neighbors
python src/k-nearest-neighbor/mknn.py

# Huấn luyện Nearest Centroid
python src/nearest-centroid/mnc.py
```

### Bước 1: Chạy chương trình
Tất cả các lệnh đều hỗ trợ tham số `-m` (hoặc `--model`) để chọn mô hình:
- `-m lr`  : Logistic Regression
- `-m nb`  : Naive Bayes
- `-m knn` : K-Nearest Neighbors
- `-m nc`  : Nearest Centroid
- `-m all` : Tất cả (mặc định)

#### `chat` — Mở chế độ tương tác liên tục
Người dùng nhập câu liên tục, mô hình trả về kết quả dự đoán. Gõ `exit` hoặc `quit` để thoát.
```bash
python main.py chat           # So sánh cả 4 model (mặc định)
python main.py chat -m lr     # Chỉ Logistic Regression
```

#### `predict` — Dự đoán riêng lẻ một câu
```bash
python main.py predict "<câu văn người dùng muốn kiểm tra>"           # Cả 4 model (mặc định)
python main.py predict "<câu văn người dùng muốn kiểm tra>" -m nc     # Chỉ Nearest Centroid
```

#### `eval` — Đánh giá mô hình trên tập Test
Chạy đánh giá và trả về các chỉ số Accuracy, Precision, Recall, F1-Score.
```bash
python main.py eval           # Đánh giá cả 4 model (mặc định)
python main.py eval -m nb     # Chỉ Naive Bayes
```

## Kết quả Benchmark (CLINIC 150)

Chạy `python benchmark.py` để tái lập toàn bộ bảng dưới đây. Các chỉ số accuracy/F1/size mang tính tất định (deterministic); riêng thời gian train/inference biến động theo máy.

> **Điều kiện đo:** TF-IDF `fit` trên train (vocab 5.219 từ); val/test chỉ `transform`. Khi đánh giá accuracy/F1, bỏ các mẫu nhãn `oos` để đo phân loại in-scope thuần (`predict()`, không áp ngưỡng OOS). Macro F1/Precision/Recall tính trên test set với `average='macro'`.

| Metric | Logistic Regression | Naive Bayes | KNN (k=11) | Nearest Centroid |
|---|---|---|---|---|
| Train Accuracy | 98.31% | 98.51% | 100.00% | 89.11% |
| Val Accuracy | 89.37% | 87.70% | 81.20% | 82.23% |
| Test Accuracy | **89.62%** | 87.78% | 82.16% | 83.07% |
| Overfit Gap (train − val) | +8.95% | +10.81% | +18.80% | **+6.88%** |
| Macro F1 (test) | **0.8954** | 0.8757 | 0.8182 | 0.8286 |
| Macro Precision | **0.9009** | 0.8831 | 0.8327 | 0.8445 |
| Macro Recall | **0.8962** | 0.8778 | 0.8216 | 0.8307 |
| Training Time | 2.84 m | **0.60 s** | 7.65 s | 0.97 s |
| Inference Time | 0.0249 ms/câu | 0.0264 ms/câu | 2.0065 ms/câu | 0.0514 ms/câu |
| Model Size (.pkl) | 6.13 MB | 6.13 MB | 597.54 MB | 6.13 MB |

