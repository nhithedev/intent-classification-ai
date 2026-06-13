"""
testMKNN.py  —  đặt tại: src/k-nearest-neighbor/testMKNN.py
----------------------------------------------------------
Đánh giá KNN trên test set bằng sklearn.metrics (chỉ để đo, không phải
thuật toán). Nhất quán với testMRL.py / testMNB.py: lọc nhãn lạ (oos),
dự đoán với threshold=0.0 (raw class) rồi tính macro Accuracy/Precision/Recall/F1.
"""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── Đường dẫn ─────────────────────────────────────────────
KNN_DIR = Path(__file__).resolve().parent
LR_DIR  = KNN_DIR.parent / "logistic-regression"
for d in (KNN_DIR, LR_DIR):
    if str(d) not in sys.path:
        sys.path.append(str(d))

# Import class để pickle.load hoạt động.
# Khi mknn.py chạy trực tiếp (python mknn.py), pickle ghi class dưới module
# '__main__'. Đăng ký lại vào __main__ của tiến trình này để pickle tìm thấy.
from mknn import KNearestNeighbors          # type: ignore
import __main__
if not hasattr(__main__, "KNearestNeighbors"):
    __main__.KNearestNeighbors = KNearestNeighbors

from mrl import TfIdfVectorizer, LabelEncoder  # type: ignore  # noqa: F401

MODEL_PATH       = KNN_DIR.parent / "model" / "KNN_model.pkl"
INPUT_DIR        = KNN_DIR.parent / "input"
TEST_CORPUS_PATH = INPUT_DIR / "test_corpus.txt"
TEST_LABELS_PATH = INPUT_DIR / "test_labels.txt"


def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình tại {MODEL_PATH}.\n"
            "Hãy train trước: python src/k-nearest-neighbor/mknn.py"
        )
    print(f"--> Đang tải mô hình từ: {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts['vectorizer'], artifacts['label_encoder'], artifacts['model']


def load_test_data(corpus_path, labels_path):
    print(f"--> Đang đọc dữ liệu test từ:\n  [Corpus]: {corpus_path}\n  [Labels]: {labels_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        raw_corpus = f.readlines()
    with open(labels_path, 'r', encoding='utf-8') as f:
        raw_labels = f.readlines()

    min_len = min(len(raw_corpus), len(raw_labels))
    corpus, labels = [], []
    for c_line, l_line in zip(raw_corpus[:min_len], raw_labels[:min_len]):
        c, l = c_line.strip(), l_line.strip()
        if c and l:
            corpus.append(c)
            labels.append(l)
    if not corpus:
        raise ValueError("Dữ liệu test trống sau khi lọc dòng rỗng!")
    print(f"--> Tải thành công. Số mẫu test: {len(corpus)}")
    return corpus, labels


if __name__ == "__main__":

    # [BƯỚC 1] Đọc dữ liệu test
    try:
        test_corpus, test_labels_text = load_test_data(TEST_CORPUS_PATH, TEST_LABELS_PATH)
    except Exception as e:
        print(f"[LỖI ĐỌC DỮ LIỆU]: {e}")
        exit()

    # [BƯỚC 2] Tải mô hình
    try:
        vectorizer, label_encoder, model = load_trained_model()
        print("--> Load mô hình thành công!\n")
    except Exception as e:
        print(f"[LỖI LOAD MÔ HÌNH]: {e}")
        exit()

    # [BƯỚC 3] Lọc mẫu hợp lệ (bỏ nhãn lạ như 'oos' chưa thấy lúc train)
    print("[Tiến trình] Lọc các mẫu hợp lệ và mã hóa TF-IDF...")
    filtered_corpus, y_test_true, skipped = [], [], 0
    for text, lbl in zip(test_corpus, test_labels_text):
        if lbl in label_encoder.label_to_index:
            filtered_corpus.append(text)
            y_test_true.append(label_encoder.label_to_index[lbl])
        else:
            skipped += 1
    if skipped > 0:
        print(f"  Đã loại bỏ {skipped} mẫu có nhãn lạ (ví dụ: 'oos') chưa xuất hiện lúc Train.")

    y_test_true = np.array(y_test_true)
    X_test      = vectorizer.transform(filtered_corpus)
    assert X_test.shape[0] == len(y_test_true), "Số text và label không khớp!"

    # [BƯỚC 4] Dự đoán (threshold=0.0 → raw class, không lọc OOS)
    print("\n[Tiến trình] Đang dự đoán (có thể mất vài phút)...")
    y_test_pred, _ = model.predict_with_oos(X_test, threshold=0.0)

    # [BƯỚC 5] In metrics
    print("\n" + "=" * 40)
    print(f"{'KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (TEST METRICS)':^40}")
    print("=" * 40)

    avg_method = 'macro' if len(label_encoder.classes) > 2 else 'binary'
    print(f"Mô hình          : K-Nearest Neighbors (k={model.k}, weighted={model.weighted})")
    print(f"Tổng số mẫu test : {len(y_test_true)}")
    print(f"Chế độ trung bình: {avg_method}\n")

    acc  = accuracy_score(y_test_true, y_test_pred)
    prec = precision_score(y_test_true, y_test_pred, average=avg_method, zero_division=0)
    rec  = recall_score(y_test_true, y_test_pred, average=avg_method, zero_division=0)
    f1   = f1_score(y_test_true, y_test_pred, average=avg_method, zero_division=0)

    print(f"{'Accuracy':<15}: {acc:.4f}")
    print(f"{'Precision':<15}: {prec:.4f}")
    print(f"{'Recall':<15}: {rec:.4f}")
    print(f"{'F1-Score':<15}: {f1:.4f}")
    print("=" * 40)
