"""
testMRF.py  —  đặt tại: src/random-forest/testMRF.py
----------------------------------------------------
Đánh giá Random Forest trên test set (macro metrics), nhất quán với các model khác.
"""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RF_DIR    = Path(__file__).resolve().parent
BASE_DIR  = RF_DIR.parent
UTILS_DIR = BASE_DIR / "utils"
for d in (RF_DIR, UTILS_DIR):
    if str(d) not in sys.path:
        sys.path.append(str(d))

# Import class để pickle.load nhận diện (model train bằng cách chạy mrf.py trực tiếp
# → class lưu dưới '__main__'; đăng ký lại vào __main__ của tiến trình này).
from mrf import RandomForest                # type: ignore
import __main__
if not hasattr(__main__, "RandomForest"):
    __main__.RandomForest = RandomForest

from tfidfcal import TfIdfVectorizer        # type: ignore  # noqa: F401
from labelEncode import LabelEncoder        # type: ignore  # noqa: F401

MODEL_PATH       = BASE_DIR / "model" / "RandomForest_model.pkl"
INPUT_DIR        = BASE_DIR / "input"
TEST_CORPUS_PATH = INPUT_DIR / "test_corpus.txt"
TEST_LABELS_PATH = INPUT_DIR / "test_labels.txt"


def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình tại {MODEL_PATH}.\n"
            "Hãy train trước: python src/random-forest/mrf.py"
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
        raise ValueError("Dữ liệu test trống sau khi lọc!")
    print(f"--> Tải thành công. Số mẫu test: {len(corpus)}")
    return corpus, labels


if __name__ == "__main__":
    try:
        test_corpus, test_labels_text = load_test_data(TEST_CORPUS_PATH, TEST_LABELS_PATH)
    except Exception as e:
        print(f"[LỖI ĐỌC DỮ LIỆU]: {e}")
        exit()

    try:
        vectorizer, label_encoder, model = load_trained_model()
        print("--> Load mô hình thành công!\n")
    except Exception as e:
        print(f"[LỖI LOAD MÔ HÌNH]: {e}")
        exit()

    print("[Tiến trình] Lọc các mẫu hợp lệ và mã hóa TF-IDF...")
    filtered_corpus, y_test_true, skipped = [], [], 0
    for text, lbl in zip(test_corpus, test_labels_text):
        if lbl in label_encoder.label_to_index:
            filtered_corpus.append(text)
            y_test_true.append(label_encoder.label_to_index[lbl])
        else:
            skipped += 1
    if skipped > 0:
        print(f"  Đã loại bỏ {skipped} mẫu có nhãn lạ (ví dụ: 'oos').")

    y_test_true = np.array(y_test_true)
    X_test      = vectorizer.transform(filtered_corpus)

    print("\n[Tiến trình] Đang dự đoán...")
    y_test_pred, _ = model.predict_with_oos(X_test, threshold=0.0)

    print("\n" + "=" * 40)
    print(f"{'KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (TEST METRICS)':^40}")
    print("=" * 40)
    avg_method = 'macro' if len(label_encoder.classes) > 2 else 'binary'
    print(f"Mô hình          : Random Forest (n_trees={len(model.trees)})")
    print(f"Tổng số mẫu test : {len(y_test_true)}")
    print(f"Chế độ trung bình: {avg_method}\n")
    print(f"{'Accuracy':<15}: {accuracy_score(y_test_true, y_test_pred):.4f}")
    print(f"{'Precision':<15}: {precision_score(y_test_true, y_test_pred, average=avg_method, zero_division=0):.4f}")
    print(f"{'Recall':<15}: {recall_score(y_test_true, y_test_pred, average=avg_method, zero_division=0):.4f}")
    print(f"{'F1-Score':<15}: {f1_score(y_test_true, y_test_pred, average=avg_method, zero_division=0):.4f}")
    print("=" * 40)