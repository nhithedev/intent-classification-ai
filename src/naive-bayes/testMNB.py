import pickle
import numpy as np
from pathlib import Path

# Import các hàm tính metric từ scikit-learn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import các class từ mnb.py để pickle có thể giải nén file .pkl thành công
import sys
BASE_DIR     = Path(__file__).resolve().parent.parent       # → src/
LOGISTIC_DIR = BASE_DIR / "logistic-regression"
NAIVE_DIR    = BASE_DIR / "naive-bayes"
for _dir in (LOGISTIC_DIR, NAIVE_DIR):
    if str(_dir) not in sys.path:
        sys.path.append(str(_dir))

from mrl import TfIdfVectorizer, LabelEncoder, INPUT_DIR    # type: ignore
from mnb import MultinomialNaiveBayes                       # type: ignore

MODEL_DIR        = BASE_DIR / "model"
TEST_CORPUS_PATH = INPUT_DIR / "test_corpus.txt"
TEST_LABELS_PATH = INPUT_DIR / "test_labels.txt"


# ── Tải mô hình đã huấn luyện ──────────────────────────────
def load_trained_model():
    """Tải lại các đối tượng (artifacts) của mô hình Naive Bayes đã lưu."""
    save_path = MODEL_DIR / "NaiveBayes_model.pkl"
    if not save_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file mô hình tại {save_path}.\n"
            "Hãy chạy 'python src/naive-bayes/mnb.py' để huấn luyện trước!"
        )

    print(f"--> Đang tải mô hình từ: {save_path}")
    with open(save_path, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts["vectorizer"], artifacts["label_encoder"], artifacts["model"]


# ── Đọc dữ liệu test ───────────────────────────────────────
def load_test_data(corpus_path, labels_path):
    """Đọc và làm sạch dữ liệu test từ các file .txt."""
    print(f"--> Đang đọc dữ liệu test từ:\n"
          f"    [Corpus]: {corpus_path}\n"
          f"    [Labels]: {labels_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_corpus = f.readlines()
    with open(labels_path, "r", encoding="utf-8") as f:
        raw_labels = f.readlines()

    min_len = min(len(raw_corpus), len(raw_labels))

    test_corpus, test_labels = [], []
    for c_line, l_line in zip(raw_corpus[:min_len], raw_labels[:min_len]):
        c, l = c_line.strip(), l_line.strip()
        if c and l:
            test_corpus.append(c)
            test_labels.append(l)

    if not test_corpus:
        raise ValueError("Dữ liệu test trống sau khi lọc dòng!")

    print(f"--> Tải thành công. Số mẫu test: {len(test_corpus)}")
    return test_corpus, test_labels


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":

    # [BƯỚC 1] Đọc dữ liệu test
    try:
        test_corpus, test_labels_text = load_test_data(
            TEST_CORPUS_PATH, TEST_LABELS_PATH
        )
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

    # [BƯỚC 3] Lọc mẫu hợp lệ và mã hóa TF-IDF
    print("[Tiến trình] Lọc các mẫu hợp lệ và mã hóa dữ liệu...")

    filtered_corpus = []
    y_test_true     = []
    skipped         = 0

    for text, lbl in zip(test_corpus, test_labels_text):
        if lbl in label_encoder.label_to_index:
            # Chỉ giữ lại các câu có nhãn đã xuất hiện trong tập train
            filtered_corpus.append(text)
            y_test_true.append(label_encoder.label_to_index[lbl])
        else:
            # Bỏ qua các nhãn lạ (ví dụ: "oos" không có trong train)
            skipped += 1

    y_test_true = np.array(y_test_true)
    X_test      = vectorizer.transform(filtered_corpus)

    if skipped > 0:
        print(f"⚠️  Đã bỏ qua {skipped} mẫu có nhãn chưa xuất hiện lúc Train (vd: 'oos').")

    assert X_test.shape[0] == len(y_test_true), \
        "Lỗi: Số lượng text và label không khớp!"

    # [BƯỚC 4] Dự đoán
    print("\n[Tiến trình] Mô hình đang dự đoán...")
    y_test_pred = model.predict(X_test)

    # [BƯỚC 5] Tính toán và in Metrics
    print("\n" + "=" * 40)
    print(f"{'KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (TEST METRICS)':^40}")
    print("=" * 40)

    avg_method = "macro" if len(label_encoder.classes) > 2 else "binary"
    print(f"Mô hình                : Multinomial Naive Bayes")
    print(f"Chế độ tính (Average)  : {avg_method}\n")

    acc  = accuracy_score(y_test_true, y_test_pred)
    prec = precision_score(y_test_true, y_test_pred, average=avg_method, zero_division=0)
    rec  = recall_score(y_test_true, y_test_pred, average=avg_method, zero_division=0)
    f1   = f1_score(y_test_true, y_test_pred, average=avg_method, zero_division=0)

    print(f"{'Accuracy':<15}: {acc:.4f}  (Độ chính xác tổng thể)")
    print(f"{'Precision':<15}: {prec:.4f}  (Độ chính xác của các dự đoán)")
    print(f"{'Recall':<15}: {rec:.4f}  (Độ phủ / Độ nhạy)")
    print(f"{'F1-Score':<15}: {f1:.4f}  (Trung bình hài hòa Prec & Rec)")
    print("=" * 40)
