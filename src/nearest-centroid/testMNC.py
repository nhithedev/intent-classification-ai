import sys
import pickle
import numpy as np
from pathlib import Path

# 1. Các hàm metric từ scikit-learn (chỉ dùng để ĐO, không để phân loại)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 2. Import class từ mnc.py để pickle.load nhận diện
from mnc import NearestCentroid, TfIdfVectorizer, LabelEncoder, MODEL_DIR, INPUT_DIR

TEST_CORPUS_PATH = INPUT_DIR / "test_corpus.txt"
TEST_LABELS_PATH = INPUT_DIR / "test_labels.txt"


def load_trained_model():
    save_path = MODEL_DIR / "NearestCentroid_model.pkl"
    if not save_path.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình tại {save_path}. Hãy chạy train trước!")
    print(f"--> Đang tải mô hình từ: {save_path}")
    with open(save_path, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts["vectorizer"], artifacts["label_encoder"], artifacts["model"]


def load_processed_test_data(corpus_path, labels_path):
    print(f"--> Đang đọc dữ liệu test từ:\n  [Corpus]: {corpus_path}\n  [Labels]: {labels_path}")
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


if __name__ == "__main__":

    # [BƯỚC 1] Đọc dữ liệu test
    try:
        test_corpus, test_labels_text = load_processed_test_data(TEST_CORPUS_PATH, TEST_LABELS_PATH)
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

    # [BƯỚC 3] Lọc các mẫu in-scope (bỏ nhãn lạ như 'oos' chưa thấy lúc train)
    print("[Tiến trình] Lọc mẫu hợp lệ và mã hóa TF-IDF...")
    filtered_corpus, y_test_true, skipped = [], [], 0
    for text, lbl in zip(test_corpus, test_labels_text):
        if lbl in label_encoder.label_to_index:
            filtered_corpus.append(text)
            y_test_true.append(label_encoder.label_to_index[lbl])
        else:
            skipped += 1

    y_test_true = np.array(y_test_true)
    X_test = vectorizer.transform(filtered_corpus)
    if skipped > 0:
        print(f"  Đã loại bỏ {skipped} mẫu có nhãn lạ (ví dụ: 'oos') chưa xuất hiện lúc Train.")

    assert X_test.shape[0] == len(y_test_true), "Số text và label không khớp!"

    # [BƯỚC 4] Dự đoán phân loại thuần (không lọc OOS) — đo năng lực phân loại
    print("\n[Tiến trình] Đang dự đoán...")
    y_test_pred = model.predict(X_test)

    # [BƯỚC 5] Tính metrics macro
    acc  = accuracy_score(y_test_true, y_test_pred)
    prec = precision_score(y_test_true, y_test_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test_true, y_test_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)

    print("\n" + "=" * 40)
    print(f"{'KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (TEST METRICS)':^40}")
    print("=" * 40)
    print(f"Mô hình          : Nearest Centroid")
    print(f"Tổng số mẫu test : {len(y_test_true)}")
    print(f"Chế độ trung bình: macro\n")
    print(f"{'Accuracy':<15}: {acc:.4f} (Độ chính xác tổng thể)")
    print(f"{'Precision':<15}: {prec:.4f} (Độ chính xác của các dự đoán)")
    print(f"{'Recall':<15}: {rec:.4f} (Độ phủ/Độ nhạy)")
    print(f"{'F1-Score':<15}: {f1:.4f} (Trung bình hài hòa Prec & Rec)")
    print("=" * 40)
