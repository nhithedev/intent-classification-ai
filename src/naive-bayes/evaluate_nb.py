import sys
import time
import os
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Fix encoding tiếng Việt trên Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass


# ── Cấu hình đường dẫn ──
BASE_DIR = Path(__file__).resolve().parent.parent   # → src/
UTILS_DIR = BASE_DIR / "utils"
NAIVE_DIR = BASE_DIR / "naive-bayes"

# Đảm bảo import được các module từ src/utils/ và src/naive-bayes/
for _dir in (UTILS_DIR, NAIVE_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

from tfidfcal import TfIdfVectorizer
from labelEncode import LabelEncoder
from dataio import load_split, INPUT_DIR, MODEL_DIR
from mnb import MultinomialNaiveBayes

def main():
    print("=" * 60)
    print("ĐANG CHẠY PIPELINE HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH NAIVE BAYES...")
    print("=" * 60)

    # 1. Đọc dữ liệu
    TRAIN_CORPUS = INPUT_DIR / "train_corpus.txt"
    TRAIN_LABELS = INPUT_DIR / "train_labels.txt"
    VAL_CORPUS   = INPUT_DIR / "val_corpus.txt"
    VAL_LABELS   = INPUT_DIR / "val_labels.txt"
    TEST_CORPUS  = INPUT_DIR / "test_corpus.txt"
    TEST_LABELS  = INPUT_DIR / "test_labels.txt"

    train_corpus, train_labels = load_split(TRAIN_CORPUS, TRAIN_LABELS)
    val_corpus, val_labels = load_split(VAL_CORPUS, VAL_LABELS)
    test_corpus, test_labels = load_split(TEST_CORPUS, TEST_LABELS)

    # 2. Huấn luyện mô hình & Đo thời gian huấn luyện
    print("\n1. Bắt đầu Vector hóa và Huấn luyện...")
    start_train_time = time.time()

    vectorizer = TfIdfVectorizer()
    X_train = vectorizer.fit_transform(train_corpus)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)

    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(X_train, y_train)

    train_time = time.time() - start_train_time
    print(f"   --> Thời gian huấn luyện: {train_time:.4f} s")

    # 3. Lưu mô hình xuống đĩa
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_DIR / "NaiveBayes_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "vectorizer": vectorizer,
            "label_encoder": label_encoder,
            "model": model,
        }, f)
    
    # Đo dung lượng mô hình (MB)
    model_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"   --> Lưu mô hình thành công tại {save_path} ({model_size_mb:.4f} MB)")

    # 4. Lọc dữ liệu Val và Test (Bỏ qua OOS nhãn lạ chưa xuất hiện trong train)
    print("\n2. Đánh giá mô hình trên các tập dữ liệu...")
    
    # Train Predict
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred) * 100

    # Val Predict (Chỉ giữ lại nhãn trong tập train để đo độ chính xác phân loại)
    filtered_val_corpus = []
    y_val_true = []
    for text, lbl in zip(val_corpus, val_labels):
        if lbl in label_encoder.label_to_index:
            filtered_val_corpus.append(text)
            y_val_true.append(label_encoder.label_to_index[lbl])
    
    y_val_true = np.array(y_val_true)
    X_val = vectorizer.transform(filtered_val_corpus)
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val_true, y_val_pred) * 100

    # Test Predict & Đo Inference Time (bao gồm cả Vectorization và Prediction trên tập Test sạch)
    filtered_test_corpus = []
    y_test_true = []
    for text, lbl in zip(test_corpus, test_labels):
        if lbl in label_encoder.label_to_index:
            filtered_test_corpus.append(text)
            y_test_true.append(label_encoder.label_to_index[lbl])
            
    y_test_true = np.array(y_test_true)
    
    # Bắt đầu đo inference time
    start_inf_time = time.time()
    X_test = vectorizer.transform(filtered_test_corpus)
    y_test_pred = model.predict(X_test)
    total_inf_time = time.time() - start_inf_time
    
    inference_time_ms_per_query = (total_inf_time / len(filtered_test_corpus)) * 1000
    test_acc = accuracy_score(y_test_true, y_test_pred) * 100

    # 5. Tính toán các chỉ số Macro (F1, Precision, Recall) trên tập Test
    macro_f1 = f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    macro_precision = precision_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_test_true, y_test_pred, average='macro', zero_division=0)

    # 6. Tìm Top 5 cặp nhầm lẫn nhiều nhất (Confused Pairs) trên tập Test
    confusions = []
    for true_idx, pred_idx in zip(y_test_true, y_test_pred):
        if true_idx != pred_idx:
            true_label = label_encoder.inverse_transform([true_idx])[0]
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            confusions.append((true_label, pred_label))
            
    confused_counter = Counter(confusions)
    top_5_confused = confused_counter.most_common(5)

    # Thêm các cặp trống nếu ít hơn 5 cặp bị nhầm lẫn
    while len(top_5_confused) < 5:
        top_5_confused.append((("N/A", "N/A"), 0))

    # 7. In ra bảng số liệu chuẩn theo yêu cầu của đề bài
    overfit_gap = train_acc - val_acc

    print("\n" + "=" * 50)
    print("=== BẢNG SỐ LIỆU — [NaiveBayes] ===")
    print("=" * 50)
    print(f"Train Accuracy  : {train_acc:.2f}%")
    print(f"Val Accuracy    : {val_acc:.2f}%")
    print(f"Test Accuracy   : {test_acc:.2f}%")
    print(f"Overfit gap     : {overfit_gap:.2f}% (train - val)")
    print()
    print(f"Macro F1        : {macro_f1:.4f}")
    print(f"Macro Precision : {macro_precision:.4f}")
    print(f"Macro Recall    : {macro_recall:.4f}")
    print()
    print(f"Training Time   : {train_time:.4f} s")
    print(f"Inference Time  : {inference_time_ms_per_query:.4f} ms/câu")
    print()
    print(f"Model Size      : {model_size_mb:.4f} MB")
    print()
    print("Top 5 confused pairs:")
    for idx, ((true_lbl, pred_lbl), count) in enumerate(top_5_confused, 1):
        if true_lbl != "N/A":
            print(f"{idx}. {true_lbl} → {pred_lbl}  ({count} lần)")
        else:
            print(f"{idx}. __________ → __________  (__ lần)")
    print("=" * 50)

if __name__ == "__main__":
    main()
