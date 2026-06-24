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
KNN_DIR = BASE_DIR / "k-nearest-neighbor"

# Đảm bảo import được các module từ src/utils/ và src/k-nearest-neighbor/
for _dir in (UTILS_DIR, KNN_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

from tfidfcal import TfIdfVectorizer
from labelEncode import LabelEncoder
from dataio import load_split, INPUT_DIR, MODEL_DIR
from mknn import KNearestNeighbors, tune_hyperparams

def main():
    print("=" * 60)
    print("ĐANG CHẠY PIPELINE HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH KNN...")
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

    # Lọc OOS khỏi train (KNN chỉ học các nhãn in-scope)
    pairs = [(c, l) for c, l in zip(train_corpus, train_labels) if l != "oos"]
    train_corpus, train_labels = [p[0] for p in pairs], [p[1] for p in pairs]

    print(f"  Train: {len(train_corpus)} mẫu (in-scope)")
    print(f"  Val  : {len(val_corpus)} mẫu")
    print(f"  Test : {len(test_corpus)} mẫu")

    # 2. Vector hóa và mã hóa nhãn
    print("\n1. Vector hóa TF-IDF và mã hóa nhãn...")
    vectorizer = TfIdfVectorizer()
    X_train = vectorizer.fit_transform(train_corpus)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)

    # 3. Huấn luyện mô hình KNN & Đo thời gian huấn luyện (bao gồm cả quá trình tune siêu tham số)
    print("\n2. Bắt đầu Tune tham số và Huấn luyện KNN...")
    start_train_time = time.time()

    # Tạo vector TF-IDF cho Validation để phục vụ tuning
    X_val = vectorizer.transform(val_corpus)

    tuner = KNearestNeighbors(k=15, weighted=True, normalize=True)
    tuner.fit(X_train, y_train)

    print("  Đang tune k và weighted voting...")
    best_k, best_weighted = tune_hyperparams(
        tuner, X_val, val_labels, label_encoder,
        candidate_ks=[1, 3, 5, 7, 9, 11, 15, 21, 31],
    )

    print(f"  --> Siêu tham số tốt nhất tìm được: k={best_k}, weighted={best_weighted}")
    
    # Train KNN cuối với các tham số tốt nhất
    model = KNearestNeighbors(k=best_k, weighted=best_weighted, normalize=True)
    model.fit(X_train, y_train)

    train_time = time.time() - start_train_time
    print(f"   --> Thời gian huấn luyện + tuning: {train_time:.4f} s")

    # 4. Lưu mô hình KNN
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_DIR / "KNN_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "vectorizer": vectorizer,
            "label_encoder": label_encoder,
            "model": model,
        }, f)
    
    # Đo dung lượng mô hình
    model_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"   --> Lưu mô hình thành công tại {save_path} ({model_size_mb:.4f} MB)")

    # 5. Đánh giá mô hình trên các tập dữ liệu
    print("\n3. Đánh giá mô hình...")

    # Dự đoán trên Train (sử dụng batching của KNN để tối ưu tốc độ)
    print("  Đang dự đoán trên tập Train...")
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred) * 100

    # Lọc Val
    filtered_val_corpus = []
    y_val_true = []
    for text, lbl in zip(val_corpus, val_labels):
        if lbl in label_encoder.label_to_index:
            filtered_val_corpus.append(text)
            y_val_true.append(label_encoder.label_to_index[lbl])
    y_val_true = np.array(y_val_true)
    X_val_clean = vectorizer.transform(filtered_val_corpus)
    y_val_pred = model.predict(X_val_clean)
    val_acc = accuracy_score(y_val_true, y_val_pred) * 100

    # Lọc Test
    filtered_test_corpus = []
    y_test_true = []
    for text, lbl in zip(test_corpus, test_labels):
        if lbl in label_encoder.label_to_index:
            filtered_test_corpus.append(text)
            y_test_true.append(label_encoder.label_to_index[lbl])
    y_test_true = np.array(y_test_true)

    # Dự đoán trên Test & Đo Inference Time
    print("  Đang dự đoán trên tập Test...")
    start_inf_time = time.time()
    X_test = vectorizer.transform(filtered_test_corpus)
    y_test_pred = model.predict(X_test)
    total_inf_time = time.time() - start_inf_time
    
    inference_time_ms_per_query = (total_inf_time / len(filtered_test_corpus)) * 1000
    test_acc = accuracy_score(y_test_true, y_test_pred) * 100

    # 6. Tính toán các chỉ số Macro (F1, Precision, Recall) trên tập Test
    macro_f1 = f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    macro_precision = precision_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_test_true, y_test_pred, average='macro', zero_division=0)

    # 7. Tìm Top 5 cặp nhầm lẫn nhiều nhất (Confused Pairs) trên tập Test
    confusions = []
    for true_idx, pred_idx in zip(y_test_true, y_test_pred):
        if true_idx != pred_idx:
            true_label = label_encoder.inverse_transform([true_idx])[0]
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            confusions.append((true_label, pred_label))
            
    confused_counter = Counter(confusions)
    top_5_confused = confused_counter.most_common(5)

    while len(top_5_confused) < 5:
        top_5_confused.append((("N/A", "N/A"), 0))

    # 8. In bảng số liệu
    overfit_gap = train_acc - val_acc

    print("\n" + "=" * 50)
    print("=== BẢNG SỐ LIỆU — [KNN] ===")
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
