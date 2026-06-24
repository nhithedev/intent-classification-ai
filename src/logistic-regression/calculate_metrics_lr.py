import time
import os
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Thêm utils vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
UTILS_DIR = BASE_DIR / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from dataio import load_split, INPUT_DIR, MODEL_DIR
from mrl import MultinomialLogisticRegression
from tfidfcal import TfIdfVectorizer
from labelEncode import LabelEncoder

def evaluate_metrics():
    print("Đang tải dữ liệu...")
    TRAIN_CORPUS = INPUT_DIR / "train_corpus.txt"
    TRAIN_LABELS = INPUT_DIR / "train_labels.txt"
    VAL_CORPUS   = INPUT_DIR / "val_corpus.txt"
    VAL_LABELS   = INPUT_DIR / "val_labels.txt"
    TEST_CORPUS  = INPUT_DIR / "test_corpus.txt"
    TEST_LABELS  = INPUT_DIR / "test_labels.txt"

    train_corpus, train_labels = load_split(TRAIN_CORPUS, TRAIN_LABELS)
    val_corpus, val_labels     = load_split(VAL_CORPUS, VAL_LABELS)
    test_corpus, test_labels   = load_split(TEST_CORPUS, TEST_LABELS)

    model_path = MODEL_DIR / "LogisticRegression_model.pkl"
    
    print("Đang tải mô hình...")
    if not model_path.exists():
        print("Không tìm thấy mô hình. Hãy chạy mrl.py trước!")
        return
        
    with open(model_path, "rb") as f:
        saved_data = pickle.load(f)
        vectorizer = saved_data["vectorizer"]
        label_encoder = saved_data["label_encoder"]
        model = saved_data["model"]

    print("Vector hóa dữ liệu...")
    X_train = vectorizer.transform(train_corpus)
    X_val = vectorizer.transform(val_corpus)
    X_test = vectorizer.transform(test_corpus)

    y_train = label_encoder.transform(train_labels)
    y_val = label_encoder.transform(val_labels)
    y_test = label_encoder.transform(test_labels)

    # Loại bỏ nhãn 'oos' (-1) để tính các metric chuẩn
    train_mask = y_train != -1
    val_mask = y_val != -1
    test_mask = y_test != -1

    X_train_in, y_train_in = X_train[train_mask], y_train[train_mask]
    X_val_in, y_val_in = X_val[val_mask], y_val[val_mask]
    X_test_in, y_test_in = X_test[test_mask], y_test[test_mask]

    print("Dự đoán trên tập Train...")
    train_preds = model.predict(X_train_in)
    train_acc = accuracy_score(y_train_in, train_preds)

    print("Dự đoán trên tập Val...")
    val_preds = model.predict(X_val_in)
    val_acc = accuracy_score(y_val_in, val_preds)

    print("Dự đoán trên tập Test...")
    # Đo thời gian inference
    start_time = time.time()
    test_preds = model.predict(X_test_in)
    end_time = time.time()
    inference_time_ms = ((end_time - start_time) / len(X_test_in)) * 1000

    test_acc = accuracy_score(y_test_in, test_preds)
    
    overfit_gap = train_acc - val_acc

    print("Tính toán F1, Precision, Recall...")
    macro_f1 = f1_score(y_test_in, test_preds, average='macro')
    macro_prec = precision_score(y_test_in, test_preds, average='macro', zero_division=0)
    macro_rec = recall_score(y_test_in, test_preds, average='macro', zero_division=0)

    # Đo thời gian training (ước lượng bằng cách train 1 epoch rồi nhân lên)
    print("Ước lượng thời gian training...")
    dummy_model = MultinomialLogisticRegression(learning_rate=0.5, epochs=1, l2_penalty=0.01)
    start_train = time.time()
    dummy_model.fit(X_train_in, y_train_in)
    end_train = time.time()
    estimated_training_time = (end_train - start_train) * model.epochs

    # Kích thước mô hình
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # Top 5 confused pairs
    print("Tính toán Confusion Matrix...")
    cm = confusion_matrix(y_test_in, test_preds)
    confused_pairs = []
    classes = label_encoder.classes
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true": classes[i],
                    "pred": classes[j],
                    "count": cm[i, j]
                })
    
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)
    top_5 = confused_pairs[:5]

    # Ghi ra file solieu_LR.md
    output_path = BASE_DIR / "logistic-regression" / "solieu_LR.md"
    
    md_content = f"""=== BẢNG SỐ LIỆU — [LogisticRegression] ===

Train Accuracy  : {train_acc * 100:.2f}%
Val Accuracy    : {val_acc * 100:.2f}%
Test Accuracy   : {test_acc * 100:.2f}%
Overfit gap     : {overfit_gap * 100:.2f}% (train - val)

Macro F1        : {macro_f1:.4f}
Macro Precision : {macro_prec:.4f}
Macro Recall    : {macro_rec:.4f}

Training Time   : ~{estimated_training_time:.4f} s (ước lượng)
Inference Time  : {inference_time_ms:.4f} ms/câu

Model Size      : {model_size_mb:.4f} MB

Top 5 confused pairs:
"""
    for idx, pair in enumerate(top_5, 1):
        md_content += f"{idx}. {pair['true']} → {pair['pred']}  ({pair['count']} lần)\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nĐã ghi báo cáo thành công vào {output_path}!")
    print(md_content)

if __name__ == "__main__":
    evaluate_metrics()
