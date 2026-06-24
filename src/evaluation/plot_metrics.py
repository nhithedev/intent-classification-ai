import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Thêm root dir vào sys.path để import main.py
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Fix encode tiếng việt trên windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import main
from main import ALL_MODELS, load_model
from src.utils.dataio import INPUT_DIR

# Đăng ký các class vào namespace của plot_metrics (__main__) để pickle load thành công
import mrl, mnb, mknn, mnc
setattr(sys.modules['__main__'], 'MultinomialLogisticRegression', mrl.MultinomialLogisticRegression)
setattr(sys.modules['__main__'], 'MultinomialNaiveBayes', mnb.MultinomialNaiveBayes)
setattr(sys.modules['__main__'], 'KNearestNeighbors', mknn.KNearestNeighbors)
setattr(sys.modules['__main__'], 'NearestCentroid', mnc.NearestCentroid)

def evaluate_models():
    TEST_CORPUS_PATH = INPUT_DIR / "test_corpus.txt"
    TEST_LABELS_PATH = INPUT_DIR / "test_labels.txt"

    with open(TEST_CORPUS_PATH, 'r', encoding='utf-8') as f:
        raw_corpus = f.readlines()
    with open(TEST_LABELS_PATH, 'r', encoding='utf-8') as f:
        raw_labels = f.readlines()

    min_len = min(len(raw_corpus), len(raw_labels))
    test_corpus = [l.strip() for l in raw_corpus[:min_len]]
    test_labels_text = [l.strip() for l in raw_labels[:min_len]]

    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for m in ALL_MODELS:
        vectorizer, label_encoder, model, display_name = load_model(m)
        model_names.append(display_name)
        
        filtered_corpus, y_test_true, skipped = [], [], 0
        for text, lbl in zip(test_corpus, test_labels_text):
            if not text or not lbl:
                continue
            if lbl in label_encoder.label_to_index:
                filtered_corpus.append(text)
                y_test_true.append(label_encoder.label_to_index[lbl])
            else:
                skipped += 1

        y_test_true = np.array(y_test_true)
        X_test = vectorizer.transform(filtered_corpus)
        
        # threshold=0.0 để đánh giá phân loại thuần
        if hasattr(model, "predict_with_oos"):
            y_test_pred, _ = model.predict_with_oos(X_test, threshold=0.0)
        else:
            y_test_pred = model.predict(X_test)
            
        avg_method = 'macro' if len(label_encoder.classes) > 2 else 'binary'
        
        acc = accuracy_score(y_test_true, y_test_pred)
        prec = precision_score(y_test_true, y_test_pred, average=avg_method, zero_division=0)
        rec = recall_score(y_test_true, y_test_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y_test_true, y_test_pred, average=avg_method, zero_division=0)
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        
        print(f"[{display_name}] Đã đánh giá xong.")

    return model_names, accuracies, precisions, recalls, f1_scores

def plot_metrics(model_names, accuracies, precisions, recalls, f1_scores):
    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width*1.5, accuracies, width, label='Accuracy', color='#4CAF50')
    rects2 = ax.bar(x - width*0.5, precisions, width, label='Precision', color='#2196F3')
    rects3 = ax.bar(x + width*0.5, recalls, width, label='Recall', color='#FFC107')
    rects4 = ax.bar(x + width*1.5, f1_scores, width, label='F1-Score', color='#F44336')

    ax.set_ylabel('Điểm số (0.0 - 1.0)')
    ax.set_title('So sánh hiệu suất các thuật toán phân loại ý định (In-scope)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    ax.set_ylim(0, 1.1)

    # Thêm text giá trị trên đầu cột
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    fig.tight_layout()
    
    # Save the figure
    save_path = Path(__file__).resolve().parent / "metrics_comparison.png"
    plt.savefig(save_path, bbox_inches="tight")
    print(f"\nĐã lưu biểu đồ tại: {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    print("Đang chạy đánh giá các mô hình...")
    names, acc, prec, rec, f1 = evaluate_models()
    plot_metrics(names, acc, prec, rec, f1)
