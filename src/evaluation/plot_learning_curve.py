import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
for _dir in ["logistic-regression", "naive-bayes", "k-nearest-neighbor", "nearest-centroid", "utils"]:
    sys.path.append(str(ROOT_DIR / "src" / _dir))

from tfidfcal import TfIdfVectorizer
from labelEncode import LabelEncoder
from dataio import INPUT_DIR, load_split

import mrl
import mnb
import mknn
import mnc

from sklearn.metrics import accuracy_score

def plot_learning_curves():
    TRAIN_CORPUS = INPUT_DIR / "train_corpus.txt"
    TRAIN_LABELS = INPUT_DIR / "train_labels.txt"
    VAL_CORPUS   = INPUT_DIR / "val_corpus.txt"
    VAL_LABELS   = INPUT_DIR / "val_labels.txt"

    train_corpus, train_labels = load_split(TRAIN_CORPUS, TRAIN_LABELS)
    val_corpus, val_labels = load_split(VAL_CORPUS, VAL_LABELS)

    # Lọc bỏ 'oos' trong tập val để tính in-scope accuracy đơn giản
    val_corpus_inscope = [c for c, l in zip(val_corpus, val_labels) if l != 'oos']
    val_labels_inscope = [l for l in val_labels if l != 'oos']

    label_encoder = LabelEncoder()
    # Chú ý: Cần fit trên toàn bộ dữ liệu (hoặc train+val) để bảo đảm đủ index
    all_labels = list(set(train_labels + val_labels_inscope))
    label_encoder.fit_transform(all_labels)

    # Mặc dù fit_transform, nhưng để an toàn ta có thể lấy thủ công:
    y_val = []
    for l in val_labels_inscope:
        if l in label_encoder.label_to_index:
            y_val.append(label_encoder.label_to_index[l])
        else:
            y_val.append(-1)
    
    y_val = np.array(y_val)
    # Lọc bỏ các mẫu -1 trong y_val
    valid_mask = y_val != -1
    val_corpus_inscope = [c for c, m in zip(val_corpus_inscope, valid_mask) if m]
    y_val = y_val[valid_mask]

    train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    n_total = len(train_corpus)

    # Giảm epochs cho Logistic Regression để tăng tốc
    models_info = [
        ("Logistic Regression", lambda: mrl.MultinomialLogisticRegression(learning_rate=0.5, epochs=150)),
        ("Naive Bayes", lambda: mnb.MultinomialNaiveBayes()),
        ("K-Nearest Neighbors", lambda: mknn.KNearestNeighbors(k=11, weighted=True)),
        ("Nearest Centroid", lambda: mnc.NearestCentroid())
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (model_name, model_fn) in enumerate(models_info):
        print(f"\n======================================")
        print(f"Đang đánh giá {model_name}...")
        train_scores = []
        val_scores = []
        actual_sizes = []

        from sklearn.model_selection import train_test_split
        for frac in train_sizes:
            if frac == 1.0:
                subset_corpus = train_corpus
                subset_labels = train_labels
            else:
                subset_corpus, _, subset_labels, _ = train_test_split(
                    train_corpus, train_labels, train_size=frac, stratify=train_labels, random_state=42
                )
            subset_size = len(subset_corpus)
            
            # Khởi tạo lại Vectorizer cho mỗi tập con
            vectorizer = TfIdfVectorizer()
            X_train_sub = vectorizer.fit_transform(subset_corpus)
            X_val_sub = vectorizer.transform(val_corpus_inscope)
            
            # Encode y_train
            y_train_sub = np.array([label_encoder.label_to_index[l] for l in subset_labels if l in label_encoder.label_to_index])
            
            # Nếu có mẫu bị mất do không có nhãn thì điều chỉnh X_train
            # (Nhưng do ta fit label_encoder trên toàn bộ nên sẽ không mất)
            
            model = model_fn()
            
            import io
            import contextlib
            
            # Ẩn output của LR để khỏi trôi console
            with contextlib.redirect_stdout(io.StringIO()):
                model.fit(X_train_sub, y_train_sub)

            # Predict
            if hasattr(model, "predict_with_oos"):
                preds_train, _ = model.predict_with_oos(X_train_sub, threshold=0.0)
                preds_val, _ = model.predict_with_oos(X_val_sub, threshold=0.0)
            else:
                preds_train = model.predict(X_train_sub)
                preds_val = model.predict(X_val_sub)

            # Tính Accuracy
            train_acc = accuracy_score(y_train_sub, preds_train)
            val_acc = accuracy_score(y_val, preds_val)

            train_scores.append(train_acc)
            val_scores.append(val_acc)
            actual_sizes.append(subset_size)
            
            print(f"  Size: {subset_size:5d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Vẽ đồ thị
        ax = axes[idx]
        ax.plot(actual_sizes, train_scores, 'o-', color="r", label="Training score")
        ax.plot(actual_sizes, val_scores, 'o-', color="g", label="Validation score")
        ax.set_title(f"Learning Curve ({model_name})")
        ax.set_xlabel("Số lượng mẫu huấn luyện")
        ax.set_ylabel("Accuracy")
        ax.legend(loc="lower right")
        ax.grid(True)
        ax.set_ylim(0.0, 1.05)

    plt.tight_layout()
    save_path = Path(__file__).resolve().parent / "learning_curves.png"
    plt.savefig(save_path, bbox_inches="tight")
    print(f"\nĐã lưu biểu đồ tại: {save_path}")

if __name__ == "__main__":
    plot_learning_curves()
