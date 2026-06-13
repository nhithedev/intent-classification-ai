import sys
import numpy as np # type: ignore
import pickle
from pathlib import Path

# ── Thư viện chung trong src/utils/ ───────────────────────
# TfIdfVectorizer, LabelEncoder, load_split + đường dẫn đều lấy từ src/utils/
# (nguồn DUY NHẤT) thay vì định nghĩa riêng ở đây. Các model khác cũng import
# từ utils/, không còn phụ thuộc ngược vào file Logistic Regression này.
BASE_DIR  = Path(__file__).resolve().parent.parent   # → src/
UTILS_DIR = BASE_DIR / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from tfidfcal import TfIdfVectorizer            # type: ignore
from labelEncode import LabelEncoder            # type: ignore
from dataio import load_split, INPUT_DIR, MODEL_DIR  # type: ignore

# ========================================================
# MULTINOMIAL LOGISTIC REGRESSION
# ========================================================
class MultinomialLogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        self.weights = np.zeros((n_features, n_classes))
        self.bias    = np.zeros((1, n_classes))
        y_one_hot    = np.eye(n_classes)[y]
        
        print(f"Bắt đầu huấn luyện với {self.epochs} vòng lặp...")
        for epoch in range(self.epochs):
            scores        = np.dot(X, self.weights) + self.bias
            probabilities = self._softmax(scores)
            error         = probabilities - y_one_hot
            
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error, axis=0, keepdims=True)
            
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db
            
            if epoch % 100 == 0 or epoch == self.epochs - 1:
                loss = -np.mean(np.sum(y_one_hot * np.log(probabilities + 1e-15), axis=1))
                print(f"  - Vòng lặp {epoch:4d} | Loss: {loss:.4f}")

    def predict(self, X):
        scores = np.dot(X, self.weights) + self.bias
        return np.argmax(self._softmax(scores), axis=1)
    
    def predict_with_oos(self, X, threshold=0.5):
        """
        Dự đoán nhãn với cơ chế Out-Of-Scope (OOS).
        Trả về:
        - final_predictions: chỉ số nhãn (-1 nếu dưới threshold → OOS)
        - max_probs        : confidence score tương ứng
        """
        probabilities  = self._softmax(np.dot(X, self.weights) + self.bias)
        max_probs      = np.max(probabilities, axis=1)
        predicted_idx  = np.argmax(probabilities, axis=1)
        final_predictions = np.where(max_probs >= threshold, predicted_idx, -1)
        return final_predictions, max_probs

# ========================================================
# HELPER: ĐÁNH GIÁ TRÊN TẬP VALIDATION
# ========================================================
def evaluate(model, vectorizer, label_encoder, corpus, labels, threshold=0.5):
    """
    Đánh giá mô hình trên tập val.
    - Nhãn 'oos' trong file được coi là out-of-scope thực sự.
    - model.predict_with_oos() trả về -1 cho câu dưới threshold.
    In ra: accuracy tổng thể, in-scope acc, oos recall.
    """
    X = vectorizer.transform(corpus)

    # Tách in-scope và oos theo nhãn thực
    is_oos_true = np.array([lbl == "oos" for lbl in labels])

    # Dự đoán với OOS threshold
    preds_idx, max_probs = model.predict_with_oos(X, threshold=threshold)

    # ── Accuracy tổng thể (chỉ tính in-scope) ──────────────
    inscope_mask   = ~is_oos_true
    y_true_inscope = label_encoder.fit_transform(
        [lbl for lbl in labels if lbl != "oos"]
    ) if False else None   # chỉ để gợi ý; dùng cách dưới cho nhất quán

    # So sánh: in-scope đúng khi pred khớp nhãn thực VÀ không bị đánh OOS
    correct_inscope = 0
    total_inscope   = 0
    for i, (pred, true_lbl) in enumerate(zip(preds_idx, labels)):
        if true_lbl == "oos":
            continue
        total_inscope += 1
        if pred != -1 and label_encoder.index_to_label.get(pred) == true_lbl:
            correct_inscope += 1

    inscope_acc = correct_inscope / total_inscope if total_inscope else 0

    # ── OOS Recall: câu oos thực sự được đánh là -1 ────────
    oos_indices    = [i for i, lbl in enumerate(labels) if lbl == "oos"]
    oos_recall     = (
        sum(1 for i in oos_indices if preds_idx[i] == -1) / len(oos_indices)
        if oos_indices else 0
    )

    # ── In-scope bị nhầm thành OOS (False Rejection Rate) ──
    frr = (
        sum(1 for i, lbl in enumerate(labels)
            if lbl != "oos" and preds_idx[i] == -1) / total_inscope
        if total_inscope else 0
    )

    print(f"\n{'─'*45}")
    print(f"  Threshold        : {threshold:.2f}")
    print(f"  In-scope Accuracy: {inscope_acc*100:.2f}%  ({correct_inscope}/{total_inscope})")
    print(f"  OOS Recall       : {oos_recall*100:.2f}%  ({sum(1 for i in oos_indices if preds_idx[i]==-1)}/{len(oos_indices)})")
    print(f"  False Rejection  : {frr*100:.2f}%  (in-scope bị đánh nhầm thành OOS)")
    print(f"{'─'*45}")
    return inscope_acc, oos_recall

# ========================================================
# HỆ THỐNG ĐIỀU HÀNH CHÍNH (MAIN PIPELINE)
# ========================================================
if __name__ == "__main__":

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Chọn file input ────────────────────────────────────
    TRAIN_CORPUS = INPUT_DIR / "train_corpus.txt"
    TRAIN_LABELS = INPUT_DIR / "train_labels.txt"
    VAL_CORPUS   = INPUT_DIR / "val_corpus.txt"
    VAL_LABELS   = INPUT_DIR / "val_labels.txt"

    # ── 1. Đọc dữ liệu train ──────────────────────────────
    print("=" * 50)
    print("[Bước 1] Đọc dữ liệu...")
    try:
        train_corpus, train_labels = load_split(TRAIN_CORPUS, TRAIN_LABELS)
        val_corpus,   val_labels   = load_split(VAL_CORPUS,   VAL_LABELS)
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        exit()

    print(f"  Train : {len(train_corpus)} mẫu")
    print(f"  Val   : {len(val_corpus)} mẫu  "
          f"(trong đó oos: {sum(1 for l in val_labels if l == 'oos')})")

    # ── 2. Vector hóa (fit trên train, transform val) ─────
    print("\n[Bước 2] Vector hóa TF-IDF...")
    vectorizer = TfIdfVectorizer()
    X_train    = vectorizer.fit_transform(train_corpus)
    X_val      = vectorizer.transform(val_corpus)
    print(f"  Ma trận train: {X_train.shape}")
    print(f"  Ma trận val  : {X_val.shape}")

    # ── 3. Mã hóa nhãn (chỉ fit trên train) ──────────────
    print("\n[Bước 3] Mã hóa nhãn...")
    label_encoder = LabelEncoder()
    y_train       = label_encoder.fit_transform(train_labels)
    print(f"  Số class: {len(label_encoder.classes)}")

    # ── 4. Huấn luyện ─────────────────────────────────────
    print("\n[Bước 4] Huấn luyện mô hình MLR...")
    model = MultinomialLogisticRegression(learning_rate=0.5, epochs=500)
    model.fit(X_train, y_train)

    # ── 5. Đánh giá trên tập Validation ──────────────────
    print("\n[Bước 5] Đánh giá trên tập Validation:")
    evaluate(model, vectorizer, label_encoder,
             val_corpus, val_labels, threshold=0.5)

    # ── 6. Dự đoán thử nghiệm ─────────────────────────────
    print("\n[Bước 6] Dự đoán thử nghiệm:")
    new_sentences = [
        "set warning bank account starts running low",
        "my credit card was declined at the store",
    ]
    X_new        = vectorizer.transform(new_sentences)
    preds_idx, _ = model.predict_with_oos(X_new, threshold=0.5)
    for sentence, idx in zip(new_sentences, preds_idx):
        label = label_encoder.index_to_label.get(idx, "OOS (out-of-scope)")
        print(f"  Câu  : '{sentence}'")
        print(f"  → Dự đoán: {label}\n")

    # ── 7. Lưu mô hình ────────────────────────────────────
    save_path = MODEL_DIR / "LogisticRegression_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "vectorizer"   : vectorizer,
            "label_encoder": label_encoder,
            "model"        : model,
        }, f)
    print(f"[Thành công] Mô hình đã lưu tại: {save_path}")