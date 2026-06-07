import math
import numpy as np
import pickle
from pathlib import Path

# ── Đường dẫn (mnb.py nằm tại src/naive-bayes/) ───────────
BASE_DIR  = Path(__file__).resolve().parent.parent   # → src/
INPUT_DIR = BASE_DIR / "input"
MODEL_DIR = BASE_DIR / "model"

# Thêm thư mục logistic-regression vào path để dùng lại
# TfIdfVectorizer và LabelEncoder của đồng đội
import sys
LOGISTIC_DIR = BASE_DIR / "logistic-regression"
if str(LOGISTIC_DIR) not in sys.path:
    sys.path.append(str(LOGISTIC_DIR))

from mrl import TfIdfVectorizer, LabelEncoder, load_split  # type: ignore


# ========================================================
# THÀNH PHẦN CỐT LÕI: MULTINOMIAL NAIVE BAYES
# ========================================================
class MultinomialNaiveBayes:
    """
    Bộ phân loại Multinomial Naive Bayes tự viết từ đầu bằng NumPy.

    Ý tưởng toán học:
        Với một câu văn X gồm các từ (w1, w2, ..., wn), ta tìm nhãn ý định
        c* sao cho xác suất hậu nghiệm P(c | X) là lớn nhất:

            c* = argmax_c [ log P(c) + Σ log P(w_i | c) ]
                                       i

        Trong đó:
            - P(c)       : Xác suất tiên nghiệm (Prior) — tỉ lệ xuất hiện của nhãn c
                           trong tập huấn luyện.
            - P(w | c)   : Xác suất hợp lẽ (Likelihood) — xác suất gặp từ w
                           trong các câu thuộc nhãn c.
            - Laplace Smoothing (alpha): Cộng thêm alpha vào tử số và
                           alpha * |V| vào mẫu số để tránh xác suất bằng 0
                           khi gặp từ lạ chưa xuất hiện lúc huấn luyện.

        Vì xác suất rất nhỏ dễ gây tràn số (underflow), ta chuyển sang
        không gian logarithm để biến tích thành tổng.

    Attributes:
        alpha          : Hệ số làm mịn Laplace (mặc định 1.0).
        log_priors     : Mảng log P(c) cho từng lớp, shape (n_classes,).
        log_likelihoods: Ma trận log P(w | c) cho từng cặp từ-lớp,
                         shape (n_classes, n_features).
        classes        : Mảng chứa danh sách chỉ số nhãn duy nhất.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Khởi tạo mô hình Multinomial Naive Bayes.

        Args:
            alpha: Hệ số làm mịn Laplace. Giá trị càng cao → mô hình
                   càng ít bị ảnh hưởng bởi các từ hiếm gặp.
                   Thường dùng alpha = 1.0 (Laplace smoothing thuần túy).
        """
        self.alpha           = alpha
        self.log_priors      = None   # shape: (n_classes,)
        self.log_likelihoods = None   # shape: (n_classes, n_features)
        self.classes         = None   # shape: (n_classes,)

    # ──────────────────────────────────────────────────────────
    # BƯỚC HUẤN LUYỆN
    # ──────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Tính log-prior và log-likelihood từ tập huấn luyện.

        Luồng tính toán:
            1. Xác định danh sách các lớp (nhãn duy nhất).
            2. Tính log P(c) = log(số mẫu của lớp c / tổng số mẫu).
            3. Với mỗi lớp c:
                - Tổng hợp tất cả giá trị TF-IDF của các câu thuộc lớp c
                  → được vector "tổng từ" của lớp c (word_counts_c).
                - Áp dụng Laplace Smoothing:
                    log P(w | c) = log( (word_counts_c[w] + alpha)
                                       / (sum(word_counts_c) + alpha * n_features) )

        Args:
            X: Ma trận TF-IDF, shape (n_samples, n_features).
            y: Mảng nhãn số nguyên, shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.classes          = np.unique(y)
        n_classes             = len(self.classes)

        # Khởi tạo các mảng lưu kết quả
        self.log_priors      = np.zeros(n_classes)
        self.log_likelihoods = np.zeros((n_classes, n_features))

        print(f"Bắt đầu huấn luyện Multinomial Naive Bayes...")
        print(f"  Số mẫu          : {n_samples}")
        print(f"  Số đặc trưng    : {n_features}")
        print(f"  Số lớp          : {n_classes}")

        for idx, c in enumerate(self.classes):
            # Lấy tất cả các hàng (câu văn) thuộc lớp c
            X_c = X[y == c]

            # --- Tính Log Prior: log P(c) ---
            # P(c) = số mẫu thuộc lớp c / tổng số mẫu
            self.log_priors[idx] = math.log(X_c.shape[0] / n_samples)

            # --- Tính Log Likelihood: log P(w | c) với Laplace Smoothing ---
            # Tổng tất cả giá trị TF-IDF của từng từ trong các câu thuộc lớp c
            word_counts_c = X_c.sum(axis=0)        # shape: (n_features,)

            # Mẫu số: tổng tất cả từ của lớp c + alpha * kích thước từ vựng
            denom = word_counts_c.sum() + self.alpha * n_features

            # Tử số: giá trị từng từ + alpha (tránh xác suất = 0)
            self.log_likelihoods[idx] = np.log(
                (word_counts_c + self.alpha) / denom
            )

        print("Huấn luyện hoàn tất!")

    # ──────────────────────────────────────────────────────────
    # DỰ ĐOÁN CƠ BẢN (không lọc OOS)
    # ──────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán nhãn cho từng câu trong X.

        Với mỗi câu x, tính điểm số (score) cho từng lớp c:
            score(c) = log P(c) + Σ x[w] * log P(w | c)
                                   w
        Nhãn được chọn là lớp có điểm số cao nhất (argmax).

        Args:
            X: Ma trận TF-IDF, shape (n_samples, n_features).

        Returns:
            Mảng chỉ số nhãn dự đoán, shape (n_samples,).
        """
        # X @ log_likelihoods.T → (n_samples, n_classes)
        # Cộng thêm log_priors (broadcast tự động)
        scores = X @ self.log_likelihoods.T + self.log_priors
        return np.argmax(scores, axis=1)

    # ──────────────────────────────────────────────────────────
    # DỰ ĐOÁN KÈM LỌC OUT-OF-SCOPE (OOS)
    # ──────────────────────────────────────────────────────────
    def predict_with_oos(
        self, X: np.ndarray, threshold: float = 0.5
    ):
        """
        Dự đoán nhãn kèm cơ chế phát hiện câu ngoài phạm vi (Out-Of-Scope).

        Luồng xử lý:
            1. Tính điểm số thô (log-probability scores) cho từng lớp.
            2. Chuyển điểm thô thành xác suất (confidence) bằng hàm Softmax.
            3. Nếu xác suất cao nhất < threshold → đánh dấu là OOS (trả về -1).

        Tại sao dùng Softmax?
            Các giá trị log-probability âm và có scale rất khác nhau.
            Softmax chuẩn hóa chúng về khoảng [0, 1] và tổng bằng 1,
            giúp ta diễn giải như xác suất tự tin (confidence score).

        Args:
            X        : Ma trận TF-IDF, shape (n_samples, n_features).
            threshold: Ngưỡng confidence tối thiểu. Câu có max confidence
                       nhỏ hơn threshold sẽ bị phân loại là OOS.

        Returns:
            Tuple:
                final_predictions: Mảng nhãn dự đoán (-1 nếu là OOS),
                                   shape (n_samples,).
                max_probs        : Mảng điểm confidence tương ứng,
                                   shape (n_samples,).
        """
        # Bước 1: Tính điểm số thô — shape: (n_samples, n_classes)
        scores = X @ self.log_likelihoods.T + self.log_priors

        # Bước 2: Áp dụng Softmax (trừ max trước để tránh tràn số)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs      = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Bước 3: Lấy xác suất và nhãn có giá trị cao nhất
        max_probs     = np.max(probs, axis=1)
        predicted_idx = np.argmax(probs, axis=1)

        # Bước 4: Nếu confidence < threshold → OOS (-1)
        final_predictions = np.where(max_probs >= threshold, predicted_idx, -1)

        return final_predictions, max_probs


# ========================================================
# HÀM TIỆN ÍCH: ĐÁNH GIÁ TRÊN TẬP VALIDATION
# ========================================================
def evaluate(
    model: MultinomialNaiveBayes,
    vectorizer: TfIdfVectorizer,
    label_encoder: LabelEncoder,
    corpus: list,
    labels: list,
    threshold: float = 0.5,
):
    """
    Đánh giá mô hình MNB trên một tập dữ liệu (thường là tập Validation).

    In ra:
        - Ngưỡng OOS đang dùng.
        - In-scope Accuracy: tỉ lệ câu in-scope được phân loại đúng nhãn.
        - OOS Recall: tỉ lệ câu OOS thực sự được nhận diện là OOS.
        - False Rejection Rate: tỉ lệ câu in-scope bị nhầm thành OOS.

    Args:
        model        : Mô hình MultinomialNaiveBayes đã huấn luyện.
        vectorizer   : TfIdfVectorizer đã fit trên tập train.
        label_encoder: LabelEncoder đã fit trên tập train.
        corpus       : Danh sách câu văn cần đánh giá.
        labels       : Danh sách nhãn chữ tương ứng.
        threshold    : Ngưỡng confidence để lọc OOS.

    Returns:
        Tuple (inscope_acc, oos_recall).
    """
    X = vectorizer.transform(corpus)
    preds_idx, max_probs = model.predict_with_oos(X, threshold=threshold)

    # Tách chỉ số OOS thực sự trong tập đánh giá
    oos_indices   = [i for i, lbl in enumerate(labels) if lbl == "oos"]
    total_inscope = sum(1 for lbl in labels if lbl != "oos")

    # Đếm số câu in-scope được phân loại đúng nhãn
    correct_inscope = 0
    for i, (pred, true_lbl) in enumerate(zip(preds_idx, labels)):
        if true_lbl == "oos":
            continue
        if pred != -1 and label_encoder.index_to_label.get(pred) == true_lbl:
            correct_inscope += 1

    inscope_acc = correct_inscope / total_inscope if total_inscope else 0.0

    # OOS Recall: câu OOS thực sự được đánh là -1
    oos_correct = sum(1 for i in oos_indices if preds_idx[i] == -1)
    oos_recall  = oos_correct / len(oos_indices) if oos_indices else 0.0

    # False Rejection Rate: câu in-scope bị nhầm thành OOS
    frr = (
        sum(1 for i, lbl in enumerate(labels)
            if lbl != "oos" and preds_idx[i] == -1) / total_inscope
        if total_inscope else 0.0
    )

    print(f"\n{'─'*45}")
    print(f"  Threshold        : {threshold:.2f}")
    print(f"  In-scope Accuracy: {inscope_acc*100:.2f}%  ({correct_inscope}/{total_inscope})")
    print(f"  OOS Recall       : {oos_recall*100:.2f}%  ({oos_correct}/{len(oos_indices)})")
    print(f"  False Rejection  : {frr*100:.2f}%  (in-scope bị đánh nhầm thành OOS)")
    print(f"{'─'*45}")

    return inscope_acc, oos_recall


# ========================================================
# MAIN PIPELINE
# ========================================================
if __name__ == "__main__":

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Chọn file input ────────────────────────────────────
    TRAIN_CORPUS = INPUT_DIR / "train_corpus.txt"
    TRAIN_LABELS = INPUT_DIR / "train_labels.txt"
    VAL_CORPUS   = INPUT_DIR / "val_corpus.txt"
    VAL_LABELS   = INPUT_DIR / "val_labels.txt"


    # ── Bước 1: Đọc dữ liệu ───────────────────────────────
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

    # ── Bước 2: Vector hóa TF-IDF ─────────────────────────
    print("\n[Bước 2] Vector hóa TF-IDF...")
    vectorizer = TfIdfVectorizer()
    X_train    = vectorizer.fit_transform(train_corpus)
    X_val      = vectorizer.transform(val_corpus)
    print(f"  Ma trận train: {X_train.shape}")
    print(f"  Ma trận val  : {X_val.shape}")

    # ── Bước 3: Mã hóa nhãn ───────────────────────────────
    print("\n[Bước 3] Mã hóa nhãn...")
    label_encoder = LabelEncoder()
    y_train       = label_encoder.fit_transform(train_labels)
    print(f"  Số class: {len(label_encoder.classes)}")

    # ── Bước 4: Huấn luyện ────────────────────────────────
    print("\n[Bước 4] Huấn luyện mô hình Naive Bayes...")
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(X_train, y_train)

    # ── Bước 5: Đánh giá trên tập Validation ──────────────
    print("\n[Bước 5] Đánh giá trên tập Validation:")
    evaluate(model, vectorizer, label_encoder,
             val_corpus, val_labels, threshold=0.5)

    # ── Bước 6: Lưu mô hình ───────────────────────────────
    save_path = MODEL_DIR / "NaiveBayes_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "vectorizer"   : vectorizer,
            "label_encoder": label_encoder,
            "model"        : model,
        }, f)
    print(f"\n[Thành công] Mô hình đã lưu tại: {save_path}")
