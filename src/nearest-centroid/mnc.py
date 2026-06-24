"""
mnc.py  —  đặt tại: src/nearest-centroid/mnc.py
-----------------------------------------------
Nearest Centroid (Rocchio) cho Intent Classification, implement thuần NumPy.

Ý tưởng:
    - Mỗi lớp được tóm tắt bằng MỘT vector "nguyên mẫu" (centroid) = trung bình
      các vector TF-IDF (đã L2-normalize) của các câu thuộc lớp đó.
    - Phân loại 1 câu mới: tính cosine similarity tới cả 150 centroid, chọn lớp
      có centroid gần nhất (cosine cao nhất).

Đối lập có chủ đích với KNN (cùng dùng khoảng cách nhưng ngược nhau hoàn toàn):
    KNN      : lazy — lưu TOÀN BỘ 15.000 câu train, inference chậm, model nặng.
    Centroid : eager — chỉ lưu 150 vector tóm tắt, inference cực nhanh, model nhẹ.

OOS detection:
    confidence = cosine similarity tới centroid gần nhất. Câu lạ (OOS) xa mọi
    centroid → cosine thấp → bị từ chối. Đây là tín hiệu OOS rất tự nhiên.

Dùng chung thư viện trong src/utils/ (KHÔNG sửa → không ảnh hưởng model khác).
Định dạng pickle: {"vectorizer", "label_encoder", "model"}.
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Fix encoding tiếng Việt trên Windows PowerShell
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ── Đường dẫn (mnc.py nằm tại src/nearest-centroid/) ──────
BASE_DIR  = Path(__file__).resolve().parent.parent   # → src/
INPUT_DIR = BASE_DIR / "input"
MODEL_DIR = BASE_DIR / "model"

# Thư viện chung trong src/utils/
UTILS_DIR = BASE_DIR / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from tfidfcal import TfIdfVectorizer       # type: ignore
from labelEncode import LabelEncoder       # type: ignore
from dataio import load_split             # type: ignore


# ========================================================
# THÀNH PHẦN CỐT LÕI: NEAREST CENTROID
# ========================================================
class NearestCentroid:
    """
    Nearest Centroid tự viết bằng NumPy.

    - fit()    : tính 1 centroid (vector trung bình đã chuẩn hóa) cho mỗi lớp.
    - predict(): với mỗi câu, chọn lớp có cosine similarity cao nhất.
    - predict_with_oos(): trả -1 nếu cosine < threshold (Out-Of-Scope).
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize     # L2-normalize → khoảng cách ≈ cosine
        self.centroids = None          # (n_classes, n_features)
        self.classes   = None
        self.eps       = 1e-8

    # ── L2 normalization ──────────────────────────────────
    def _l2(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if not self.normalize:
            return X
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    # ── Học centroid cho từng lớp ─────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xn = self._l2(X)
        y  = np.asarray(y)
        self.classes = np.unique(y)
        n_classes  = len(self.classes)
        n_features = Xn.shape[1]

        print(f"Bắt đầu huấn luyện Nearest Centroid...")
        print(f"  Số mẫu       : {Xn.shape[0]}")
        print(f"  Số đặc trưng : {n_features}")
        print(f"  Số lớp       : {n_classes}")

        self.centroids = np.zeros((n_classes, n_features), dtype=np.float64)
        for idx, c in enumerate(self.classes):
            Xc = Xn[y == c]
            self.centroids[idx] = Xc.mean(axis=0)   # vector trung bình của lớp c

        # Chuẩn hóa centroid → tích vô hướng với câu (đã chuẩn hóa) = cosine
        norms = np.linalg.norm(self.centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.centroids = self.centroids / norms
        print("Huấn luyện hoàn tất!")

    # ── Cosine similarity tới mọi centroid ────────────────
    def _similarities(self, X: np.ndarray) -> np.ndarray:
        Xn = self._l2(X)
        return Xn @ self.centroids.T   # (n_samples, n_classes)

    # ── Dự đoán nhãn thô (không lọc OOS) ──────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self._similarities(X), axis=1)

    # ── Dự đoán kèm OOS detection ─────────────────────────
    def predict_with_oos(self, X: np.ndarray, threshold: float = 0.5):
        sims      = self._similarities(X)
        max_sim   = np.max(sims, axis=1)
        pred_idx  = np.argmax(sims, axis=1)
        final     = np.where(max_sim >= threshold, pred_idx, -1)
        return final, max_sim


# ========================================================
# HÀM TIỆN ÍCH: ĐÁNH GIÁ TRÊN TẬP VALIDATION
# ========================================================
def evaluate(model, vectorizer, label_encoder, corpus, labels, threshold=0.5):
    """In in-scope accuracy / OOS recall / false rejection trên tập val."""
    X = vectorizer.transform(corpus)
    preds_idx, _ = model.predict_with_oos(X, threshold=threshold)

    oos_indices   = [i for i, lbl in enumerate(labels) if lbl == "oos"]
    total_inscope = sum(1 for lbl in labels if lbl != "oos")

    correct_inscope = sum(
        1 for pred, true_lbl in zip(preds_idx, labels)
        if true_lbl != "oos" and pred != -1
        and label_encoder.index_to_label.get(pred) == true_lbl
    )
    inscope_acc = correct_inscope / total_inscope if total_inscope else 0.0

    oos_correct = sum(1 for i in oos_indices if preds_idx[i] == -1)
    oos_recall  = oos_correct / len(oos_indices) if oos_indices else 0.0

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

    # Lọc OOS khỏi train (Centroid chỉ học các nhãn in-scope)
    pairs = [(c, l) for c, l in zip(train_corpus, train_labels) if l != "oos"]
    train_corpus, train_labels = [p[0] for p in pairs], [p[1] for p in pairs]

    print(f"  Train : {len(train_corpus)} mẫu (in-scope)")
    print(f"  Val   : {len(val_corpus)} mẫu  "
          f"(trong đó oos: {sum(1 for l in val_labels if l == 'oos')})")

    # ── Bước 2: Vector hóa TF-IDF (dùng chung từ utils) ───
    print("\n[Bước 2] Vector hóa TF-IDF...")
    vectorizer = TfIdfVectorizer()
    X_train    = vectorizer.fit_transform(train_corpus)
    print(f"  Ma trận train: {X_train.shape}")

    # ── Bước 3: Mã hóa nhãn ───────────────────────────────
    print("\n[Bước 3] Mã hóa nhãn...")
    label_encoder = LabelEncoder()
    y_train       = label_encoder.fit_transform(train_labels)
    print(f"  Số class: {len(label_encoder.classes)}")

    # ── Bước 4: Huấn luyện ────────────────────────────────
    print("\n[Bước 4] Huấn luyện Nearest Centroid...")
    model = NearestCentroid(normalize=True)
    model.fit(X_train, y_train)

    # ── Bước 5: Đánh giá trên tập Validation ──────────────
    print("\n[Bước 5] Đánh giá trên tập Validation:")
    evaluate(model, vectorizer, label_encoder, val_corpus, val_labels, threshold=0.5)

    # ── Bước 6: Lưu mô hình ───────────────────────────────
    save_path = MODEL_DIR / "NearestCentroid_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "vectorizer"   : vectorizer,
            "label_encoder": label_encoder,
            "model"        : model,
        }, f)
    print(f"\n[Thành công] Mô hình đã lưu tại: {save_path}")