"""
mknn.py  —  đặt tại: src/k-nearest-neighbor/mknn.py
----------------------------------------------------
K-Nearest Neighbors cho Intent Classification, implement thuần NumPy.

Nguyên tắc tích hợp (KHÔNG ảnh hưởng các thuật toán khác):
  - Dùng lại `TfIdfVectorizer`, `LabelEncoder`, `load_split` từ mrl.py
    (import, KHÔNG sửa file đó → LR / NB / DT không bị tác động).
  - L2 normalization được áp dụng BÊN TRONG class KNN (không nằm trong
    vectorizer dùng chung) → chỉ KNN hưởng lợi, không lan sang model khác.

Tối ưu accuracy:
  - L2 normalization: trên vector TF-IDF, khoảng cách Euclidean sau khi
    chuẩn hóa L2 tương đương cosine similarity → giảm ảnh hưởng curse of
    dimensionality, là đòn bẩy lớn nhất cho KNN.
  - Distance-weighted voting: hàng xóm gần đóng góp nhiều hơn hàng xóm xa.
  - Tự động tune k (và chế độ vote) trên tập validation.

Định dạng pickle (nhất quán toàn project):
    {"vectorizer": ..., "label_encoder": ..., "model": ...}
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Fix encoding tiếng Việt trên Windows PowerShell (cp1258 không hỗ trợ Unicode)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ── Đường dẫn (mknn.py nằm tại src/k-nearest-neighbor/) ───
BASE_DIR  = Path(__file__).resolve().parent.parent   # → src/
INPUT_DIR = BASE_DIR / "input"
MODEL_DIR = BASE_DIR / "model"

# Dùng thư viện chung trong src/utils/ (nguồn DUY NHẤT — không phụ thuộc vào mrl)
UTILS_DIR = BASE_DIR / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from tfidfcal import TfIdfVectorizer       # type: ignore
from labelEncode import LabelEncoder       # type: ignore
from dataio import load_split             # type: ignore


# ========================================================
# THÀNH PHẦN CỐT LÕI: K-NEAREST NEIGHBORS
# ========================================================
class KNearestNeighbors:
    """
    KNN tự viết bằng NumPy.

    - fit(): lazy learner — chỉ lưu lại training data (đã L2-normalize).
    - predict(): với mỗi câu test, tìm k hàng xóm gần nhất (Euclidean trên
      vector đã L2-normalize ≈ cosine), bỏ phiếu để chọn nhãn.
    - predict_with_oos(): trả -1 nếu confidence < threshold (Out-Of-Scope).
    """

    def __init__(self, k: int = 15, weighted: bool = True,
                 normalize: bool = True, batch_size: int = 256):
        self.k          = k
        self.weighted   = weighted      # True → distance-weighted voting
        self.normalize  = normalize     # True → L2-normalize vectors
        self.batch_size = batch_size
        self.X_train    = None
        self.y_train    = None
        self.eps        = 1e-8          # tránh chia cho 0 khi weighting

    # ── L2 normalization (nội bộ KNN) ─────────────────────
    def _l2(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if not self.normalize:
            return X
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    # ── Lưu training data ─────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train = self._l2(X)
        self.y_train = np.asarray(y)

    # ── Khoảng cách Euclidean (công thức khai triển, vectorized) ──
    #   ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b  → tận dụng BLAS
    def _distances(self, X_batch: np.ndarray) -> np.ndarray:
        sq_test  = np.sum(X_batch**2, axis=1, keepdims=True)        # (b, 1)
        sq_train = np.sum(self.X_train**2, axis=1, keepdims=True).T  # (1, n_train)
        dot      = X_batch @ self.X_train.T                          # (b, n_train)
        dist_sq  = np.maximum(sq_test + sq_train - 2 * dot, 0)       # tránh âm do float
        return np.sqrt(dist_sq)

    # ── Lấy n hàng xóm gần nhất (đã sort theo khoảng cách) ──
    #   Trả (dist, idx) shape (n_test, n_neighbors). Tính 1 lần, dùng
    #   lại cho mọi k ≤ n_neighbors khi tune.
    def kneighbors(self, X: np.ndarray, n_neighbors: int):
        X = self._l2(X)
        n = X.shape[0]
        n_neighbors = min(n_neighbors, self.X_train.shape[0])
        all_dist = np.empty((n, n_neighbors), dtype=np.float64)
        all_idx  = np.empty((n, n_neighbors), dtype=np.int64)

        for s in range(0, n, self.batch_size):
            e = min(s + self.batch_size, n)
            d = self._distances(X[s:e])                              # (b, n_train)
            # argpartition lấy n_neighbors nhỏ nhất (chưa sort) — nhanh hơn argsort toàn bộ
            part = np.argpartition(d, n_neighbors - 1, axis=1)[:, :n_neighbors]
            rows = np.arange(e - s)[:, None]
            part_d = d[rows, part]
            order  = np.argsort(part_d, axis=1)                      # sort trong n_neighbors
            all_idx[s:e]  = part[rows, order]
            all_dist[s:e] = part_d[rows, order]
        return all_dist, all_idx

    # ── Bỏ phiếu từ (dist, idx) đã có sẵn — dùng cho cả tune lẫn predict ──
    def _vote(self, dist: np.ndarray, idx: np.ndarray,
              k: int, weighted: bool):
        n = idx.shape[0]
        preds = np.zeros(n, dtype=np.int64)
        confs = np.zeros(n, dtype=np.float64)
        for i in range(n):
            nb_labels = self.y_train[idx[i, :k]]
            if weighted:
                w = 1.0 / (dist[i, :k] + self.eps)
            else:
                w = np.ones(k, dtype=np.float64)
            # gom trọng số theo từng nhãn
            tally = {}
            for lab, wt in zip(nb_labels, w):
                tally[lab] = tally.get(lab, 0.0) + wt
            total  = sum(tally.values())
            winner = max(tally, key=tally.get)
            preds[i] = winner
            confs[i] = tally[winner] / total if total > 0 else 0.0
        return preds, confs

    # ── Dự đoán kèm OOS detection ─────────────────────────
    def predict_with_oos(self, X: np.ndarray, threshold: float = 0.5):
        dist, idx = self.kneighbors(X, self.k)
        preds, confs = self._vote(dist, idx, self.k, self.weighted)
        final = np.where(confs >= threshold, preds, -1)
        return final, confs

    # ── Dự đoán nhãn thô (không lọc OOS) ──────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        final, _ = self.predict_with_oos(X, threshold=0.0)
        return final


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
# TUNE: chọn (k, weighted) cho in-scope accuracy cao nhất trên val
# ========================================================
def tune_hyperparams(model, X_val, val_labels, label_encoder, candidate_ks):
    """
    Tính k hàng xóm gần nhất MỘT LẦN (với k lớn nhất), rồi quét mọi k và
    cả 2 chế độ vote → chọn cấu hình cho in-scope accuracy cao nhất.
    Chỉ tính trên các câu in-scope (bỏ oos), dùng raw prediction (threshold=0).
    """
    max_k = max(candidate_ks)
    print(f"  Tính {max_k} hàng xóm gần nhất cho {X_val.shape[0]} câu val (1 lần)...")
    dist, idx = model.kneighbors(X_val, max_k)

    # mask + nhãn đúng cho các câu in-scope
    inscope = [i for i, lbl in enumerate(val_labels) if lbl != "oos"]
    true_idx = {i: label_encoder.label_to_index[val_labels[i]] for i in inscope}

    best = (-1.0, None, None)   # (acc, k, weighted)
    print(f"\n  {'k':>3} | {'uniform':>8} | {'weighted':>8}")
    print(f"  {'-'*3}-+-{'-'*8}-+-{'-'*8}")
    for k in candidate_ks:
        row = {}
        for weighted in (False, True):
            preds, _ = model._vote(dist, idx, k, weighted)
            correct = sum(1 for i in inscope if preds[i] == true_idx[i])
            acc = correct / len(inscope)
            row[weighted] = acc
            if acc > best[0]:
                best = (acc, k, weighted)
        print(f"  {k:>3} | {row[False]*100:>7.2f}% | {row[True]*100:>7.2f}%")

    acc, k, weighted = best
    print(f"\n  → Tốt nhất: k={k}, weighted={weighted}  (val in-scope acc {acc*100:.2f}%)")
    return k, weighted


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

    # Lọc OOS khỏi train (KNN chỉ học các nhãn in-scope)
    pairs = [(c, l) for c, l in zip(train_corpus, train_labels) if l != "oos"]
    train_corpus, train_labels = [p[0] for p in pairs], [p[1] for p in pairs]

    print(f"  Train : {len(train_corpus)} mẫu (in-scope)")
    print(f"  Val   : {len(val_corpus)} mẫu  "
          f"(trong đó oos: {sum(1 for l in val_labels if l == 'oos')})")

    # ── Bước 2: Vector hóa TF-IDF (vectorizer dùng chung, KHÔNG sửa) ──
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

    # ── Bước 4: Tune k + chế độ vote trên val (L2 norm bật) ──
    print("\n[Bước 4] Tune siêu tham số trên tập Validation...")
    tuner = KNearestNeighbors(k=15, weighted=True, normalize=True)
    tuner.fit(X_train, y_train)
    best_k, best_weighted = tune_hyperparams(
        tuner, X_val, val_labels, label_encoder,
        candidate_ks=[1, 3, 5, 7, 9, 11, 15, 21, 31],
    )

    # ── Bước 5: Huấn luyện mô hình cuối với cấu hình tốt nhất ──
    print(f"\n[Bước 5] Huấn luyện KNN cuối (k={best_k}, weighted={best_weighted})...")
    model = KNearestNeighbors(k=best_k, weighted=best_weighted, normalize=True)
    model.fit(X_train, y_train)

    # ── Bước 6: Đánh giá trên tập Validation ──────────────
    print("\n[Bước 6] Đánh giá trên tập Validation (threshold=0.5):")
    evaluate(model, vectorizer, label_encoder, val_corpus, val_labels, threshold=0.5)

    # ── Bước 7: Lưu mô hình ───────────────────────────────
    save_path = MODEL_DIR / "KNN_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "vectorizer"   : vectorizer,
            "label_encoder": label_encoder,
            "model"        : model,
        }, f)
    print(f"\n[Thành công] Mô hình đã lưu tại: {save_path}")
