"""
mrf.py  —  đặt tại: src/random-forest/mrf.py
--------------------------------------------
Random Forest cho Intent Classification, implement thuần NumPy.

Ý tưởng (mạch nối tiếp từ Decision Tree):
    1 cây quyết định  →  variance cao, dễ overfit.
    Random Forest     =  trung bình NHIỀU cây để giảm variance, nhờ 2 nguồn ngẫu nhiên:
        (a) Bagging         : mỗi cây học trên 1 mẫu bootstrap (lấy có hoàn lại).
        (b) Random features : mỗi lần split chỉ xét √(số chiều) đặc trưng ngẫu nhiên
                              → các cây ít tương quan → trung bình mạnh hơn.

Quyết định thiết kế cho dữ liệu text (TF-IDF thưa, 150 lớp):
    Mỗi node split theo "TỪ CÓ XUẤT HIỆN HAY KHÔNG" (giá trị > 0 hay = 0), thay vì
    quét ngưỡng liên tục. Lý do:
        - Với bag-of-words, câu hỏi phân biệt chủ yếu là "có chứa từ X không".
        - Tránh dựng mảng cumulative (n × 150 lớp) cho mỗi ngưỡng → nhanh hơn ~100×.
    Tiêu chí chọn split: giảm Gini impurity nhiều nhất.

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

# ── Đường dẫn (mrf.py nằm tại src/random-forest/) ─────────
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
# THÀNH PHẦN CỐT LÕI: RANDOM FOREST
# ========================================================
class RandomForest:
    """
    Random Forest tự viết bằng NumPy.

    Node của cây lưu dưới dạng dict (pickle gọn, không cần class phụ):
        - lá   : {"leaf": True,  "cls": <nhãn>}
        - nhánh: {"leaf": False, "feat": <chỉ số đặc trưng>,
                  "left": <node nếu từ XUẤT HIỆN>, "right": <node nếu VẮNG>, "cls": <nhãn dự phòng>}
    """

    def __init__(self, n_estimators=40, max_depth=30, min_samples_split=5,
                 max_features="sqrt", min_df=5, seed=42):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.min_df            = min_df          # lọc từ hiếm trước khi train
        self.seed              = seed
        self.trees             = []
        self.feature_idx       = None            # cột TF-IDF được giữ lại
        self.n_classes         = None
        self.max_features_     = None
        self.oob_score_        = None

    # ── Gini impurity từ vector đếm theo lớp ──────────────
    @staticmethod
    def _gini_from_counts(counts, total):
        if total == 0:
            return 0.0
        p = counts / total
        return 1.0 - np.sum(p * p)

    # ── Tìm đặc trưng split tốt nhất (present/absent) ─────
    # Nhận indices thay vì slice để tránh sao chép mảng lớn trên call stack.
    def _best_split(self, X, y, indices, feat_candidates, parent_gini, n):
        y_local = y[indices]
        counts_total = np.bincount(y_local, minlength=self.n_classes)
        best_gain, best_feat = 0.0, None
        for f in feat_candidates:
            present = X[indices, f] > 0
            n_p = int(present.sum())
            n_a = n - n_p
            if n_p == 0 or n_a == 0:
                continue                          # không tách được
            counts_p = np.bincount(y_local[present], minlength=self.n_classes)
            counts_a = counts_total - counts_p
            gini_p = self._gini_from_counts(counts_p, n_p)
            gini_a = self._gini_from_counts(counts_a, n_a)
            weighted = (n_p * gini_p + n_a * gini_a) / n
            gain = parent_gini - weighted
            if gain > best_gain:
                best_gain, best_feat = gain, f
        return best_feat, best_gain

    # ── Dựng 1 cây (đệ quy) ───────────────────────────────
    # Truyền indices thay vì X[slice] để mỗi frame chỉ cần mảng int nhỏ,
    # không nhân bản mảng X (363 MB) ở mỗi tầng đệ quy.
    def _build(self, X, y, indices, depth, rng):
        y_local = y[indices]
        counts = np.bincount(y_local, minlength=self.n_classes)
        majority = int(counts.argmax())
        n = len(indices)

        # Điều kiện dừng → lá
        if (depth >= self.max_depth or n < self.min_samples_split
                or np.count_nonzero(counts) == 1):
            return {"leaf": True, "cls": majority}

        parent_gini = self._gini_from_counts(counts, n)
        d = X.shape[1]
        k = min(self.max_features_, d)
        feat_candidates = rng.choice(d, size=k, replace=False)

        best_feat, best_gain = self._best_split(X, y, indices, feat_candidates, parent_gini, n)
        if best_feat is None or best_gain <= 0:
            return {"leaf": True, "cls": majority}

        present = X[indices, best_feat] > 0
        left  = self._build(X, y, indices[present],  depth + 1, rng)   # từ XUẤT HIỆN
        right = self._build(X, y, indices[~present], depth + 1, rng)   # từ VẮNG
        return {"leaf": False, "feat": int(best_feat),
                "left": left, "right": right, "cls": majority}

    # ── Huấn luyện rừng ───────────────────────────────────
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.int64)
        n = X.shape[0]
        self.n_classes = int(y.max()) + 1

        # Lọc đặc trưng hiếm (xuất hiện < min_df câu) → giảm chiều, train nhanh hơn
        df = (X > 0).sum(axis=0)
        self.feature_idx = np.where(df >= self.min_df)[0]
        Xr = X[:, self.feature_idx]
        d = Xr.shape[1]
        self.max_features_ = max(1, int(np.sqrt(d))) if self.max_features == "sqrt" \
            else min(self.max_features, d)

        print(f"  Số đặc trưng sau lọc (min_df={self.min_df}): {d}/{X.shape[1]}")
        print(f"  max_features mỗi split: {self.max_features_}")
        print(f"  Đang dựng {self.n_estimators} cây...")

        rng = np.random.default_rng(self.seed)
        self.trees = []
        oob_votes = np.zeros((n, self.n_classes), dtype=np.float64)

        for t in range(self.n_estimators):
            sample_idx = rng.integers(0, n, n)            # bootstrap (có hoàn lại)
            Xr_boot = Xr[sample_idx]                      # 1 bản sao duy nhất / cây
            y_boot  = y[sample_idx]
            tree = self._build(Xr_boot, y_boot, np.arange(n), depth=0, rng=rng)
            self.trees.append(tree)

            # Out-Of-Bag: các mẫu KHÔNG được chọn → đánh giá miễn phí
            oob_mask = np.ones(n, dtype=bool)
            oob_mask[sample_idx] = False
            if oob_mask.any():
                preds = self._predict_tree(tree, Xr[oob_mask])
                rows = np.where(oob_mask)[0]
                oob_votes[rows, preds] += 1
            if (t + 1) % 10 == 0:
                print(f"    ...{t + 1}/{self.n_estimators} cây")

        # OOB score: với mẫu có ít nhất 1 phiếu OOB
        voted = oob_votes.sum(axis=1) > 0
        if voted.any():
            oob_pred = oob_votes[voted].argmax(axis=1)
            self.oob_score_ = float((oob_pred == y[voted]).mean())

    # ── Dự đoán 1 cây cho cả batch (duyệt theo mask, vectorized) ──
    def _predict_tree(self, tree, X):
        n = X.shape[0]
        out = np.empty(n, dtype=np.int64)
        stack = [(tree, np.arange(n))]
        while stack:
            node, idx = stack.pop()
            if node["leaf"] or idx.size == 0:
                if idx.size:
                    out[idx] = node["cls"]
                continue
            present = X[idx, node["feat"]] > 0
            stack.append((node["left"],  idx[present]))
            stack.append((node["right"], idx[~present]))
        return out

    # ── Dự đoán kèm OOS (bỏ phiếu giữa các cây) ───────────
    def predict_with_oos(self, X, threshold=0.5):
        Xr = np.asarray(X)[:, self.feature_idx]
        T = len(self.trees)
        all_preds = np.stack([self._predict_tree(tree, Xr) for tree in self.trees])  # (T, n)

        n = Xr.shape[0]
        final = np.empty(n, dtype=np.int64)
        conf  = np.empty(n, dtype=np.float64)
        for i in range(n):
            c = np.bincount(all_preds[:, i], minlength=self.n_classes)
            w = int(c.argmax())
            final[i] = w
            conf[i]  = c[w] / T
        return np.where(conf >= threshold, final, -1), conf

    def predict(self, X):
        preds, _ = self.predict_with_oos(X, threshold=0.0)
        return preds


# ========================================================
# HÀM TIỆN ÍCH: ĐÁNH GIÁ TRÊN TẬP VALIDATION
# ========================================================
def evaluate(model, vectorizer, label_encoder, corpus, labels, threshold=0.5):
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
    frr = (sum(1 for i, lbl in enumerate(labels)
               if lbl != "oos" and preds_idx[i] == -1) / total_inscope
           if total_inscope else 0.0)

    print(f"\n{'─'*45}")
    print(f"  Threshold        : {threshold:.2f}")
    print(f"  In-scope Accuracy: {inscope_acc*100:.2f}%  ({correct_inscope}/{total_inscope})")
    print(f"  OOS Recall       : {oos_recall*100:.2f}%  ({oos_correct}/{len(oos_indices)})")
    print(f"  False Rejection  : {frr*100:.2f}%")
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

    print("=" * 50)
    print("[Bước 1] Đọc dữ liệu...")
    try:
        train_corpus, train_labels = load_split(TRAIN_CORPUS, TRAIN_LABELS)
        val_corpus,   val_labels   = load_split(VAL_CORPUS,   VAL_LABELS)
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        exit()

    # Lọc OOS khỏi train
    pairs = [(c, l) for c, l in zip(train_corpus, train_labels) if l != "oos"]
    train_corpus, train_labels = [p[0] for p in pairs], [p[1] for p in pairs]
    print(f"  Train : {len(train_corpus)} mẫu (in-scope)")
    print(f"  Val   : {len(val_corpus)} mẫu  "
          f"(trong đó oos: {sum(1 for l in val_labels if l == 'oos')})")

    print("\n[Bước 2] Vector hóa TF-IDF...")
    vectorizer = TfIdfVectorizer()
    X_train    = vectorizer.fit_transform(train_corpus)
    print(f"  Ma trận train: {X_train.shape}")

    print("\n[Bước 3] Mã hóa nhãn...")
    label_encoder = LabelEncoder()
    y_train       = label_encoder.fit_transform(train_labels)
    print(f"  Số class: {len(label_encoder.classes)}")

    print("\n[Bước 4] Huấn luyện Random Forest...")
    # Cấu hình chốt qua thực nghiệm trên val: sqrt + nhiều cây cho kết quả tốt nhất.
    # (Tăng max_features làm cây tương quan hơn → val giảm; nên giữ 'sqrt' chuẩn RF.)
    model = RandomForest(n_estimators=80, max_depth=50,
                         min_samples_split=2, max_features="sqrt", min_df=2)
    model.fit(X_train, y_train)
    if model.oob_score_ is not None:
        print(f"  OOB accuracy: {model.oob_score_*100:.2f}%")

    print("\n[Bước 5] Đánh giá trên tập Validation:")
    evaluate(model, vectorizer, label_encoder, val_corpus, val_labels, threshold=0.0)
    evaluate(model, vectorizer, label_encoder, val_corpus, val_labels, threshold=0.15)

    save_path = MODEL_DIR / "RandomForest_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "vectorizer"   : vectorizer,
            "label_encoder": label_encoder,
            "model"        : model,
        }, f)
    print(f"\n[Thành công] Mô hình đã lưu tại: {save_path}")