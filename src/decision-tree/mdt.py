import os
import math
import numpy as np # type: ignore
import pickle
import gc
from pathlib import Path
from collections import Counter

# ── Đường dẫn hệ thống ─────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent   # → src/
INPUT_DIR = BASE_DIR / "input"
MODEL_DIR = BASE_DIR / "model"

# ========================================================
# 1. COMPONENT: IMPROVED TF-IDF VECTORIZER (Sublinear & Min_DF)
# ========================================================
class TfIdfVectorizer:
    def __init__(self, min_df=3):
        self.min_df = min_df
        self.vocab = []
        self.word_to_index = {}
        self.idf_dict = {}

    def fit_transform(self, documents):
        tokenized_docs = [doc.lower().split() for doc in documents]
        n_docs = len(documents)
        
        # Tính Document Frequency (DF) để lọc nhiễu thưa
        df_counts = {}
        for doc in tokenized_docs:
            for word in set(doc):
                df_counts[word] = df_counts.get(word, 0) + 1
                
        # Chỉ giữ lại các từ xuất hiện trong ít nhất min_df văn bản để ép chiều không gian feature
        self.vocab = sorted([word for word, count in df_counts.items() if count >= self.min_df])
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        
        for word in self.vocab:
            self.idf_dict[word] = math.log(n_docs / df_counts[word])
            
        return self.transform(documents)

    def transform(self, documents):
        tokenized_docs = [doc.lower().split() for doc in documents]
        X = np.zeros((len(documents), len(self.vocab)))
        
        for i, doc in enumerate(tokenized_docs):
            tf_raw = {}
            for word in doc:
                tf_raw[word] = tf_raw.get(word, 0) + 1
            
            for word, freq in tf_raw.items():
                if word in self.word_to_index:
                    j = self.word_to_index[word]
                    # Áp dụng Sublinear Scaling: 1 + log(tf) giúp triệt tiêu nhiễu lặp từ
                    sublinear_tf = 1.0 + math.log(freq)
                    X[i, j] = sublinear_tf * self.idf_dict[word]
        return X

# ========================================================
# 2. COMPONENT: LABEL ENCODER
# ========================================================
class LabelEncoder:
    def __init__(self):
        self.classes = []
        self.label_to_index = {}
        self.index_to_label = {}

    def fit_transform(self, labels):
        self.classes = sorted(list(set(labels)))
        self.label_to_index = {label: idx for idx, label in enumerate(self.classes)}
        self.index_to_label = {idx: label for idx, label in enumerate(self.classes)}
        return np.array([self.label_to_index[label] for label in labels])
    
    def inverse_transform(self, indices):
        return [self.index_to_label[idx] for idx in indices]

# ========================================================
# 3. CORE ELEMENT: DECISION TREE ALGORITHM
# ========================================================
class DecisionNode:
    def __init__(self):
        self.feature_index = None
        self.threshold     = None
        self.left          = None
        self.right         = None
        self.value         = None   
        self.class_counts  = {}     
        self.prob          = 0.0  

class DecisionTreeClassifier:
    def __init__(self, max_depth=60, min_samples_split=2, num_classes=None):
        self.max_depth          = max_depth
        self.min_samples_split  = min_samples_split
        self.root               = None
        self.n_features_in_     = None 
        self.num_classes        = num_classes 

    def _entropy(self, y):
        n = len(y)
        if n == 0: return 0.0
        counts = Counter(y)
        entropy = 0.0
        for count in counts.values():
            p = count / n
            if p > 0: entropy -= p * math.log2(p)
        return max(0.0, entropy)

    def _information_gain(self, y, y_left, y_right):
        n = len(y)
        if n == 0 or len(y_left) == 0 or len(y_right) == 0: return 0.0
        return max(0.0, self._entropy(y) - (len(y_left)/n)*self._entropy(y_left) - (len(y_right)/n)*self._entropy(y_right))

    def _best_split(self, X, y):
        best_ig = -1.0
        best_feature, best_threshold = None, None

        active_features = np.where(X.sum(axis=0) > 0)[0]
        if len(active_features) == 0: return None, None, 0.0

        for fi in active_features:
            col = X[:, fi]
            threshold = 0.0
            left_mask, right_mask = col <= threshold, col > threshold
            if not np.any(left_mask) or not np.any(right_mask): continue
            
            ig = self._information_gain(y, y[left_mask], y[right_mask])
            if ig > best_ig:
                best_ig, best_feature, best_threshold = ig, fi, threshold
        return best_feature, best_threshold, best_ig

    def _build_tree(self, X, y, depth=0):
        node = DecisionNode()
        raw_counts = Counter(y)
        node.class_counts = {int(k): int(v) for k, v in raw_counts.items()}
        
        total_samples = len(y)
        if total_samples > 0:
            most_common_class, most_common_count = raw_counts.most_common(1)[0]
            node.value = int(most_common_class)
            
            if self.num_classes is not None:
                node.prob = float((most_common_count + 1) / (total_samples + self.num_classes))
            else:
                node.prob = float(most_common_count / total_samples)
        else:
            node.prob = 0.0

        if depth >= self.max_depth or total_samples < self.min_samples_split or self._entropy(y) == 0.0:
            return node

        best_feature, best_threshold, best_ig = self._best_split(X, y)
        if best_feature is None or best_ig <= 1e-4:
            return node

        left_mask, right_mask = X[:, best_feature] <= best_threshold, X[:, best_feature] > best_threshold
        if not np.any(left_mask) or not np.any(right_mask):
            return node

        node.feature_index = int(best_feature)
        node.threshold     = float(best_threshold)
        node.left  = self._build_tree(X[left_mask],  y[left_mask],  depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        self.n_features_in_ = X.shape[1]
        self.root = self._build_tree(X, y, depth=0)
        gc.collect()

    def _predict_one(self, x, node):
        total_samples = sum(node.class_counts.values())
        if node.feature_index is None or (node.value is not None and node.left is None and node.right is None): 
            return node.value, node.prob, total_samples
        if node.feature_index >= len(x): 
            return node.value, node.prob, total_samples
            
        if x[node.feature_index] <= node.threshold: 
            return self._predict_one(x, node.left)
        else: 
            return self._predict_one(x, node.right)

    def predict_node_info(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] < self.n_features_in_:
            X = np.hstack((X, np.zeros((X.shape[0], self.n_features_in_ - X.shape[1]), dtype=np.float32)))
        else:
            X = X[:, :self.n_features_in_]
            
        preds, probs, samples = [], [], []
        for x in X:
            val, pr, smp = self._predict_one(x, self.root)
            preds.append(val)
            probs.append(pr)
            samples.append(smp)
        return np.array(preds, dtype=np.int32), np.array(probs, dtype=np.float32), np.array(samples, dtype=np.int32)

    def predict(self, X):
        preds, _, _ = self.predict_node_info(X)
        return preds

# ========================================================
# 4. COMPONENT: HIERARCHICAL DECISION TREE PIPELINE
# ========================================================
class HierarchicalDecisionTree:
    def __init__(self):
        self.parent_tree    = None
        self.sub_trees      = {}
        self.domain_encoder = LabelEncoder()
        self.n_features_in_ = None
        self.intent_to_domain = {}  
        
    def _generate_domains_from_data(self, train_labels):
        unique_intents = sorted(list(set(train_labels)))
        mapping = {}
        for intent in unique_intents:
            if intent == "oos":
                mapping["oos"] = "oos"
                continue
            parts = intent.lower().split('_')
            if len(parts) > 1:
                domain = f"domain_{parts[0]}"
            else:
                domain = "domain_general"
            mapping[intent] = domain
        return mapping

    def fit(self, X_train_raw, train_labels, label_encoder):
        X_train_raw = np.ascontiguousarray(X_train_raw, dtype=np.float32)
        self.n_features_in_ = X_train_raw.shape[1]

        self.intent_to_domain = self._generate_domains_from_data(train_labels)
        train_domains = [self.intent_to_domain.get(lbl, "domain_general") for lbl in train_labels]
        y_train_domains = self.domain_encoder.fit_transform(train_domains)
        
        print(f"--> [AI Clustering] Tự động phân chia hệ thống thành: {len(self.domain_encoder.classes)} miền dữ liệu.")
        
        # CẢI TIẾN 1: Đẩy max_depth cây mẹ lên 40 để bóc tách domain tinh tế và chính xác hơn cho 151 lớp
        self.parent_tree = DecisionTreeClassifier(max_depth=40, min_samples_split=2, num_classes=len(self.domain_encoder.classes))
        self.parent_tree.fit(X_train_raw, y_train_domains)
        
        unique_domains = set(train_domains)
        for domain in unique_domains:
            indices = [i for i, d in enumerate(train_domains) if d == domain]
            X_sub = X_train_raw[indices]
            y_sub = np.array([label_encoder.label_to_index[train_labels[i]] for i in indices], dtype=np.int32)
            
            # CẢI TIẾN 2: Siết min_samples_split lên 10 ở cây con nhằm bóp chết các lá overfit học vẹt siêu nhỏ
            sub_tree = DecisionTreeClassifier(max_depth=60, min_samples_split=10, num_classes=None)
            sub_tree.fit(X_sub, y_sub)
            self.sub_trees[domain] = sub_tree

    def predict_with_oos(self, X, threshold=0.5):
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] < self.n_features_in_:
            X = np.hstack((X, np.zeros((X.shape[0], self.n_features_in_ - X.shape[1]), dtype=np.float32)))
        else:
            X = X[:, :self.n_features_in_]

        parent_preds, parent_probs, parent_samples = self.parent_tree.predict_node_info(X)
        final_predictions = []
        max_probs = []
        
        for i, p_idx in enumerate(parent_preds):
            domain_name = self.domain_encoder.index_to_label[p_idx]
            p_prob = parent_probs[i]
            x_sample = X[i].reshape(1, -1)
            
            # Kiểm soát màng lọc tầng mẹ dựa trên điểm ngọt tối ưu đã tìm ra
            if domain_name == "oos" or p_prob < 0.05: 
                final_predictions.append(-1)
                max_probs.append(p_prob)
            elif domain_name in self.sub_trees:
                sub_tree = self.sub_trees[domain_name]
                intent_pred, sub_prob, sub_samples = sub_tree.predict_node_info(x_sample)
                
                # Kiểm soát màng lọc tầng con chặn nhiễu Overfit hiệu quả nhất
                if sub_prob[0] >= 0.50 and sub_samples[0] >= 4:
                    final_predictions.append(int(intent_pred[0]))
                    max_probs.append(float(sub_prob[0]))
                else:
                    final_predictions.append(-1)
                    max_probs.append(float(sub_prob[0]))
            else:
                final_predictions.append(-1)
                max_probs.append(0.0)
                
        return np.array(final_predictions, dtype=np.int32), np.array(max_probs, dtype=np.float32)

# ========================================================
# HELPER: ĐỌC FILE CORPUS + LABELS
# ========================================================
def load_split(corpus_path: Path, labels_path: Path):
    with open(corpus_path, encoding="utf-8") as fc:
        raw_corpus = fc.readlines()
    with open(labels_path, encoding="utf-8") as fl:
        raw_labels = fl.readlines()

    min_len = min(len(raw_corpus), len(raw_labels))
    corpus, labels = [], []
    for c_line, l_line in zip(raw_corpus[:min_len], raw_labels[:min_len]):
        c, l = c_line.strip(), l_line.strip()
        if c and l:
            corpus.append(c)
            labels.append(l)

    if not corpus:
        raise ValueError(f"Không có dữ liệu hợp lệ trong {corpus_path.name}!")
    return corpus, labels

# ========================================================
# HELPER: ĐÁNH GIÁ TRÊN TẬP VALIDATION
# ========================================================
def evaluate(model, vectorizer, label_encoder, corpus, labels, threshold=0.5):
    X = vectorizer.transform(corpus)
    preds_idx, max_probs = model.predict_with_oos(X, threshold=threshold)

    correct_inscope = 0
    total_inscope   = 0
    for i, (pred, true_lbl) in enumerate(zip(preds_idx, labels)):
        if true_lbl == "oos":
            continue
        total_inscope += 1
        if pred != -1 and label_encoder.index_to_label.get(pred) == true_lbl:
            correct_inscope += 1

    final_acc = sum(1 for i, lbl in enumerate(labels) if (lbl == "oos" and preds_idx[i] == -1) or (lbl != "oos" and preds_idx[i] != -1 and label_encoder.index_to_label.get(preds_idx[i]) == lbl)) / len(labels)
    inscope_acc = correct_inscope / total_inscope if total_inscope else 0

    oos_indices    = [i for i, lbl in enumerate(labels) if lbl == "oos"]
    oos_recall     = (
        sum(1 for i in oos_indices if preds_idx[i] == -1) / len(oos_indices)
        if oos_indices else 0
    )

    frr = (
        sum(1 for i, lbl in enumerate(labels)
            if lbl != "oos" and preds_idx[i] == -1) / total_inscope
        if total_inscope else 0
    )

    print(f"\n{'─'*45}")
    print(f"  Threshold        : {threshold:.2f}")
    print(f"  System Accuracy  : {final_acc*100:.2f}%")
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

    print(f"  Train : {len(train_corpus)} mẫu")
    print(f"  Val   : {len(val_corpus)} mẫu  (oos: {sum(1 for l in val_labels if l == 'oos')})")

    print("\n[Bước 2] Vector hóa TF-IDF với Sublinear Scaling & Lọc min_df=3...")
    vectorizer = TfIdfVectorizer(min_df=3)
    X_train    = vectorizer.fit_transform(train_corpus)
    X_val      = vectorizer.transform(val_corpus)
    print(f"  --> Số lượng đặc trưng từ vựng giữ lại: {X_train.shape[1]} features.")

    print("\n[Bước 3] Mã hóa nhãn...")
    label_encoder = LabelEncoder()
    y_train       = label_encoder.fit_transform(train_labels)

    print("\n[Bước 4] Huấn luyện mô hình Hierarchical Decision Tree cải tiến...")
    model = HierarchicalDecisionTree()
    model.fit(X_train, train_labels, label_encoder)

    print("\n[Bước 5] Đánh giá trên tập Validation:")
    evaluate(model, vectorizer, label_encoder, val_corpus, val_labels, threshold=0.5)

    print("\n[Bước 6] Dự đoán thử nghiệm câu mẫu:")
    new_sentences = [
        "set warning bank account starts running low",
        "my credit card was declined at the store",
    ]
    X_new        = vectorizer.transform(new_sentences)
    preds_idx, _ = model.predict_with_oos(X_new, threshold=0.5)
    for sentence, idx in zip(new_sentences, preds_idx):
        label = label_encoder.index_to_label.get(idx, "OOS (out-of-scope)") if idx != -1 else "OOS (out-of-scope)"
        print(f"  Câu  : '{sentence}'")
        print(f"  → Dự đoán: {label}\n")

    print("[Bước 7] Ghi nhận và đóng gói mô hình...")
    save_path = MODEL_DIR / "DecisionTree_model.pkl"
    
    import sys
    current_module = sys.modules[__name__]
    
    TfIdfVectorizer.__module__ = "mdt"
    LabelEncoder.__module__ = "mdt"
    HierarchicalDecisionTree.__module__ = "mdt"
    DecisionNode.__module__ = "mdt"
    DecisionTreeClassifier.__module__ = "mdt"
    
    sys.modules['mdt'] = current_module
    
    with open(save_path, "wb") as f:
        pickle.dump({
            "vectorizer"   : vectorizer,
            "label_encoder": label_encoder,
            "model"        : model,
        }, f)
    print(f"[Thành công] Mô hình đã lưu tại: {save_path}")