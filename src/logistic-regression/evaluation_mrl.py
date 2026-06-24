import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold

# ========================================================
# 1. CÁC COMPONENT TIỀN XỬ LÝ
# ========================================================
class TfIdfVectorizer:
    def __init__(self):
        self.vocab = []
        self.word_to_index = {}
        self.idf_dict = {}

    def fit_transform(self, documents):
        tokenized_docs = [doc.lower().split() for doc in documents]
        n_docs = len(documents)
        
        df_counts = {}
        for doc in tokenized_docs:
            for word in set(doc):
                df_counts[word] = df_counts.get(word, 0) + 1
                
        self.vocab = sorted(list(df_counts.keys()))
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.idf_dict = {word: math.log(n_docs / df) for word, df in df_counts.items()}
        
        return self.transform(documents)

    def transform(self, documents):
        tokenized_docs = [doc.lower().split() for doc in documents]
        X = np.zeros((len(documents), len(self.vocab)))
        
        for i, doc in enumerate(tokenized_docs):
            tf = {}
            for word in doc:
                tf[word] = tf.get(word, 0) + 1
            for word, freq in tf.items():
                if word in self.word_to_index:
                    j = self.word_to_index[word]
                    X[i, j] = freq * self.idf_dict[word]
        return X

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
    
    def transform(self, labels):
        return np.array([self.label_to_index.get(label, -1) for label in labels])
    
    def inverse_transform(self, indices):
        return [self.index_to_label[idx] for idx in indices]

# ========================================================
# 2. LOGISTIC REGRESSION CẢI TIẾN
# ========================================================
class MultinomialLogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000, l2_penalty=0.01):
        self.lr = learning_rate
        self.epochs = epochs
        self.l2_penalty = l2_penalty
        self.weights = None
        self.bias = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(y_train))
        
        self.weights = np.zeros((n_features, n_classes))
        self.bias    = np.zeros((1, n_classes))
        y_train_one_hot = np.eye(n_classes)[y_train]
        
        if y_val is not None:
            y_val_one_hot = np.eye(n_classes)[y_val]

        train_loss_history = []
        val_loss_history = []
        
        for epoch in range(self.epochs):
            scores_train = np.dot(X_train, self.weights) + self.bias
            probs_train  = self._softmax(scores_train)
            error        = probs_train - y_train_one_hot
            
            # Tính Loss với L2 Regularization Cost
            l2_cost = (self.l2_penalty / (2 * n_samples)) * np.sum(self.weights ** 2)
            train_loss = -np.mean(np.sum(y_train_one_hot * np.log(probs_train + 1e-15), axis=1)) + l2_cost
            train_loss_history.append(train_loss)
            
            if X_val is not None and y_val is not None:
                scores_val = np.dot(X_val, self.weights) + self.bias
                probs_val  = self._softmax(scores_val)
                val_loss   = -np.mean(np.sum(y_val_one_hot * np.log(probs_val + 1e-15), axis=1)) + l2_cost
                val_loss_history.append(val_loss)
            
            dw = (1 / n_samples) * np.dot(X_train.T, error)
            db = (1 / n_samples) * np.sum(error, axis=0, keepdims=True)
            
            # Đạo hàm của L2 Penalty
            dw += (self.l2_penalty / n_samples) * self.weights
            
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

        return train_loss_history, val_loss_history

    def predict(self, X):
        scores = np.dot(X, self.weights) + self.bias
        return np.argmax(self._softmax(scores), axis=1)

# ========================================================
# 3. HELPER: ĐỌC DỮ LIỆU
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
    return np.array(corpus), np.array(labels)

# ========================================================
# 4. CHỨC NĂNG 1: VẼ ĐƯỜNG CONG HỌC TẬP (LEARNING CURVE)
# ========================================================
def plot_learning_curve(X_train, y_train, X_val, y_val, learning_rate=0.5, epochs=500):
    print(f"\n[Learning Curve] Đang huấn luyện với {epochs} vòng lặp...")
    model = MultinomialLogisticRegression(learning_rate=learning_rate, epochs=epochs)
    
    train_losses, val_losses = model.fit(X_train, y_train, X_val, y_val)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(range(epochs), val_losses, label='Validation Loss', color='red', linewidth=2)
    
    plt.title('Đường Cong Học Tập (Learning Curve) - Logistic Regression', fontsize=14)
    plt.xlabel('Số vòng huấn luyện (Epochs)', fontsize=12)
    plt.ylabel('Sai số (Cross-Entropy Loss)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# ========================================================
# 5. CHỨC NĂNG 2: K-FOLD CROSS VALIDATION
# ========================================================
def k_fold_cross_validation(corpus, labels, k=5, learning_rate=0.5, epochs=500):
    print(f"\n[K-Fold Cross Validation] Đang thực hiện {k}-Fold trên bộ dữ liệu Train gốc ({len(corpus)} mẫu)...")
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_accuracies = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(corpus)):
        fold_corpus_train, fold_corpus_val = corpus[train_index], corpus[val_index]
        fold_labels_train, fold_labels_val = labels[train_index], labels[val_index]
        
        # Khởi tạo vectorizer mới cho mỗi fold để tránh rò rỉ dữ liệu
        vectorizer = TfIdfVectorizer()
        label_encoder = LabelEncoder()
        
        X_train_fold = vectorizer.fit_transform(fold_corpus_train)
        X_val_fold   = vectorizer.transform(fold_corpus_val)
        
        y_train_fold = label_encoder.fit_transform(fold_labels_train)
        y_val_fold   = label_encoder.transform(fold_labels_val)
        
        valid_indices = y_val_fold != -1
        X_val_fold = X_val_fold[valid_indices]
        y_val_fold = y_val_fold[valid_indices]
        
        model = MultinomialLogisticRegression(learning_rate=learning_rate, epochs=epochs)
        model.fit(X_train_fold, y_train_fold)
        
        preds = model.predict(X_val_fold)
        accuracy = np.mean(preds == y_val_fold) * 100 if len(y_val_fold) > 0 else 0.0
        fold_accuracies.append(accuracy)
        
        print(f"  - Fold {fold + 1}: Accuracy = {accuracy:.2f}%")
        
    print(f"\n=> ĐỘ CHÍNH XÁC TRUNG BÌNH ({k}-Fold): {np.mean(fold_accuracies):.2f}% (±{np.std(fold_accuracies):.2f}%)")

# ========================================================
# HỆ THỐNG ĐIỀU HÀNH CHÍNH
# ========================================================
if __name__ == "__main__":
    BASE_DIR  = Path(__file__).resolve().parent.parent
    INPUT_DIR = BASE_DIR / "input"
    
    # Chỉ sử dụng dữ liệu gốc của bạn
    TRAIN_CORPUS = INPUT_DIR / "train_corpus.txt"
    TRAIN_LABELS = INPUT_DIR / "train_labels.txt"
    VAL_CORPUS   = INPUT_DIR / "val_corpus.txt"
    VAL_LABELS   = INPUT_DIR / "val_labels.txt"

    print("=" * 50)
    print("CHỌN CHỨC NĂNG:")
    print("1. Xem biểu đồ Đường Cong Học Tập (Learning Curve)")
    print("2. Đánh giá mô hình bằng K-Fold Cross Validation trên tập Train")
    choice = input("Nhập lựa chọn của bạn (1 hoặc 2): ").strip()
    
    if choice == "1":
        try:
            train_corpus, train_labels = load_split(TRAIN_CORPUS, TRAIN_LABELS)
            val_corpus, val_labels     = load_split(VAL_CORPUS, VAL_LABELS)
            
            valid_val_idx = [i for i, lbl in enumerate(val_labels) if lbl != "oos"]
            val_corpus = val_corpus[valid_val_idx]
            val_labels = val_labels[valid_val_idx]
            
            vectorizer = TfIdfVectorizer()
            label_encoder = LabelEncoder()
            
            X_train = vectorizer.fit_transform(train_corpus)
            y_train = label_encoder.fit_transform(train_labels)
            
            X_val = vectorizer.transform(val_corpus)
            y_val = label_encoder.transform(val_labels)
            
            plot_learning_curve(X_train, y_train, X_val, y_val, learning_rate=0.5, epochs=500)
        except FileNotFoundError:
             print("Lỗi: Không tìm thấy file trong thư mục input/")
             
    elif choice == "2":
        try:
            print("[*] Đang nạp dữ liệu từ tập Train gốc...")
            train_corpus, train_labels = load_split(TRAIN_CORPUS, TRAIN_LABELS)
            # Áp dụng trực tiếp K-Fold lên tập Train
            k_fold_cross_validation(train_corpus, train_labels, k=5, learning_rate=0.5, epochs=500)
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file Train tại {TRAIN_CORPUS}. Hãy đảm bảo bạn đã có tập dữ liệu gốc.")
    else:
        print("Lựa chọn không hợp lệ.")