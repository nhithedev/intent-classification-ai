import os
import math
import numpy as np # type: ignore
import pickle
import sys
from pathlib import Path

notebook_dir = Path(os.getcwd())
preprocess_dir = notebook_dir.parent / "preprocess"
if str(preprocess_dir) not in sys.path:
    sys.path.append(str(preprocess_dir))

# 2. Import hàm process từ module data_processor
try:
    from data_processor import process
    print("--> Import module data_processor thành công!")
except ImportError as e:
    print(f"--> Thất bại: Hãy kiểm tra lại vị trí các file. Lỗi: {e}")

DATA_PATH = notebook_dir.parent.parent / "dataset" / "data" / "data_full.json"
STOPWORDS_PATH = notebook_dir.parent.parent / "dataset" / "stopwords.txt"
INPUT_DIR = notebook_dir.parent / "input"
MODEL_DIR = notebook_dir.parent / "model"



# ========================================================
# 1. COMPONENT: TF-IDF VECTORIZER
# ========================================================
class TfIdfVectorizer:
    def __init__(self):
        self.vocab = []
        self.word_to_index = {}
        self.idf_dict = {}

    def fit_transform(self, documents):
        """Học từ vựng từ tập huấn luyện và tạo luôn ma trận X"""
        tokenized_docs = [doc.lower().split() for doc in documents]
        n_docs = len(documents)
        
        # Đếm Document Frequency (DF)
        df_counts = {}
        for doc in tokenized_docs:
            for word in set(doc):
                df_counts[word] = df_counts.get(word, 0) + 1
                
        # Xây dựng từ vựng (Vocabulary)
        self.vocab = sorted(list(df_counts.keys()))
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        
        # Tính IDF cho từng từ
        self.idf_dict = {word: math.log(n_docs / df) for word, df in df_counts.items()}
        
        # Biến đổi thành Ma trận X
        return self.transform(documents)

    def transform(self, documents):
        """Dùng cho dữ liệu mới (không học thêm từ vựng mới)"""
        tokenized_docs = [doc.lower().split() for doc in documents]
        X = np.zeros((len(documents), len(self.vocab)))
        
        for i, doc in enumerate(tokenized_docs):
            # Tính TF
            tf = {}
            for word in doc:
                tf[word] = tf.get(word, 0) + 1
                
            # Ghi vào ma trận X (chỉ lấy những từ đã có trong từ vựng)
            for word, freq in tf.items():
                if word in self.word_to_index:
                    j = self.word_to_index[word]
                    X[i, j] = freq * self.idf_dict[word]
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
# 3. COMPONENT: MULTINOMIAL LOGISTIC REGRESSION
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
        self.bias = np.zeros((1, n_classes))
        y_one_hot = np.eye(n_classes)[y]
        
        print(f"Bắt đầu huấn luyện với {self.epochs} vòng lặp...")
        for epoch in range(self.epochs):
            scores = np.dot(X, self.weights) + self.bias
            probabilities = self._softmax(scores)
            error = probabilities - y_one_hot
            
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error, axis=0, keepdims=True)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if epoch % 100 == 0 or epoch == self.epochs - 1:
                loss = -np.mean(np.sum(y_one_hot * np.log(probabilities + 1e-15), axis=1))
                print(f"  - Vòng lặp {epoch:4d} | Loss: {loss:.4f}")

    def predict(self, X):
        scores = np.dot(X, self.weights) + self.bias
        probabilities = self._softmax(scores)
        return np.argmax(probabilities, axis=1)
    
    def predict_with_oos(self, X, threshold=0.5):
        """
        Dự đoán nhãn với cơ chế Out-Of-Scope (OOS).
        Trả về 2 mảng:
        - final_predictions: Nhãn dự đoán (nếu dưới threshold sẽ gán là -1)
        - max_probs: Điểm tự tin (Confidence score) tương ứng
        """
        # 1. Tính toán điểm số và xác suất
        scores = np.dot(X, self.weights) + self.bias
        probabilities = self._softmax(scores)
        
        # 2. Lấy độ tự tin cao nhất và nhãn tương ứng cho từng câu
        max_probs = np.max(probabilities, axis=1)
        predicted_idx = np.argmax(probabilities, axis=1)
        
        # 3. Áp dụng Threshold
        # np.where(điều_kiện, giá_trị_nếu_đúng, giá_trị_nếu_sai)
        # Nếu max_prob < threshold, ta gán nhãn dự đoán thành -1 (Đại diện cho OOS)
        final_predictions = np.where(max_probs >= threshold, predicted_idx, -1)
        
        return final_predictions, max_probs

# ========================================================
# HỆ THỐNG ĐIỀU HÀNH CHÍNH (MAIN PIPELINE)
# ========================================================
if __name__ == "__main__":
    
    # Đảm bảo thư mục input tồn tại
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Gọi hàm process và lấy trực tiếp đường dẫn trả về
    corpus_file, labels_file = process(
        data_path       = DATA_PATH,
        stopwords_path  = STOPWORDS_PATH,
        output_dir      = INPUT_DIR,
        tokenizer       = "whitespace", 
        min_freq        = 2,
        max_vocab_size  = 10000,
        pad_length      = 20, 
    )
    
    # 1. Đọc dữ liệu (Sử dụng đường dẫn trả về từ hàm process)
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            raw_corpus = f.readlines()
        with open(labels_file, 'r', encoding='utf-8') as f:
            raw_labels = f.readlines()
            
        # Xử lý trường hợp file bị thừa dấu enter ở cuối cùng gây lệch dòng
        min_len = min(len(raw_corpus), len(raw_labels))
        raw_corpus = raw_corpus[:min_len]
        raw_labels = raw_labels[:min_len]

        corpus = []
        labels = []
        
        # Duyệt song song từng cặp (câu văn, nhãn)
        for c_line, l_line in zip(raw_corpus, raw_labels):
            c_clean = c_line.strip()
            l_clean = l_line.strip()
            
            # Chỉ lấy những cặp mà câu văn KHÔNG bị trống
            if c_clean and l_clean:
                corpus.append(c_clean)
                labels.append(l_clean)
                
        if len(corpus) == 0:
            raise ValueError("Dữ liệu trống sau khi lọc!")
            
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        exit()

    print(f"Đã đọc {len(corpus)} mẫu dữ liệu.")
    
    # 2. Xử lý dữ liệu (Tiền xử lý)
    print("\n[Bước 1] Đang vector hóa văn bản (TF-IDF)...")
    vectorizer = TfIdfVectorizer()
    X_train = vectorizer.fit_transform(corpus)
    
    print("[Bước 2] Đang mã hóa nhãn (Label Encoding)...")
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(labels)
    
    print(f"-> Ma trận X: {X_train.shape}")
    print(f"-> Ma trận y: {y_train.shape} (Số class: {len(label_encoder.classes)})")
    
    # 3. Huấn luyện mô hình
    print("\n[Bước 3] Huấn luyện mô hình MLR...")
    model = MultinomialLogisticRegression(learning_rate=0.5, epochs=500)
    model.fit(X_train, y_train)
    
    # 4. Kiểm thử với câu văn mới (Dự đoán)
    print("\n[Bước 4] Dự đoán thử nghiệm:")
    new_sentences = [
        "set warning bank account starts running low", 
        "my credit card was declined at the store"     
    ]
    
    X_new = vectorizer.transform(new_sentences)
    predictions_idx = model.predict(X_new)
    predicted_labels = label_encoder.inverse_transform(predictions_idx)
    
    for sentence, label in zip(new_sentences, predicted_labels):
        print(f"Câu: '{sentence}'")
        print(f" -> Dự đoán: {label}\n")

    # 5. Lưu mô hình (Dùng thư viện Path để tạo đường dẫn tuyệt đối an toàn)
    model_artifacts = {
        'vectorizer': vectorizer,
        'label_encoder': label_encoder,
        'model': model
    }

    # Sinh đường dẫn file an toàn bằng pathlib
    save_path = MODEL_DIR / "model_data.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(model_artifacts, f)

    print(f"\n[Thành công] Đã lưu toàn bộ mô hình và bộ mã hóa tại:\n{save_path}")