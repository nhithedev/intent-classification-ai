import math
import re
import numpy as np # type: ignore

# ========================================================
# 0. PREPROCESSING
# ========================================================
def clean(text: str) -> str:
    """
    Chuẩn hóa văn bản trước khi tokenize:
      1. Lowercase
      2. Xóa ký tự đặc biệt (giữ lại chữ cái, chữ số, khoảng trắng)
      3. Chuẩn hóa khoảng trắng
    Dùng chung cho cả train data lẫn chat input.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ========================================================
# 1. COMPONENT: TF-IDF VECTORIZER
# ========================================================
class TfIdfVectorizer:
    def __init__(self, min_df=2):
        self.min_df = min_df
        self.vocab = []
        self.word_to_index = {}
        self.idf_dict = {}

    def fit_transform(self, documents):
        """Học từ vựng từ tập huấn luyện và tạo luôn ma trận X"""
        tokenized_docs = [clean(doc).split() for doc in documents]
        n_docs = len(documents)
        
        # Đếm Document Frequency (DF)
        df_counts = {}
        for doc in tokenized_docs:
            for word in set(doc):
                df_counts[word] = df_counts.get(word, 0) + 1
                
        # Xây dựng từ vựng (Vocabulary) lọc theo min_df
        self.vocab = sorted([word for word, count in df_counts.items() if count >= self.min_df])
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        
        # Tính IDF cho từng từ trong vocab
        self.idf_dict = {word: math.log(n_docs / df_counts[word]) for word in self.vocab}
        
        # Biến đổi thành Ma trận X
        return self.transform(documents)

    def transform(self, documents):
        """Dùng cho dữ liệu mới (không học thêm từ vựng mới)"""
        tokenized_docs = [clean(doc).split() for doc in documents]
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