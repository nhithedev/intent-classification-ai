import os
import pickle
import numpy as np
from pathlib import Path

# 1. Import các hàm tính metric trực tiếp từ scikit-learn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 2. Import các class từ mrl.py để nạp dữ liệu pkl thành công
from mrl import TfIdfVectorizer, LabelEncoder, MultinomialLogisticRegression, MODEL_DIR, INPUT_DIR

# Định nghĩa đường dẫn tới file dữ liệu test
TEST_CORPUS_PATH = INPUT_DIR / "test_corpus.txt"
TEST_LABELS_PATH = INPUT_DIR / "test_labels.txt"

def load_trained_model():
    """Tải lại các đối vật (artifacts) của mô hình đã lưu từ trước"""
    save_path = MODEL_DIR / "LogisticRegression_model.pkl"
    if not save_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file mô hình tại {save_path}. Hãy chạy train trước!")
        
    print(f"--> Đang tải mô hình từ: {save_path}")
    with open(save_path, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts['vectorizer'], artifacts['label_encoder'], artifacts['model']

def load_processed_test_data(corpus_path, labels_path):
    """Đọc và làm sạch dữ liệu test đã được xử lý từ các file txt"""
    print(f"--> Đang đọc dữ liệu test từ:\n  [Corpus]: {corpus_path}\n  [Labels]: {labels_path}")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        raw_corpus = f.readlines()
    with open(labels_path, 'r', encoding='utf-8') as f:
        raw_labels = f.readlines()
        
    min_len = min(len(raw_corpus), len(raw_labels))
    raw_corpus = raw_corpus[:min_len]
    raw_labels = raw_labels[:min_len]

    test_corpus = []
    test_labels = []
    
    for c_line, l_line in zip(raw_corpus, raw_labels):
        c_clean = c_line.strip()
        l_clean = l_line.strip()
        
        if c_clean and l_clean:
            test_corpus.append(c_clean)
            test_labels.append(l_clean)
            
    if len(test_corpus) == 0:
        raise ValueError("Dữ liệu test trống sau khi thực hiện lọc dòng!")
        
    print(f"--> Tải dữ liệu thành công. Số lượng mẫu test: {len(test_corpus)}")
    return test_corpus, test_labels

if __name__ == "__main__":
    
    # [BƯỚC 1] Đọc dữ liệu test
    try:
        test_corpus, test_labels_text = load_processed_test_data(TEST_CORPUS_PATH, TEST_LABELS_PATH)
    except Exception as e:
        print(f"[LỖI ĐỌC DỮ LIỆU]: {e}")
        exit()

    # [BƯỚC 2] Tải mô hình
    try:
        vectorizer, label_encoder, model = load_trained_model()
        print("--> Load mô hình thành công!\n")
    except Exception as e:
        print(f"[LỖI LOAD MÔ HÌNH]: {e}")
        exit()

    # [BƯỚC 3] Tiền xử lý dữ liệu test
    print("[Tiến trình] Lọc các mẫu hợp lệ và Mã hóa dữ liệu...")
    
    filtered_corpus = []
    y_test_true = []
    skipped_samples = 0
    
    # Duyệt song song cả câu text và nhãn
    for text, lbl in zip(test_corpus, test_labels_text):
        if lbl in label_encoder.label_to_index:
            # Chỉ giữ lại text và nhãn nếu nhãn đó đã có trong từ điển lúc train
            filtered_corpus.append(text)
            y_test_true.append(label_encoder.label_to_index[lbl])
        else:
            # Bỏ qua cả text lẫn nhãn nếu là nhãn lạ
            skipped_samples += 1

    y_test_true = np.array(y_test_true)
    
    # Biến đổi TF-IDF dựa trên tập corpus ĐÃ LỌC (bây giờ sẽ chỉ còn 4500 câu)
    X_test = vectorizer.transform(filtered_corpus)
    
    if skipped_samples > 0:
        print(f"⚠️ Cảnh báo: Đã loại bỏ {skipped_samples} mẫu khỏi tập test do chứa nhãn chưa từng xuất hiện lúc Train.")
    
    # Kiểm tra an toàn trước khi dự đoán
    assert X_test.shape[0] == len(y_test_true), "Lỗi: Số lượng text và label vẫn chưa khớp nhau!"
    
    # [BƯỚC 4] Dự đoán
    print("\n[Tiến trình] Mô hình đang dự đoán...")
    y_test_pred = model.predict(X_test)
    
    # [BƯỚC 5] TÍNH TOÁN VÀ IN METRICS TRỰC TIẾP
    print("\n" + "="*40)
    print(f"{'KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (TEST METRICS)':^40}")
    print("="*40)
    
    # Chọn average='macro' nếu bài toán có từ 3 nhãn trở lên, ngược lại là 'binary'
    avg_method = 'macro' if len(label_encoder.classes) > 2 else 'binary'
    print(f"Chế độ tính toán (Average): {avg_method}\n")
    
    # Tính toán trực tiếp
    acc = accuracy_score(y_test_true, y_test_pred)
    prec = precision_score(y_test_true, y_test_pred, average=avg_method, zero_division=0)
    rec = recall_score(y_test_true, y_test_pred, average=avg_method, zero_division=0)
    f1 = f1_score(y_test_true, y_test_pred, average=avg_method, zero_division=0)
    
    # In kết quả
    print(f"{'Accuracy':<15}: {acc:.4f} (Độ chính xác tổng thể)")
    print(f"{'Precision':<15}: {prec:.4f} (Độ chính xác của các dự đoán)")
    print(f"{'Recall':<15}: {rec:.4f} (Độ phủ/Độ nhạy)")
    print(f"{'F1-Score':<15}: {f1:.4f} (Trung bình hài hòa Prec & Rec)")
    print("="*40)