import os
import pickle
import numpy as np
from pathlib import Path
import sys

# 1. Import các hàm tính metric trực tiếp từ scikit-learn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 2. Thiết lập hệ thống đường dẫn để nạp module mdt chính xác
BASE_DIR = Path(__file__).resolve().parent.parent   # → src/
DT_DIR   = BASE_DIR / "decision-tree"
if str(DT_DIR) not in sys.path:
    sys.path.append(str(DT_DIR))

# IMPORT TƯỜNG MINH CÁC LỚP ĐỂ PICKLE ĐỊNH TUYẾN NAMESPACE
import mdt
from mdt import TfIdfVectorizer, LabelEncoder, HierarchicalDecisionTree, DecisionNode, DecisionTreeClassifier

# Thừa hưởng đường dẫn chung từ module mdt phân cấp
INPUT_DIR = mdt.INPUT_DIR
MODEL_DIR = mdt.MODEL_DIR

# Định nghĩa đường dẫn tới file dữ liệu test
TEST_CORPUS_PATH = INPUT_DIR / "test_corpus.txt"
TEST_LABELS_PATH = INPUT_DIR / "test_labels.txt"

def load_trained_model():
    """Tải lại các đối vật (artifacts) của mô hình đã lưu từ trước"""
    save_path = MODEL_DIR / "DecisionTree_model.pkl"
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
        
    print(f"--> Tải dữ liệu thành công. Số lượng mẫu test ban đầu: {len(test_corpus)}")
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

    # [BƯỚC 3] Tiền xử lý dữ liệu test & Ánh xạ nhãn OOS thực tế
    print("[Tiến trình] Lọc các mẫu hợp lệ và Mã hóa dữ liệu...")
    
    filtered_corpus = []
    y_test_true = []
    skipped_samples = 0
    
    for text, lbl in zip(test_corpus, test_labels_text):
        if lbl.lower() == "oos":
            # Nhãn oos thực tế trong tệp được ánh xạ trực tiếp thành nhãn số -1 nhằm đồng bộ với mô hình
            filtered_corpus.append(text)
            y_test_true.append(-1)
        elif lbl in label_encoder.label_to_index:
            # Câu thuộc phạm vi giải quyết (In-scope)
            filtered_corpus.append(text)
            y_test_true.append(label_encoder.label_to_index[lbl])
        else:
            # Nhãn lỗi/lạ không nằm trong danh mục hệ thống
            skipped_samples += 1

    y_test_true = np.array(y_test_true)
    
    # Biến đổi TF-IDF dựa trên tập dữ liệu đã đồng bộ nhãn
    X_test = vectorizer.transform(filtered_corpus)
    
    if skipped_samples > 0:
        print(f"⚠️ Cảnh báo: Đã loại bỏ {skipped_samples} mẫu do chứa nhãn lạ cấu trúc.")
    
    # Kiểm tra an toàn trước khi dự đoán
    assert X_test.shape[0] == len(y_test_true), "Lỗi: Số lượng text và label vẫn chưa khớp nhau!"
    
    # [BƯỚC 4] Dự đoán với Ngưỡng lọc OOS chuẩn của hệ thống phân cấp
    print("\n[Tiến trình] Mô hình đang tiến hành dự đoán...")
    
    TEST_THRESHOLD = 0.50  # Đồng bộ ngưỡng lọc chuẩn với bước validation
    if hasattr(model, "predict_with_oos"):
        y_test_pred, _ = model.predict_with_oos(X_test, threshold=TEST_THRESHOLD)
    else:
        y_test_pred = model.predict(X_test)
    
    # [BƯỚC 5] TÍNH TOÁN VÀ IN METRICS TRỰC TIẾP
    print("\n" + "="*50)
    print(f"{'KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (TEST METRICS)':^50}")
    print("="*50)
    
    # Bài toán phân lớp nhiều nhãn (gồm các Intent + lớp OOS) -> Dùng macro
    print(f"Chế độ tính toán (Average): macro\n")
    print(f"Ngưỡng đánh giá OOS thực tế (Threshold): {TEST_THRESHOLD}\n")
    
    # Tính toán trực tiếp chỉ số hiệu năng hệ thống
    acc  = accuracy_score(y_test_true, y_test_pred)
    prec = precision_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    rec  = recall_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    f1   = f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    
    # Tính toán bổ sung chỉ số OOS Recall cụ thể trên tập kiểm thử độc lập
    oos_true_mask = (y_test_true == -1)
    total_oos_samples = np.sum(oos_true_mask)
    if total_oos_samples > 0:
        correct_oos_preds = np.sum((y_test_pred == -1) & oos_true_mask)
        oos_recall = correct_oos_preds / total_oos_samples
    else:
        oos_recall = 0.0

    # In kết quả chuẩn biểu mẫu đánh giá kiểm thử độc lập
    print(f"{'Accuracy':<18}: {acc:.4f} (Độ chính xác tổng thể toàn hệ thống)")
    print(f"{'Precision':<18}: {prec:.4f} (Độ chính xác của các dự đoán phát ra)")
    print(f"{'Recall':<18}: {rec:.4f} (Độ phủ/Độ nhạy phân lớp)")
    print(f"{'F1-Score':<18}: {f1:.4f} (Trung bình hài hòa Prec & Rec)")
    print(f"{'OOS Recall':<18}: {oos_recall*100:.2f}% ({total_oos_samples} mẫu kiểm thử thực tế)")
    print("="*50)