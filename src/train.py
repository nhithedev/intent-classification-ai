import os
import pickle
import pandas as pd
from pathlib import Path

from preprocess.preprocess_pipeline import PipelineConfig, NLPPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def main():
    print("=" * 60)
    print(" BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI Ý ĐỊNH")
    print("=" * 60)

    # 1. Cài đặt đường dẫn
    current_dir = Path(__file__).parent 
    root_dir = current_dir.parent 
    
    dataset_path = root_dir / "dataset" / "data" / "train.csv"
    sw_path = root_dir / "dataset" / "stopwords.txt"
    model_dir = current_dir / "saved_models"
    model_dir.mkdir(exist_ok=True) # Tạo thư mục lưu model nếu chưa có

    # 2. Đọc Dataset
    if not dataset_path.exists():
        print(f"[!] LỖI: Không tìm thấy dataset tại {dataset_path}")
        return

    print(f"[*] Đang đọc dữ liệu từ: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    corpus_train = df['text'].astype(str).tolist() 
    labels_train = df['intent'].tolist()

    # 3. Khởi tạo và chạy Pipeline NLP
    config = PipelineConfig(
        stopwords_file=str(sw_path) if sw_path.exists() else None,
        tokenizer_method="whitespace",  
        pad_length=15,                  
        min_freq=2,         
        max_vocab_size=5000             
    )
    
    pipeline = NLPPipeline(config)
    print("[*] Đang xử lý NLP và trích xuất đặc trưng...")
    processed_samples = pipeline.fit_transform(corpus_train)
    _, X_train = pipeline.to_tfidf_matrix(processed_samples)
    y_train = labels_train

    # 4. Huấn luyện Logistic Regression
    print("[*] Đang huấn luyện mô hình Logistic Regression...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # In báo cáo kết quả
    print("\nBÁO CÁO TRÊN TẬP HUẤN LUYỆN:")
    print(classification_report(y_train, model.predict(X_train)))

    # 5. LƯU MÔ HÌNH VÀ PIPELINE
    pipeline_path = model_dir / "nlp_pipeline.pkl"
    model_path = model_dir / "logistic_model.pkl"

    # Lưu toàn bộ object pipeline (để giữ lại cả Vocab và bộ đếm IDF)
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)
        
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n[+] Đã lưu Pipeline tại: {pipeline_path}")
    print(f"[+] Đã lưu Mô hình tại: {model_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()