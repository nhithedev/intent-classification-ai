import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN IMPORT TRÁNH LỖI
# ==========================================
ROOT_DIR     = Path(__file__).resolve().parent
LOGISTIC_DIR = ROOT_DIR / "src" / "logistic-regression"
NAIVE_DIR    = ROOT_DIR / "src" / "naive-bayes"
DECISION_DIR = ROOT_DIR / "src" / "decision-tree"

# Thêm cả ba thư mục vào sys.path để pickle có thể tìm thấy class khi giải nén
for _dir in (LOGISTIC_DIR, NAIVE_DIR, DECISION_DIR):
    if str(_dir) not in sys.path:
        sys.path.append(str(_dir))

try:
    from mrl import TfIdfVectorizer, LabelEncoder, MultinomialLogisticRegression, MODEL_DIR, INPUT_DIR # type: ignore
except ImportError as e:
    print(f"Lỗi Import mrl: {e}. Vui lòng kiểm tra thư mục src/logistic-regression/")
    exit(1)

try:
    from mnb import MultinomialNaiveBayes  # type: ignore
except ImportError as e:
    print(f"Lỗi Import mnb: {e}. Vui lòng kiểm tra thư mục src/naive-bayes/")
    exit(1)

try:
    from mdt import HierarchicalDecisionTree  # type: ignore
except ImportError as e:
    print(f"Lỗi Import mdt: {e}. Vui lòng kiểm tra thư mục src/decision-tree/")
    exit(1)

# Hỗ trợ nạp module ảo chéo để pickle không bị lỗi "AttributeError: Can't get attribute"
import mdt
sys.modules['mdt'] = mdt

# ==========================================
# 2. CÁC HÀM TIỆN ÍCH LÕI
# ==========================================
MODEL_FILES = {
    "lr": ("LogisticRegression_model.pkl", "Logistic Regression"),
    "nb": ("NaiveBayes_model.pkl",         "Naive Bayes"),
    "dt": ("DecisionTree_model.pkl",       "Decision Tree"),
}

# Ngưỡng OOS mặc định theo từng loại model
DEFAULT_THRESHOLD = {
    "lr":  0.5,
    "nb":  0.5,
    "dt":  0.35,
    "all": 0.5,
}

def load_model(model_type: str = "lr"):
    """Tải mô hình đã lưu theo loại model_type ('lr', 'nb', hoặc 'dt')."""
    if model_type not in MODEL_FILES:
        print(f"Loại mô hình '{model_type}' không hợp lệ. Chọn 'lr', 'nb', hoặc 'dt'.")
        exit(1)

    filename, display_name = MODEL_FILES[model_type]
    save_path = MODEL_DIR / filename

    if not save_path.exists():
        train_scripts = {
            "lr": "logistic-regression/mrl.py",
            "nb": "naive-bayes/mnb.py",
            "dt": "decision-tree/mdt.py"
        }
        print(f"Không tìm thấy mô hình '{display_name}' tại {save_path}.")
        print(f"Bạn cần chạy train trước: python src/{train_scripts[model_type]}")
        exit(1)

    with open(save_path, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts['vectorizer'], artifacts['label_encoder'], artifacts['model'], display_name

# ==========================================
# 3. CÁC CHỨC NĂNG CỦA CLI
# ==========================================
def run_predict(text, threshold, model_type):
    """Chế độ dự đoán 1 câu"""
    vectorizer, label_encoder, model, display_name = load_model(model_type)
    X = vectorizer.transform([text])
    preds, max_probs = model.predict_with_oos(X, threshold=threshold)

    print(f"> {text}")
    print(f"  Model     : {display_name}")

    if preds[0] == -1:
        print(f"  Intent    : [OUT OF SCOPE]")
    else:
        label_text = label_encoder.index_to_label.get(preds[0], "[OUT OF SCOPE]")
        print(f"  Intent    : {label_text}")

    print(f"  Confidence: {max_probs[0]:.2f}\n")


def run_chat(threshold, model_type):
    """Chế độ chat liên tục với model"""

    # "all" → load cả 3 mô hình để so sánh song song
    models_to_run = ["lr", "nb", "dt"] if model_type == "all" else [model_type]
    loaded_models = []

    for m in models_to_run:
        vec, enc, mod, name = load_model(m)
        loaded_models.append({
            "vectorizer": vec,
            "encoder":    enc,
            "model":      mod,
            "name":       name,
            # Mỗi model dùng threshold riêng khi chạy "all" (người dùng không truyền -t)
            "threshold":  threshold if threshold is not None else DEFAULT_THRESHOLD[m],
        })

    effective_threshold = threshold if threshold is not None else DEFAULT_THRESHOLD[model_type]
    print(f"\n=== ĐÃ VÀO CHẾ ĐỘ CHAT (Gõ 'exit' hoặc 'quit' để thoát) ===")
    print(f"=== Ngưỡng lọc OOS hiện tại: {effective_threshold} ===\n")

    while True:
        text = input("> ")
        if text.lower() in ['exit', 'quit']:
            print("Đã thoát chế độ chat.")
            break

        if not text.strip():
            continue

        # NẾU CÓ NHIỀU MÔ HÌNH: IN DƯỚI DẠNG BẢNG NGANG
        if len(loaded_models) > 1:
            print("-" * 65)
            print(f"| {'Model':<20} | {'Intent':<22} | {'Confidence':<12} |")
            print("-" * 65)
            for item in loaded_models:
                X = item["vectorizer"].transform([text])
                preds, probs = item["model"].predict_with_oos(X, threshold=item["threshold"])

                intent = "[OUT OF SCOPE]" if preds[0] == -1 else item["encoder"].index_to_label.get(preds[0], "[UNKNOWN]")
                conf = probs[0]

                print(f"| {item['name']:<20} | {intent:<22} | {conf:<12.2f} |")
            print("-" * 65 + "\n")

        # NẾU CHỈ CÓ 1 MÔ HÌNH: IN KHỐI DỌC BÌNH THƯỜNG
        else:
            item = loaded_models[0]
            X = item["vectorizer"].transform([text])
            preds, probs = item["model"].predict_with_oos(X, threshold=item["threshold"])

            intent = "[OUT OF SCOPE]" if preds[0] == -1 else item["encoder"].index_to_label.get(preds[0], "[UNKNOWN]")
            conf = probs[0]

            print(f"  Model     : {item['name']}")
            print(f"  Intent    : {intent}")
            print(f"  Confidence: {conf:.2f}\n")


def run_eval(model_type, threshold):
    """Chế độ chạy test toàn bộ file dữ liệu (Metrics)"""
    TEST_CORPUS_PATH = INPUT_DIR / "test_corpus.txt"
    TEST_LABELS_PATH = INPUT_DIR / "test_labels.txt"

    vectorizer, label_encoder, model, display_name = load_model(model_type)

    with open(TEST_CORPUS_PATH, 'r', encoding='utf-8') as f:
        raw_corpus = f.readlines()
    with open(TEST_LABELS_PATH, 'r', encoding='utf-8') as f:
        raw_labels = f.readlines()

    min_len = min(len(raw_corpus), len(raw_labels))
    test_corpus = [line.strip() for line in raw_corpus[:min_len] if line.strip()]
    test_labels_text = [line.strip() for line in raw_labels[:min_len] if line.strip()]

    print("[Tiến trình] Lọc mẫu hợp lệ & Mã hóa TF-IDF...")
    filtered_corpus = []
    y_test_true = []
    skipped = 0

    for text, lbl in zip(test_corpus, test_labels_text):
        if lbl in label_encoder.label_to_index:
            filtered_corpus.append(text)
            y_test_true.append(label_encoder.label_to_index[lbl])
        else:
            skipped += 1

    y_test_true = np.array(y_test_true)
    X_test = vectorizer.transform(filtered_corpus)

    if skipped > 0:
        print(f"⚠️ Bỏ qua {skipped} mẫu có nhãn lạ chưa xuất hiện lúc Train.")

    # Dùng đúng threshold thực tế (không hard-code 0.0) để kết quả eval phản ánh thực tế
    if hasattr(model, "predict_with_oos"):
        y_test_pred, _ = model.predict_with_oos(X_test, threshold=threshold)
    else:
        y_test_pred = model.predict(X_test)

    print("\n" + "="*40)
    print(f"{'KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (TEST METRICS)':^40}")
    print("="*40)
    avg_method = 'macro' if len(label_encoder.classes) > 2 else 'binary'
    print(f"Mô hình          : {display_name}")
    print(f"Ngưỡng OOS       : {threshold}")
    print(f"Tổng số mẫu test : {len(y_test_true)}")
    print(f"Chế độ trung bình: {avg_method}\n")

    print(f"{'Accuracy':<15}: {accuracy_score(y_test_true, y_test_pred):.4f}")
    print(f"{'Precision':<15}: {precision_score(y_test_true, y_test_pred, average=avg_method, zero_division=0):.4f}")
    print(f"{'Recall':<15}: {recall_score(y_test_true, y_test_pred, average=avg_method, zero_division=0):.4f}")
    print(f"{'F1-Score':<15}: {f1_score(y_test_true, y_test_pred, average=avg_method, zero_division=0):.4f}")
    print("="*40)

# ==========================================
# 4. ENTRY POINT & PARSER CẤU HÌNH CLI
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Tool for Intent Classification Model")
    subparsers = parser.add_subparsers(dest="command", help="Chọn chế độ chạy")

    # Command 1: Dự đoán 1 câu
    parser_predict = subparsers.add_parser("predict", help="Đoán nhãn cho 1 câu văn cụ thể")
    parser_predict.add_argument("text", type=str, help="Câu văn cần kiểm tra (đặt trong dấu ngoặc kép)")
    parser_predict.add_argument("-t", "--threshold", type=float, default=None,
                                help="Ngưỡng tự tin (OOS Threshold). Mặc định: 0.5 cho LR/NB, 0.35 cho DT")
    parser_predict.add_argument("-m", "--model", type=str, default="lr", choices=["lr", "nb", "dt"],
                                help="Chọn mô hình: lr | nb | dt  (mặc định: lr)")

    # Command 2: Chế độ Chat
    parser_chat = subparsers.add_parser("chat", help="Mở chế độ chat liên tục để test")
    parser_chat.add_argument("-t", "--threshold", type=float, default=None,
                             help="Ngưỡng tự tin (OOS Threshold). Mặc định: 0.5 cho LR/NB, 0.35 cho DT")
    parser_chat.add_argument("-m", "--model", type=str, default="all", choices=["lr", "nb", "dt", "all"],
                             help="Chọn mô hình: lr | nb | dt | all  (mặc định: all – hiện bảng so sánh)")

    # Command 3: Đánh giá mô hình
    parser_eval = subparsers.add_parser("eval", help="Chạy đánh giá toàn bộ trên tập test_corpus.txt")
    parser_eval.add_argument("-t", "--threshold", type=float, default=None,
                             help="Ngưỡng tự tin (OOS Threshold). Mặc định: 0.5 cho LR/NB, 0.35 cho DT")
    parser_eval.add_argument("-m", "--model", type=str, default="lr", choices=["lr", "nb", "dt"],
                             help="Chọn mô hình: lr | nb | dt  (mặc định: lr)")

    args = parser.parse_args()

    # Tự động gán ngưỡng threshold tối ưu nếu người dùng để trống (None)
    model_key = getattr(args, "model", "lr")
    current_threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD.get(model_key, 0.5)

    if args.command == "predict":
        run_predict(args.text, current_threshold, args.model)
    elif args.command == "chat":
        run_chat(current_threshold, args.model)
    elif args.command == "eval":
        run_eval(args.model, current_threshold)
    else:
        parser.print_help()