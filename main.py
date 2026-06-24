import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Fix encoding tiếng Việt trên Windows PowerShell (cp1258 không hỗ trợ Unicode)
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN IMPORT
# ==========================================
ROOT_DIR     = Path(__file__).resolve().parent
UTILS_DIR    = ROOT_DIR / "src" / "utils"
LOGISTIC_DIR = ROOT_DIR / "src" / "logistic-regression"
NAIVE_DIR    = ROOT_DIR / "src" / "naive-bayes"
KNN_DIR      = ROOT_DIR / "src" / "k-nearest-neighbor"
CENTROID_DIR = ROOT_DIR / "src" / "nearest-centroid"

# Thêm utils/ + 4 thư mục model vào sys.path để pickle tìm thấy class khi giải nén
for _dir in (UTILS_DIR, LOGISTIC_DIR, NAIVE_DIR, KNN_DIR, CENTROID_DIR):
    if str(_dir) not in sys.path:
        sys.path.append(str(_dir))

# Thư viện chung trong src/utils/ — nguồn TfIdfVectorizer / LabelEncoder / đường dẫn.
# (import tfidfcal & labelEncode còn để pickle.load nhận diện class của vectorizer.)
try:
    from tfidfcal import TfIdfVectorizer     # type: ignore  # noqa: F401
    from labelEncode import LabelEncoder     # type: ignore  # noqa: F401
    from dataio import MODEL_DIR, INPUT_DIR  # type: ignore
except ImportError as e:
    print(f"Lỗi Import thư viện chung từ src/utils/: {e}")
    exit(1)

try:
    from mrl import MultinomialLogisticRegression  # type: ignore  # noqa: F401
except ImportError as e:
    print(f"Lỗi Import mrl: {e}. Kiểm tra thư mục src/logistic-regression/")
    exit(1)

try:
    from mnb import MultinomialNaiveBayes  # type: ignore  # noqa: F401
except ImportError as e:
    print(f"Lỗi Import mnb: {e}. Kiểm tra thư mục src/naive-bayes/")
    exit(1)

try:
    from mknn import KNearestNeighbors  # type: ignore  # noqa: F401
except ImportError as e:
    print(f"Lỗi Import mknn: {e}. Kiểm tra thư mục src/k-nearest-neighbor/")
    exit(1)

try:
    from mnc import NearestCentroid  # type: ignore  # noqa: F401
except ImportError as e:
    print(f"Lỗi Import mnc: {e}. Kiểm tra thư mục src/nearest-centroid/")
    exit(1)

# Đăng ký module ảo để pickle không lỗi "Can't get attribute".
# Các model được train bằng cách chạy file trực tiếp (python mknn.py / mnc.py)
# nên class được pickle dưới module '__main__'. Việc import ở trên đã đưa class
# vào namespace của main.py (chính là __main__) → pickle.load tìm thấy.
import mknn  # type: ignore
import mnc   # type: ignore
sys.modules['mknn'] = mknn
sys.modules['mnc']  = mnc

# ==========================================
# 2. CÁC HÀM TIỆN ÍCH LÕI
# ==========================================
# (filename, tên hiển thị, ngưỡng OOS mặc định riêng của từng model)
MODEL_FILES = {
    "lr":  ("LogisticRegression_model.pkl", "Logistic Regression", 0.50),
    "nb":  ("NaiveBayes_model.pkl",         "Naive Bayes",         0.50),
    "knn": ("KNN_model.pkl",                "K-Nearest Neighbors", 0.30),
    "nc":  ("NearestCentroid_model.pkl",    "Nearest Centroid",    0.15),
}

# Thứ tự hiển thị khi chạy chế độ "all"
ALL_MODELS = ["lr", "nb", "knn", "nc"]

TRAIN_SCRIPTS = {
    "lr":  "logistic-regression/mrl.py",
    "nb":  "naive-bayes/mnb.py",
    "knn": "k-nearest-neighbor/mknn.py",
    "nc":  "nearest-centroid/mnc.py",
}


def default_threshold(model_type: str) -> float:
    """Ngưỡng OOS mặc định riêng cho từng model."""
    return MODEL_FILES[model_type][2]


def load_model(model_type: str = "lr"):
    """Tải mô hình đã lưu. Trả về (vectorizer, label_encoder, model, display_name)."""
    if model_type not in MODEL_FILES:
        print(f"Loại mô hình '{model_type}' không hợp lệ. Chọn: {', '.join(MODEL_FILES)}.")
        exit(1)

    filename, display_name, _ = MODEL_FILES[model_type]
    save_path = MODEL_DIR / filename

    if not save_path.exists():
        print(f"Không tìm thấy mô hình '{display_name}' tại {save_path}.")
        print(f"Bạn cần chạy train trước: python src/{TRAIN_SCRIPTS[model_type]}")
        exit(1)

    with open(save_path, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts['vectorizer'], artifacts['label_encoder'], artifacts['model'], display_name


def predict_one(item: dict, text: str):
    """Dự đoán 1 câu với 1 model đã load. Trả về (intent_text, confidence)."""
    X = item["vectorizer"].transform([text])
    preds, probs = item["model"].predict_with_oos(X, threshold=item["threshold"])
    if preds[0] == -1:
        intent = "[OUT OF SCOPE]"
    else:
        intent = item["encoder"].index_to_label.get(preds[0], "[OUT OF SCOPE]")
    return intent, float(probs[0])


def load_many(model_keys, threshold):
    """Load nhiều model. threshold=None → mỗi model dùng ngưỡng mặc định riêng."""
    loaded = []
    for m in model_keys:
        vec, enc, mod, name = load_model(m)
        loaded.append({
            "key": m,
            "vectorizer": vec,
            "encoder": enc,
            "model": mod,
            "name": name,
            "threshold": threshold if threshold is not None else default_threshold(m),
        })
    return loaded


# ==========================================
# 3. CÁC CHỨC NĂNG CỦA CLI
# ==========================================
def run_predict(text, threshold, model_type):
    """Chế độ dự đoán 1 câu (1 model, hoặc tất cả nếu model_type='all')."""
    keys = ALL_MODELS if model_type == "all" else [model_type]
    loaded = load_many(keys, threshold)
    print(f"> {text}")
    if len(loaded) > 1:
        _print_table(loaded, text)
    else:
        intent, conf = predict_one(loaded[0], text)
        print(f"  Model     : {loaded[0]['name']}")
        print(f"  Intent    : {intent}")
        print(f"  Confidence: {conf:.2f}\n")


def _print_table(loaded, text):
    """In bảng so sánh kết quả nhiều model cho cùng 1 câu."""
    line = "-" * 68
    print(line)
    print(f"| {'Model':<22} | {'Intent':<24} | {'Confidence':<10} |")
    print(line)
    for item in loaded:
        intent, conf = predict_one(item, text)
        print(f"| {item['name']:<22} | {intent:<24} | {conf:<10.2f} |")
    print(line + "\n")


def run_chat(threshold, model_type):
    """Chế độ chat liên tục. Mặc định model_type='all' → hiện cả 4 model cùng lúc."""
    keys = ALL_MODELS if model_type == "all" else [model_type]
    loaded = load_many(keys, threshold)

    names = ", ".join(item["name"] for item in loaded)
    print(f"\n=== ĐÃ VÀO CHẾ ĐỘ CHAT — {names} ===")
    thr_info = "ngưỡng mặc định riêng từng model" if threshold is None else f"ngưỡng chung = {threshold}"
    print(f"=== OOS threshold: {thr_info} | Gõ 'exit'/'quit' để thoát ===\n")

    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nĐã thoát chế độ chat.")
            break
        if text.lower() in ('exit', 'quit'):
            print("Đã thoát chế độ chat.")
            break

        if not text.strip():
            continue

        # NẾU CÓ NHIỀU MÔ HÌNH: IN DƯỚI DẠNG BẢNG NGANG
        if len(loaded) > 1:
            print("-" * 65)
            print(f"| {'Model':<20} | {'Intent':<22} | {'Confidence':<12} |")
            print("-" * 65)
            for item in loaded:
                X = item["vectorizer"].transform([text])
                preds, probs = item["model"].predict_with_oos(X, threshold=item["threshold"])

                intent = "[OUT OF SCOPE]" if preds[0] == -1 else item["encoder"].index_to_label.get(preds[0], "[UNKNOWN]")
                conf = probs[0]

                print(f"| {item['name']:<20} | {intent:<22} | {conf:<12.2f} |")
            print("-" * 65 + "\n")

        # NẾU CHỈ CÓ 1 MÔ HÌNH: IN KHỐI DỌC BÌNH THƯỜNG
        else:
            item = loaded[0]
            X = item["vectorizer"].transform([text])
            preds, probs = item["model"].predict_with_oos(X, threshold=item["threshold"])

            intent = "[OUT OF SCOPE]" if preds[0] == -1 else item["encoder"].index_to_label.get(preds[0], "[UNKNOWN]")
            conf = probs[0]

            print(f"  Model     : {item['name']}")
            print(f"  Intent    : {intent}")
            print(f"  Confidence: {conf:.2f}\n")


def run_eval(model_type):
    """Chế độ đánh giá toàn bộ test set (Metrics). model_type='all' → chạy lần lượt cả 4."""
    keys = ALL_MODELS if model_type == "all" else [model_type]
    for m in keys:
        _eval_one(m)


def _eval_one(model_type):
    TEST_CORPUS_PATH = INPUT_DIR / "test_corpus.txt"
    TEST_LABELS_PATH = INPUT_DIR / "test_labels.txt"

    vectorizer, label_encoder, model, display_name = load_model(model_type)

    with open(TEST_CORPUS_PATH, 'r', encoding='utf-8') as f:
        raw_corpus = f.readlines()
    with open(TEST_LABELS_PATH, 'r', encoding='utf-8') as f:
        raw_labels = f.readlines()

    min_len = min(len(raw_corpus), len(raw_labels))
    test_corpus = [l.strip() for l in raw_corpus[:min_len]]
    test_labels_text = [l.strip() for l in raw_labels[:min_len]]

    filtered_corpus, y_test_true, skipped = [], [], 0
    for text, lbl in zip(test_corpus, test_labels_text):
        if not text or not lbl:
            continue
        if lbl in label_encoder.label_to_index:
            filtered_corpus.append(text)
            y_test_true.append(label_encoder.label_to_index[lbl])
        else:
            skipped += 1

    y_test_true = np.array(y_test_true)
    X_test = vectorizer.transform(filtered_corpus)
    if skipped > 0:
        print(f"[{display_name}] Bỏ qua {skipped} mẫu có nhãn lạ (vd: 'oos') chưa thấy lúc train.")

    # threshold=0.0 → lấy dự đoán in-scope thô (không lọc OOS), đo phân loại thuần
    if hasattr(model, "predict_with_oos"):
        y_test_pred, _ = model.predict_with_oos(X_test, threshold=0.0)
    else:
        y_test_pred = model.predict(X_test)

    avg_method = 'macro' if len(label_encoder.classes) > 2 else 'binary'
    print("\n" + "=" * 40)
    print(f"{'KẾT QUẢ ĐÁNH GIÁ (TEST METRICS)':^40}")
    print("=" * 40)
    print(f"Mô hình          : {display_name}")
    print(f"Ngưỡng OOS       : {threshold}")
    print(f"Tổng số mẫu test : {len(y_test_true)}")
    print(f"Chế độ trung bình: {avg_method}\n")
    print(f"{'Accuracy':<15}: {accuracy_score(y_test_true, y_test_pred):.4f}")
    print(f"{'Precision':<15}: {precision_score(y_test_true, y_test_pred, average=avg_method, zero_division=0):.4f}")
    print(f"{'Recall':<15}: {recall_score(y_test_true, y_test_pred, average=avg_method, zero_division=0):.4f}")
    print(f"{'F1-Score':<15}: {f1_score(y_test_true, y_test_pred, average=avg_method, zero_division=0):.4f}")
    print("=" * 40)


# ==========================================
# 4. ENTRY POINT & PARSER CLI
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI phân loại ý định (Intent Classification)")
    subparsers = parser.add_subparsers(dest="command", help="Chọn chế độ chạy")

    CHOICES_ALL = ["lr", "nb", "knn", "nc", "all"]
    HELP_MODEL = "lr=Logistic Regression, nb=Naive Bayes, knn=K-Nearest Neighbors, nc=Nearest Centroid"

    # Command 1: predict — đoán 1 câu
    p_predict = subparsers.add_parser("predict", help="Đoán nhãn cho 1 câu")
    p_predict.add_argument("text", type=str, help="Câu cần kiểm tra (trong dấu ngoặc kép)")
    p_predict.add_argument("-t", "--threshold", type=float, default=None,
                           help="Ngưỡng OOS. Bỏ trống → dùng ngưỡng mặc định riêng từng model")
    p_predict.add_argument("-m", "--model", type=str, default="all", choices=CHOICES_ALL,
                           help=f"{HELP_MODEL}, all=tất cả (mặc định: all)")

    # Command 2: chat — chat liên tục, mặc định hiện cả 4 model
    p_chat = subparsers.add_parser("chat", help="Chat liên tục, so sánh các model")
    p_chat.add_argument("-t", "--threshold", type=float, default=None,
                        help="Ngưỡng OOS chung. Bỏ trống → mỗi model dùng ngưỡng riêng")
    p_chat.add_argument("-m", "--model", type=str, default="all", choices=CHOICES_ALL,
                        help=f"{HELP_MODEL}, all=so sánh tất cả (mặc định: all)")

    # Command 3: eval — đánh giá test set
    p_eval = subparsers.add_parser("eval", help="Đánh giá toàn bộ test_corpus.txt")
    p_eval.add_argument("-m", "--model", type=str, default="all", choices=CHOICES_ALL,
                        help=f"{HELP_MODEL}, all=đánh giá tất cả (mặc định: all)")

    args = parser.parse_args()

    if args.command == "predict":
        run_predict(args.text, args.threshold, args.model)
    elif args.command == "chat":
        run_chat(args.threshold, args.model)
    elif args.command == "eval":
        run_eval(args.model)
    else:
        parser.print_help()
