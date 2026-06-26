import sys
import pickle
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── Cấu hình đường dẫn ──
WORKSPACE_DIR = Path(r"c:\Daigaku\AAA_Chung\intent-classification-ai")
sys.path.append(str(WORKSPACE_DIR / "src" / "utils"))
sys.path.append(str(WORKSPACE_DIR / "src" / "naive-bayes"))

# Đăng ký class vào __main__ để pickle load hoạt động
from mnb import MultinomialNaiveBayes
from tfidfcal import TfIdfVectorizer
from labelEncode import LabelEncoder
from dataio import load_split

import __main__
__main__.MultinomialNaiveBayes = MultinomialNaiveBayes
__main__.TfIdfVectorizer = TfIdfVectorizer
__main__.LabelEncoder = LabelEncoder

def evaluate_threshold(preds_idx, max_probs, labels, label_encoder, threshold):
    # Áp dụng threshold
    final_preds = np.where(max_probs >= threshold, preds_idx, -1)
    
    oos_indices = [i for i, lbl in enumerate(labels) if lbl == "oos"]
    inscope_indices = [i for i, lbl in enumerate(labels) if lbl != "oos"]
    
    total_inscope = len(inscope_indices)
    total_oos = len(oos_indices)
    
    # In-scope Accuracy
    correct_inscope = 0
    for i in inscope_indices:
        pred = final_preds[i]
        true_lbl = labels[i]
        if pred != -1 and label_encoder.index_to_label.get(pred) == true_lbl:
            correct_inscope += 1
    inscope_acc = correct_inscope / total_inscope if total_inscope > 0 else 0.0
    
    # OOS Recall
    correct_oos = sum(1 for i in oos_indices if final_preds[i] == -1)
    oos_recall = correct_oos / total_oos if total_oos > 0 else 0.0
    
    # False Rejection Rate (FRR)
    false_rejected = sum(1 for i in inscope_indices if final_preds[i] == -1)
    frr = false_rejected / total_inscope if total_inscope > 0 else 0.0
    
    return inscope_acc, oos_recall, frr

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    INPUT_DIR = WORKSPACE_DIR / "src" / "input"
    MODEL_DIR = WORKSPACE_DIR / "src" / "model"
    
    # Đọc dữ liệu test (hoặc val, ở đây dùng test vì có tập OOS lớn hơn: 1000 mẫu)
    corpus, labels = load_split(INPUT_DIR / "test_corpus.txt", INPUT_DIR / "test_labels.txt")
    
    pkl_path = MODEL_DIR / "NaiveBayes_model.pkl"
    if not pkl_path.exists():
        print("Không tìm thấy file mô hình Naive Bayes!")
        return
        
    with open(pkl_path, "rb") as f:
        artifacts = pickle.load(f)
        
    vectorizer = artifacts["vectorizer"]
    label_encoder = artifacts["label_encoder"]
    model = artifacts["model"]
    
    X = vectorizer.transform(corpus)
    
    # Chạy dự đoán thô (threshold = 0.0) một lần duy nhất để lấy predicted classes và raw probabilities
    preds_idx, max_probs = model.predict_with_oos(X, threshold=0.0)
    
    # 1. Sweep linear thresholds từ 0.0 đến 0.95
    linear_thresh = np.linspace(0.0, 0.95, 50)
    
    # 2. Sweep log-spaced thresholds từ 0.95 đến 0.999999 để phóng to vùng nhạy cảm
    # 1 - threshold chạy từ 10^-1.3 (khoảng 0.95) đến 10^-6 (0.999999)
    log_one_minus = np.logspace(-1.3, -6, 50)
    log_thresh = 1.0 - log_one_minus
    
    # Gộp tất cả các ngưỡng lại và sort tăng dần
    all_thresholds = np.unique(np.concatenate([linear_thresh, log_thresh]))
    all_thresholds.sort()
    
    in_accs, oos_recs, frrs = [], [], []
    for t in all_thresholds:
        in_acc, oos_rec, frr = evaluate_threshold(preds_idx, max_probs, labels, label_encoder, t)
        in_accs.append(in_acc)
        oos_recs.append(oos_rec)
        frrs.append(frr)
        
    in_accs = np.array(in_accs) * 100
    oos_recs = np.array(oos_recs) * 100
    frrs = np.array(frrs) * 100
    
    # Thiết lập đồ thị
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Naive Bayes OOS Detection Threshold Trade-off (Test Set)", fontsize=14, fontweight='bold', y=0.98)
    
    # --- ĐỒ THỊ 1: TOÀN BỘ DẢI NGƯỠNG (0.0 -> 1.0) ---
    ax1.plot(all_thresholds, in_accs, label="In-scope Accuracy", color='#1f77b4', linewidth=2)
    ax1.plot(all_thresholds, oos_recs, label="OOS Recall (Lọc OOS)", color='#ff7f0e', linewidth=2)
    ax1.plot(all_thresholds, frrs, label="False Rejection Rate (Từ chối nhầm)", color='#d62728', linewidth=1.5, linestyle='--')
    
    # Vẽ các điểm chỉ dẫn quan trọng
    for t_val in [0.5, 0.95]:
        idx = np.abs(all_thresholds - t_val).argmin()
        ax1.axvline(x=t_val, color='gray', linestyle=':', alpha=0.7)
        ax1.scatter(t_val, in_accs[idx], color='#1f77b4', s=40)
        ax1.scatter(t_val, oos_recs[idx], color='#ff7f0e', s=40)
        ax1.text(t_val + 0.01, oos_recs[idx] - 4, f"t={t_val}\nOOS={oos_recs[idx]:.1f}%", fontsize=9)
        
    ax1.set_title("Toàn bộ dải ngưỡng (Tuyến tính)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Ngưỡng (Threshold)", fontsize=10)
    ax1.set_ylabel("Tỉ lệ (%)", fontsize=10)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-2, 102)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc="lower left")
    
    # --- ĐỒ THỊ 2: PHÓNG TO VÙNG NGƯỠNG CAO (0.9 -> 0.999999) ---
    # Sử dụng trục hoành là 1 - threshold trên thang log
    one_minus_t = 1.0 - all_thresholds
    
    # Lọc các điểm thuộc vùng threshold >= 0.90 (tức là 1 - threshold <= 0.10)
    mask = (all_thresholds >= 0.90) & (all_thresholds < 1.0)
    
    ax2.plot(one_minus_t[mask], in_accs[mask], label="In-scope Accuracy", color='#1f77b4', linewidth=2)
    ax2.plot(one_minus_t[mask], oos_recs[mask], label="OOS Recall (Lọc OOS)", color='#ff7f0e', linewidth=2)
    ax2.plot(one_minus_t[mask], frrs[mask], label="False Rejection Rate (Từ chối nhầm)", color='#d62728', linewidth=1.5, linestyle='--')
    
    # Thiết lập thang đo log cho trục X và đảo ngược trục X
    # Trục X sẽ hiển thị khoảng cách đến 1.0 (ví dụ 10^-1 = 0.9, 10^-3 = 0.999, ...)
    # Trái (cận lớn, cách xa 1.0 hơn) -> Phải (cận cực nhỏ, sát 1.0 hơn)
    ax2.set_xscale('log')
    ax2.set_xlim(1e-1, 1e-6) # Đảo ngược chiều hiển thị: từ 0.9 ở bên trái đến 0.999999 ở bên phải
    
    # Đặt nhãn cho trục hoành là giá trị ngưỡng thực tế
    x_ticks = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    x_labels = ["0.9", "0.99", "0.999", "0.9999", "0.99999", "0.999999"]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels)
    
    # Vẽ điểm nhấn cho các ngưỡng quan trọng
    for t_val in [0.95, 0.99, 0.999]:
        dist = 1.0 - t_val
        idx = np.abs(one_minus_t - dist).argmin()
        ax2.axvline(x=dist, color='gray', linestyle=':', alpha=0.7)
        ax2.scatter(dist, in_accs[idx], color='#1f77b4', s=40)
        ax2.scatter(dist, oos_recs[idx], color='#ff7f0e', s=40)
        ax2.text(dist * 0.8, oos_recs[idx] + 2, f"t={t_val}\nOOS={oos_recs[idx]:.1f}%\nIn-acc={in_accs[idx]:.1f}%", fontsize=8)

    ax2.set_title("Phóng to vùng ngưỡng cao (Logarithmic scale đến 1.0)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Ngưỡng (Threshold)", fontsize=10)
    ax2.set_ylabel("Tỉ lệ (%)", fontsize=10)
    ax2.set_ylim(-2, 102)
    ax2.grid(True, which="both", linestyle=':', alpha=0.6)
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    
    # Lưu đồ thị vào thư mục của mô hình và vào thư mục artifacts của agent để hiển thị trong chat
    output_path_repo = WORKSPACE_DIR / "src" / "naive-bayes" / "nb_threshold_tradeoff.png"
    plt.savefig(output_path_repo, dpi=150)
    plt.close()
    print(f"Đã lưu đồ thị vào repo tại: {output_path_repo}")
    
    # Sao chép vào thư mục artifacts của agent
    artifacts_dir = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\81a4705e-213a-4a47-bdd5-14bea05fffc6")
    output_path_artifacts = artifacts_dir / "nb_threshold_tradeoff.png"
    try:
        shutil.copy2(output_path_repo, output_path_artifacts)
        print(f"Đã sao chép đồ thị vào artifacts tại: {output_path_artifacts}")
    except Exception as e:
        print(f"Không thể sao chép đồ thị sang artifacts: {e}")

if __name__ == "__main__":
    main()
