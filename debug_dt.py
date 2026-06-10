import sys
import pickle
import numpy as np
from pathlib import Path

# Giả lập đường dẫn như project thật
BASE = Path("src")
sys.path.insert(0, str(BASE / "decision-tree"))
sys.path.insert(0, str(BASE / "logistic-regression"))

import mdt

model_path = BASE / "model" / "DecisionTree_model.pkl"
with open(model_path, "rb") as f:
    arts = pickle.load(f)

vectorizer    = arts["vectorizer"]
label_encoder = arts["label_encoder"]
model         = arts["model"]

# 1. Load và chuẩn hóa dữ liệu đồng bộ ngay từ đầu
val_corpus, val_labels = mdt.load_split(
    BASE / "input" / "val_corpus.txt",
    BASE / "input" / "val_labels.txt"
)

X_val_raw = vectorizer.transform(val_corpus)
X_f = np.asarray(X_val_raw, dtype=np.float32)

# Khắc phục lỗi lệch chiều ma trận đặc trưng
if X_f.shape[1] < model.n_features_in_:
    X_f = np.hstack((X_f, np.zeros((X_f.shape[0], model.n_features_in_ - X_f.shape[1]), dtype=np.float32)))
else:
    X_f = X_f[:, :model.n_features_in_]

# Lấy thông tin từ cây mẹ trên ma trận đã chuẩn hóa
parent_preds, parent_probs, _ = model.parent_tree.predict_node_info(X_f)

print("=== PHÂN PHỐI CONFIDENCE CÂY MẸ (parent_tree) ===")
bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
for i in range(len(bins)-1):
    count = np.sum((parent_probs >= bins[i]) & (parent_probs < bins[i+1]))
    print(f"  [{bins[i]:.1f} - {bins[i+1]:.1f}): {count:5d} mẫu")

print(f"\n  Mean  : {parent_probs.mean():.4f}")
print(f"  Median: {np.median(parent_probs):.4f}")
print(f"  Min   : {parent_probs.min():.4f}")
print(f"  Max   : {parent_probs.max():.4f}")

# Cấu hình màng lọc tầng con thực tế để đối chiếu chính xác
SUB_THRESHOLD = 0.45
MIN_SAMPLES   = 4

print(f"\n=== KHẢO SÁT THRESHOLD TẦNG MẸ (Cố định Tầng Con: Thresh={SUB_THRESHOLD}, MinSamples={MIN_SAMPLES}) ===")
for pt in [0.0, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]:
    correct = 0
    total_inscope = 0
    total_oos = sum(1 for l in val_labels if l == "oos")
    oos_caught = 0
    
    for i, (p_idx, p_prob, true_lbl) in enumerate(zip(parent_preds, parent_probs, val_labels)):
        domain = model.domain_encoder.index_to_label[p_idx]
        
        # Giả lập bộ gác cổng phân cấp (Hierarchical Cascading Filter)
        is_rejected_by_parent = (domain == "oos" or p_prob < pt)
        
        if true_lbl == "oos":
            total_oos += 0 # Đã đếm ở trên, giữ cấu trúc logic của bạn
            if is_rejected_by_parent:
                oos_caught += 1
            else:
                # Nếu lọt qua mẹ, tầng con vẫn có cơ hội chặn tiếp bằng màng lọc mật độ
                if domain in model.sub_trees:
                    x_s = X_f[i].reshape(1, -1)
                    _, sub_prob, sub_samples = model.sub_trees[domain].predict_node_info(x_s)
                    if not (sub_prob[0] >= SUB_THRESHOLD and sub_samples[0] >= MIN_SAMPLES):
                        oos_caught += 1 # Con chặn thành công câu OOS lọt lưới từ mẹ
        else:
            total_inscope += 1
            if not is_rejected_by_parent: # Nếu lọt qua tầng mẹ thành công
                if domain in model.sub_trees:
                    x_s = X_f[i].reshape(1, -1)
                    sp, sub_prob, sub_samples = model.sub_trees[domain].predict_node_info(x_s)
                    
                    # Thỏa mãn đồng thời cả bộ lọc xác suất thực và cỡ mẫu của cây con
                    if sub_prob[0] >= SUB_THRESHOLD and sub_samples[0] >= MIN_SAMPLES:
                        if label_encoder.index_to_label.get(int(sp[0])) == true_lbl:
                            correct += 1

    acc = correct / total_inscope if total_inscope else 0
    oos_r = oos_caught / total_oos if total_oos else 0
    print(f"  parent_threshold={pt:.2f} → Mô phỏng thực tế: InScope Acc={acc*100:.2f}%, OOS Recall={oos_r*100:.2f}%")