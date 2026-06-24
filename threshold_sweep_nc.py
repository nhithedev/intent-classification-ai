"""
threshold_sweep_nc.py
Sweep ngưỡng OOS cho Nearest Centroid trên val set.
Metrics: In-scope Accuracy, OOS Recall, False Rejection Rate.

Chạy: python threshold_sweep_nc.py
"""

import sys
import pickle
import numpy as np
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# ── sys.path ──────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent
UTILS_DIR    = ROOT_DIR / "src" / "utils"
CENTROID_DIR = ROOT_DIR / "src" / "nearest-centroid"
MODEL_DIR    = ROOT_DIR / "src" / "model"

for _d in (UTILS_DIR, CENTROID_DIR):
    if str(_d) not in sys.path:
        sys.path.append(str(_d))

from tfidfcal import TfIdfVectorizer   # type: ignore  # noqa: F401
from labelEncode import LabelEncoder   # type: ignore  # noqa: F401
from dataio import load_split, INPUT_DIR  # type: ignore
from mnc import NearestCentroid        # type: ignore  # noqa: F401

# ── Load model ────────────────────────────────────────────
pkl_path = MODEL_DIR / "NearestCentroid_model.pkl"
if not pkl_path.exists():
    print(f"Không tìm thấy {pkl_path}. Chạy train trước: python src/nearest-centroid/mnc.py")
    sys.exit(1)

with open(pkl_path, "rb") as f:
    artifacts = pickle.load(f)

vectorizer    = artifacts["vectorizer"]
label_encoder = artifacts["label_encoder"]
model         = artifacts["model"]

# ── Load val + test ───────────────────────────────────────
val_corpus,  val_labels  = load_split(INPUT_DIR / "val_corpus.txt",  INPUT_DIR / "val_labels.txt")
test_corpus, test_labels = load_split(INPUT_DIR / "test_corpus.txt", INPUT_DIR / "test_labels.txt")

X_val  = vectorizer.transform(val_corpus)
X_test = vectorizer.transform(test_corpus)

# ── Hàm đánh giá ─────────────────────────────────────────
def evaluate_threshold(X, labels, threshold):
    preds, _ = model.predict_with_oos(X, threshold=threshold)

    total_inscope = 0
    correct       = 0
    false_reject  = 0
    total_oos     = 0
    oos_detected  = 0

    for pred, lbl in zip(preds, labels):
        if lbl == "oos":
            total_oos += 1
            if pred == -1:
                oos_detected += 1
        else:
            total_inscope += 1
            if pred == -1:
                false_reject += 1
            elif label_encoder.index_to_label.get(pred) == lbl:
                correct += 1

    inscope_acc = correct / total_inscope if total_inscope else 0.0
    oos_recall  = oos_detected / total_oos if total_oos else 0.0
    frr         = false_reject / total_inscope if total_inscope else 0.0

    return {
        "inscope_acc": inscope_acc,
        "oos_recall":  oos_recall,
        "frr":         frr,
        "correct":     correct,
        "total_inscope": total_inscope,
        "oos_detected":  oos_detected,
        "total_oos":     total_oos,
    }

# ── Sweep trên val ────────────────────────────────────────
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

print("=" * 72)
print("  THRESHOLD SWEEP — Nearest Centroid  (đánh giá trên VAL SET)")
print("=" * 72)
print(f"\n  Val set: {len(val_corpus)} mẫu  "
      f"(in-scope: {sum(1 for l in val_labels if l != 'oos')}  |  "
      f"OOS: {sum(1 for l in val_labels if l == 'oos')})\n")

C = [10, 18, 16, 22, 10]  # widths: threshold, inscope_acc, oos_recall, frr, note
SEP = "─" * sum(C)

def prow(*cells):
    print("  " + "".join(f"{str(c):<{w}}" for c, w in zip(cells, C)))

print("  " + SEP)
prow("Threshold", "In-scope Acc", "OOS Recall", "False Rejection Rate", "")
print("  " + SEP)

val_results = []
for t in thresholds:
    m = evaluate_threshold(X_val, val_labels, t)
    val_results.append((t, m))

# Tìm best threshold: in-scope acc cao nhất với OOS recall >= 0.5
best_t, best_m = max(
    [(t, m) for t, m in val_results if m["oos_recall"] >= 0.50],
    key=lambda x: x[1]["inscope_acc"],
    default=val_results[0],
)

for t, m in val_results:
    tag = "◀ best" if t == best_t else ""
    prow(
        f"{t:.2f}",
        f"{m['inscope_acc']:.4f}  ({m['correct']}/{m['total_inscope']})",
        f"{m['oos_recall']:.4f}  ({m['oos_detected']}/{m['total_oos']})",
        f"{m['frr']:.4f}",
        tag,
    )

print("  " + SEP)
print(f"\n  Best (in-scope acc tối đa, OOS recall ≥ 0.50): threshold = {best_t:.2f}")

# ── Verify trên test set tại best threshold ───────────────
print("\n" + "=" * 72)
print(f"  VERIFY TRÊN TEST SET  (threshold = {best_t:.2f})")
print("=" * 72)

test_m = evaluate_threshold(X_test, test_labels, best_t)
print(f"\n  Test set: {len(test_corpus)} mẫu  "
      f"(in-scope: {sum(1 for l in test_labels if l != 'oos')}  |  "
      f"OOS: {sum(1 for l in test_labels if l == 'oos')})\n")

print(f"  In-scope Accuracy  : {test_m['inscope_acc']:.4f}  "
      f"({test_m['correct']}/{test_m['total_inscope']})")
print(f"  OOS Recall         : {test_m['oos_recall']:.4f}  "
      f"({test_m['oos_detected']}/{test_m['total_oos']})")
print(f"  False Rejection    : {test_m['frr']:.4f}\n")
