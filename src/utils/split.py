"""
split_data.py  —  đặt tại: src/utils/split_data.py
----------------------------------------------------
Đọc data_full.json và tách thành các file corpus/labels
cho train, test, val — sẵn sàng dùng với mrl.py.

Cấu trúc output (src/input/):
    train_corpus.txt  |  train_labels.txt   →  15 000 mẫu in-scope
    test_corpus.txt   |  test_labels.txt    →   5 500 mẫu (in-scope + oos)
    val_corpus.txt    |  val_labels.txt     →   3 100 mẫu (in-scope + oos)
"""

import json
from pathlib import Path

# ── Đường dẫn (tính từ vị trí file này: src/utils/) ────────
ROOT_DIR   = Path(__file__).resolve().parents[1]   # → src/
DATA_PATH  = ROOT_DIR.parents[0] / "dataset" / "data" / "data_full.json"
OUTPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Đọc dữ liệu ────────────────────────────────────────────
print(f"Đọc dữ liệu từ: {DATA_PATH}")
with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

print("\nThống kê các split trong file gốc:")
for split, samples in data.items():
    print(f"  {split:12s}: {len(samples):>5} mẫu")

# ── Ghép các split ──────────────────────────────────────────
# Train: chỉ in-scope (không lẫn oos vào train)
train_samples = data["train"]

# Test: in-scope + oos để kiểm tra cả predict_with_oos()
test_samples  = data["test"] + data["oos_test"]

# Val: in-scope + oos để tune threshold OOS
val_samples   = data["val"] + data["oos_val"]

# ── Ghi file ────────────────────────────────────────────────
def write_split(samples, corpus_path: Path, labels_path: Path):
    with open(corpus_path, "w", encoding="utf-8") as fc, \
         open(labels_path, "w", encoding="utf-8") as fl:
        for text, label in samples:
            fc.write(text.strip() + "\n")
            fl.write(label.strip() + "\n")
    print(f"  ✓ {len(samples):>5} mẫu  →  {corpus_path.name}  |  {labels_path.name}")

print(f"\nĐang ghi file vào: {OUTPUT_DIR}")
write_split(train_samples,
            OUTPUT_DIR / "train_corpus.txt",
            OUTPUT_DIR / "train_labels.txt")

write_split(test_samples,
            OUTPUT_DIR / "test_corpus.txt",
            OUTPUT_DIR / "test_labels.txt")

write_split(val_samples,
            OUTPUT_DIR / "val_corpus.txt",
            OUTPUT_DIR / "val_labels.txt")

# ── Thống kê nhanh ──────────────────────────────────────────
from collections import Counter
label_counts  = Counter(label for _, label in train_samples)
oos_in_test   = sum(1 for _, label in test_samples if label == "oos")

print(f"\nSố class trong train : {len(label_counts)}")
print(f"Top 5 nhãn phổ biến  : {label_counts.most_common(5)}")
print(f"\nTest set             : {len(test_samples)} mẫu "
      f"({len(test_samples) - oos_in_test} in-scope  +  {oos_in_test} oos)")
print("\n✓ Hoàn tất! Các file đã sẵn sàng trong src/input/")