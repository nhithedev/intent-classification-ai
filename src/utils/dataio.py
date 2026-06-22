"""
dataio.py  —  đặt tại: src/utils/dataio.py
------------------------------------------
Tiện ích I/O dùng chung cho mọi model: đường dẫn chuẩn + đọc file corpus/labels.

Đây là một phần của "thư viện chung" trong src/utils/:
    - tfidfcal.py    → TfIdfVectorizer
    - labelEncode.py → LabelEncoder
    - dataio.py      → load_split, INPUT_DIR, MODEL_DIR   (file này)

Cả 4 model (mrl, mnb, mdt, mknn) import từ đây thay vì phụ thuộc lẫn nhau.
"""

from pathlib import Path

# ── Đường dẫn chuẩn (dataio.py nằm tại src/utils/) ─────────
BASE_DIR  = Path(__file__).resolve().parents[1]   # → src/
INPUT_DIR = BASE_DIR / "input"
MODEL_DIR = BASE_DIR / "model"


def load_split(corpus_path: Path, labels_path: Path):
    """Đọc và làm sạch một cặp file corpus/labels, trả về (corpus, labels)."""
    with open(corpus_path, encoding="utf-8") as fc:
        raw_corpus = fc.readlines()
    with open(labels_path, encoding="utf-8") as fl:
        raw_labels = fl.readlines()

    # Căn độ dài phòng trường hợp file lệch dòng
    min_len = min(len(raw_corpus), len(raw_labels))

    corpus, labels = [], []
    for c_line, l_line in zip(raw_corpus[:min_len], raw_labels[:min_len]):
        c, l = c_line.strip(), l_line.strip()
        if c and l:
            corpus.append(c)
            labels.append(l)

    if not corpus:
        raise ValueError(f"Không có dữ liệu hợp lệ trong {corpus_path.name}!")
    return corpus, labels