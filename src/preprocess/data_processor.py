"""
data_processor.py
-----------------
Module xử lý NLP độc lập.

Gọi từ dòng lệnh:
    python data_processor.py --data data.json --stopwords stopwords.txt

Gọi từ Python:
    from data_processor import process

    corpus_path, label_path = process(
        data_path       = "data.json",
        stopwords_path  = "stopwords.txt",   # None nếu không có
        output_dir      = "output",           # mặc định "output"
        tokenizer       = "whitespace",       # whitespace | ngram | subword | syllable
        min_freq        = 2,
        max_vocab_size  = 10_000,
        pad_length      = 20,
    )

Đầu ra:
    output/
    ├── corpus.txt      — mỗi dòng là chuỗi token đã xử lý (join bằng space)
    └── labels.txt      — mỗi dòng là nhãn tương ứng
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════
# Import pipeline
# ══════════════════════════════════════════════════════════════

try:
    from preprocess_pipeline import NLPPipeline, PipelineConfig
except ImportError:
    sys.exit(
        "[data_processor] KHÔNG tìm thấy 'preprocess_pipeline.py'.\n"
        "Đặt file này cùng thư mục với preprocess_pipeline.py."
    )


# ══════════════════════════════════════════════════════════════
# Load JSON
# ══════════════════════════════════════════════════════════════

def _load_json(data_path: str | Path) -> tuple[list[str], list[str]]:
    """
    Đọc file JSON và trả về (texts, labels).

    Hỗ trợ các định dạng:
        A) { "split": [[text, label], ...], ... }   — nhiều splits, gộp tất cả
        B) [[text, label], ...]                      — list of pairs trực tiếp
        C) [{"text": ..., "label": ...}, ...]        — list of dicts
        D) {"text": [...], "label": [...]}           — dict of lists
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    texts: list[str] = []
    labels: list[str] = []

    def _extract_pairs(records):
        """Trích xuất (text, label) từ một records block."""
        t, l = [], []
        if isinstance(records, list):
            for item in records:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    t.append(str(item[0]).strip())
                    l.append(str(item[1]).strip())
                elif isinstance(item, dict):
                    t.append(str(item.get("text", "")).strip())
                    l.append(str(item.get("label", "")).strip())
        elif isinstance(records, dict):
            txt_key = next((k for k in ("text", "texts", "sentence") if k in records), None)
            lbl_key = next((k for k in ("label", "labels", "intent") if k in records), None)
            if txt_key and lbl_key:
                t = [str(x).strip() for x in records[txt_key]]
                l = [str(x).strip() for x in records[lbl_key]]
        return t, l

    # Định dạng A: dict of splits
    if isinstance(raw, dict):
        # Thử xem đây là dict of splits hay dict of lists
        first_val = next(iter(raw.values()))
        if isinstance(first_val, (list,)) and len(first_val) > 0 and isinstance(first_val[0], (list, tuple, dict)):
            # dict of splits
            for split_name, records in raw.items():
                t, l = _extract_pairs(records)
                texts  += t
                labels += l
                print(f"  Split '{split_name}': {len(t):,} mẫu")
        else:
            # dict of lists (định dạng D)
            texts, labels = _extract_pairs(raw)
    else:
        # Định dạng B hoặc C
        texts, labels = _extract_pairs(raw)

    if not texts:
        raise ValueError("Không đọc được dữ liệu từ file JSON. Kiểm tra lại định dạng.")

    print(f"  Tổng: {len(texts):,} mẫu  |  "
          f"{len(set(labels))} nhãn unique: {sorted(set(labels))[:5]}"
          + (" ..." if len(set(labels)) > 5 else ""))

    return texts, labels


# ══════════════════════════════════════════════════════════════
# Hàm chính
# ══════════════════════════════════════════════════════════════

def process(
    data_path      : str | Path,
    stopwords_path : Optional[str | Path] = None,
    output_dir     : str | Path = "output",
    tokenizer      : str = "whitespace",
    min_freq       : int = 2,
    max_vocab_size : Optional[int] = 10_000,
    pad_length     : Optional[int] = None,
) -> tuple[Path, Path]:
    """
    Xử lý file data JSON qua NLP pipeline, xuất corpus.txt và labels.txt.

    Args:
        data_path      : Đường dẫn tới file .json chứa dữ liệu.
        stopwords_path : Đường dẫn tới file stopwords.txt (None = bỏ qua).
        output_dir     : Thư mục lưu kết quả (tạo tự động nếu chưa có).
        tokenizer      : Phương thức tokenize: 'whitespace' | 'ngram' | 'subword' | 'syllable'.
        min_freq       : Tần suất tối thiểu để từ vào vocab (mặc định 2).
        max_vocab_size : Giới hạn kích thước vocab (None = không giới hạn).
        pad_length     : Padding cố định (None = không padding).

    Returns:
        (corpus_path, label_path) — đường dẫn tuyệt đối tới 2 file đầu ra.

    Raises:
        FileNotFoundError : Nếu data_path không tồn tại.
        ValueError        : Nếu JSON không đúng định dạng.
    """
    SEP = "─" * 55

    # ── 1. Load data ───────────────────────────────────────────
    print(f"\n{SEP}")
    print("  [1/4] LOAD DỮ LIỆU")
    print(SEP)
    texts, labels = _load_json(data_path)

    # ── 2. Cấu hình pipeline ───────────────────────────────────
    print(f"\n{SEP}")
    print("  [2/4] CẤU HÌNH PIPELINE")
    print(SEP)

    sw_file = str(stopwords_path) if stopwords_path and Path(stopwords_path).exists() else None
    if stopwords_path and not sw_file:
        print(f"  [!] Không tìm thấy stopwords '{stopwords_path}', bỏ qua.")

    cfg = PipelineConfig(
        apply_lowercase            = True,
        apply_remove_special       = True,
        apply_remove_numbers       = False,
        apply_normalize_whitespace = True,
        stopwords_file             = sw_file,
        tokenizer_method           = tokenizer,
        min_freq                   = min_freq,
        max_vocab_size             = max_vocab_size,
        pad_token                  = "<PAD>",
        unk_token                  = "<UNK>",
        pad_length                 = pad_length,
        truncate                   = True,
    )

    print(f"  tokenizer    : {tokenizer}")
    print(f"  stopwords    : {sw_file or 'không dùng'}")
    print(f"  min_freq     : {min_freq}")
    print(f"  max_vocab    : {max_vocab_size or 'không giới hạn'}")
    print(f"  pad_length   : {pad_length or 'không padding'}")

    # ── 3. Fit + transform ─────────────────────────────────────
    print(f"\n{SEP}")
    print("  [3/4] FIT & TRANSFORM")
    print(SEP)

    pipeline = NLPPipeline(cfg)
    samples  = pipeline.fit_transform(texts)

    # ── 4. Lưu corpus.txt và labels.txt ───────────────────────
    print(f"\n{SEP}")
    print("  [4/4] LƯU KẾT QUẢ")
    print(SEP)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    corpus_path = out / "corpus.txt"
    label_path  = out / "labels.txt"

    with corpus_path.open("w", encoding="utf-8") as fc, \
         label_path.open("w", encoding="utf-8") as fl:
        for sample, label in zip(samples, labels):
            fc.write(" ".join(sample.tokens) + "\n")
            fl.write(label + "\n")

    print(f"  corpus.txt : {corpus_path.resolve()}  ({len(samples):,} dòng)")
    print(f"  labels.txt : {label_path.resolve()}   ({len(labels):,} dòng)")
    print(f"\n{SEP}\n")

    return corpus_path.resolve(), label_path.resolve()


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Xử lý data JSON → corpus.txt + labels.txt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",       required=True,              help="Đường dẫn file data.json")
    parser.add_argument("--stopwords",  default=None,               help="Đường dẫn file stopwords.txt")
    parser.add_argument("--output",     default="output",           help="Thư mục lưu kết quả")
    parser.add_argument("--tokenizer",  default="whitespace",
                        choices=["whitespace", "ngram", "subword", "syllable"],
                        help="Phương thức tokenize")
    parser.add_argument("--min-freq",   type=int,   default=2,      help="Tần suất tối thiểu vào vocab")
    parser.add_argument("--max-vocab",  type=int,   default=10_000, help="Giới hạn vocab size (0 = không giới hạn)")
    parser.add_argument("--pad-length", type=int,   default=None,   help="Độ dài padding cố định")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    corpus_file, label_file = process(
        data_path      = args.data,
        stopwords_path = args.stopwords,
        output_dir     = args.output,
        tokenizer      = args.tokenizer,
        min_freq       = args.min_freq,
        max_vocab_size = args.max_vocab if args.max_vocab > 0 else None,
        pad_length     = args.pad_length,
    )

    print(f"✓ corpus : {corpus_file}")
    print(f"✓ labels : {label_file}")