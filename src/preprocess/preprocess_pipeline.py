"""
nlp_pipeline.py
---------------
Pipeline NLP tổng hợp cho bài toán học máy.

Tích hợp tuần tự 3 module:
  ┌─────────────────────────────────────────────────────────────┐
  │  [1] text_preprocessing.py                                  │
  │      lowercase → bỏ ký tự đặc biệt → bỏ số → norm space   │
  │                          ↓                                  │
  │  [2] stopwords_removal.py                                   │
  │      loại bỏ stopwords từ file .txt                         │
  │                          ↓                                  │
  │  [3] tokenizer.py                                           │
  │      whitespace / syllable / subword / ngram                │
  │                          ↓                                  │
  │  OUTPUT — chuẩn đầu vào cho học máy:                       │
  │    • token list          List[str]                          │
  │    • token ids           List[int]         (dùng vocab)     │
  │    • one-hot / BoW       Dict[str, int]    (bag of words)   │
  │    • TF-IDF vectors      Dict[str, float]  (per document)   │
  │    • padded sequences    List[int]         (fixed length)   │
  └─────────────────────────────────────────────────────────────┘

Không yêu cầu thư viện ngoài — chỉ dùng thư viện chuẩn Python.
"""

import json
import math
import re
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


# ══════════════════════════════════════════════════════════════
# Import các module nội bộ
# ══════════════════════════════════════════════════════════════

try:
    from text_process import preprocess, normalize_whitespace
except ImportError as e:
    raise ImportError(
        "[nlp_pipeline] Không tìm thấy 'text_preprocessing.py'. "
        "Đặt file cùng thư mục với nlp_pipeline.py."
    ) from e

try:
    from stopword_remove import StopwordsRemover
except ImportError as e:
    raise ImportError(
        "[nlp_pipeline] Không tìm thấy 'stopwords_removal.py'. "
        "Đặt file cùng thư mục với nlp_pipeline.py."
    ) from e

try:
    from tokenizer import WhitespaceTokenizer, SyllableTokenizer, SubwordTokenizer, NgramTokenizer
except ImportError as e:
    raise ImportError(
        "[nlp_pipeline] Không tìm thấy 'tokenizer.py'. "
        "Đặt file cùng thư mục với nlp_pipeline.py."
    ) from e


# ══════════════════════════════════════════════════════════════
# Cấu hình pipeline (dataclass)
# ══════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """
    Toàn bộ cấu hình cho NLPPipeline.

    Nhóm 1 — Text Preprocessing:
        apply_lowercase           : Chuyển chữ thường (mặc định True).
        apply_remove_special      : Bỏ ký tự đặc biệt (mặc định True).
        apply_remove_numbers      : Bỏ chữ số (mặc định True).
        apply_normalize_whitespace: Chuẩn hoá khoảng trắng (mặc định True).

    Nhóm 2 — Stopwords:
        stopwords_file            : Đường dẫn file stopwords.txt.
        extra_stopwords           : Thêm stopwords tùy ý.

    Nhóm 3 — Tokenizer:
        tokenizer_method          : 'whitespace' | 'syllable' | 'subword' | 'ngram'.
        subword_max_len           : Độ dài tối đa mỗi subword (dùng khi method='subword').
        ngram_n                   : Kích thước n-gram (dùng khi method='ngram').
        ngram_level               : 'word' | 'char' (dùng khi method='ngram').

    Nhóm 4 — Vocabulary & Encoding:
        min_freq                  : Tần suất tối thiểu để từ được đưa vào vocab (mặc định 1).
        max_vocab_size            : Giới hạn kích thước vocab (None = không giới hạn).
        pad_token                 : Token dùng để padding (mặc định '<PAD>').
        unk_token                 : Token thay thế từ ngoài vocab (mặc định '<UNK>').
        pad_length                : Độ dài cố định khi padding (None = không padding).
        truncate                  : Cắt bớt nếu dài hơn pad_length (mặc định True).
    """
    # Preprocessing
    apply_lowercase: bool = True
    apply_remove_special: bool = True
    apply_remove_numbers: bool = True
    apply_normalize_whitespace: bool = True

    # Stopwords
    stopwords_file: Optional[str] = None
    extra_stopwords: List[str] = field(default_factory=list)

    # Tokenizer
    tokenizer_method: str = "whitespace"   # whitespace | syllable | subword | ngram
    subword_max_len: int = 4
    ngram_n: int = 2
    ngram_level: str = "word"              # word | char

    # Vocabulary & Encoding
    min_freq: int = 1
    max_vocab_size: Optional[int] = None
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    pad_length: Optional[int] = None
    truncate: bool = True


# ══════════════════════════════════════════════════════════════
# Kết quả xử lý một văn bản (dataclass)
# ══════════════════════════════════════════════════════════════

@dataclass
class ProcessedSample:
    """
    Kết quả đầy đủ sau khi xử lý một chuỗi văn bản qua pipeline.

    Attributes:
        original      : Văn bản thô ban đầu.
        cleaned       : Sau bước text_preprocessing.
        no_stopwords  : Sau bước loại bỏ stopwords.
        tokens        : Danh sách token cuối cùng.
        token_ids     : Danh sách ID của từng token theo vocab.
        padded_ids    : token_ids sau khi padding / truncate (nếu bật).
        bow           : Bag-of-Words {token: count}.
        tfidf         : TF-IDF vector {token: score} (tính sau khi fit corpus).
    """
    original: str
    cleaned: str
    no_stopwords: str
    tokens: List[str]
    token_ids: List[int] = field(default_factory=list)
    padded_ids: List[int] = field(default_factory=list)
    bow: Dict[str, int] = field(default_factory=dict)
    tfidf: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════════
# NLPPipeline — Lớp pipeline chính
# ══════════════════════════════════════════════════════════════

class NLPPipeline:
    """
    Pipeline NLP tổng hợp: từ văn bản thô → chuẩn đầu vào học máy.

    Luồng xử lý:
        text_preprocessing  →  stopwords_removal  →  tokenize
        →  build_vocab  →  encode (token_ids)  →  padding
        →  BoW / TF-IDF

    Ví dụ sử dụng:
        cfg = PipelineConfig(stopwords_file="stopwords.txt", pad_length=10)
        pipeline = NLPPipeline(cfg)
        pipeline.fit(corpus)           # xây vocab + IDF từ corpus
        result = pipeline.transform("Tôi đang học Python!!!")
        batch  = pipeline.fit_transform(corpus)
    """

    TOKENIZER_METHODS = ("whitespace", "syllable", "subword", "ngram")

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._vocab: Dict[str, int] = {}          # token → id
        self._id2token: Dict[int, str] = {}       # id → token
        self._idf: Dict[str, float] = {}          # token → IDF score
        self._is_fitted: bool = False

        self._init_components()

    # ──────────────────────────────────────
    # Khởi tạo các thành phần con
    # ──────────────────────────────────────

    def _init_components(self) -> None:
        cfg = self.config

        # Stopwords
        self._remover: Optional[StopwordsRemover] = None
        if cfg.stopwords_file:
            self._remover = StopwordsRemover(
                cfg.stopwords_file,
                extra_stopwords=cfg.extra_stopwords or [],
            )

        # Tokenizer
        method = cfg.tokenizer_method
        if method not in self.TOKENIZER_METHODS:
            raise ValueError(
                f"tokenizer_method phải là một trong {self.TOKENIZER_METHODS}, "
                f"nhận được '{method}'"
            )
        if method == "whitespace":
            self._tokenizer = WhitespaceTokenizer()
        elif method == "syllable":
            self._tokenizer = SyllableTokenizer()
        elif method == "subword":
            self._tokenizer = SubwordTokenizer(max_len=cfg.subword_max_len)
        elif method == "ngram":
            self._tokenizer = NgramTokenizer(n=cfg.ngram_n, level=cfg.ngram_level)

    # ══════════════════════════════════════
    # BƯỚC 1 — Text Preprocessing
    # ══════════════════════════════════════

    def _step1_preprocess(self, text: str) -> str:
        """Lowercase, bỏ ký tự đặc biệt, bỏ số, chuẩn hoá khoảng trắng."""
        return preprocess(
            text,
            apply_lowercase=self.config.apply_lowercase,
            apply_remove_special=self.config.apply_remove_special,
            apply_remove_numbers=self.config.apply_remove_numbers,
            apply_normalize_whitespace=self.config.apply_normalize_whitespace,
        )

    # ══════════════════════════════════════
    # BƯỚC 2 — Loại bỏ Stopwords
    # ══════════════════════════════════════

    def _step2_remove_stopwords(self, text: str) -> str:
        """Loại bỏ stopwords nếu đã cấu hình file."""
        if self._remover:
            return self._remover.remove(text)
        return text

    # ══════════════════════════════════════
    # BƯỚC 3 — Tokenize
    # ══════════════════════════════════════

    def _step3_tokenize(self, text: str) -> List[str]:
        """Tokenize văn bản đã chuẩn hoá."""
        raw = self._tokenizer.tokenize(text)
        # NgramTokenizer trả về list of tuples → flatten thành chuỗi
        if self.config.tokenizer_method == "ngram":
            return ["_".join(gram) for gram in raw]
        return raw

    # ══════════════════════════════════════
    # BƯỚC 4 — Xây dựng Vocabulary
    # ══════════════════════════════════════

    def _build_vocab(self, all_tokens: List[List[str]]) -> None:
        """
        Xây dựng vocabulary từ danh sách token của toàn corpus.
        Áp dụng min_freq và max_vocab_size.
        """
        cfg = self.config
        freq: Counter = Counter()
        for tokens in all_tokens:
            freq.update(tokens)

        # Lọc theo min_freq
        filtered = [(tok, cnt) for tok, cnt in freq.most_common() if cnt >= cfg.min_freq]

        # Giới hạn vocab size
        if cfg.max_vocab_size:
            filtered = filtered[: cfg.max_vocab_size]

        # Đặt token đặc biệt ở đầu
        special = [cfg.pad_token, cfg.unk_token]
        self._vocab = {tok: idx for idx, tok in enumerate(special)}
        for tok, _ in filtered:
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab)

        self._id2token = {idx: tok for tok, idx in self._vocab.items()}

    # ══════════════════════════════════════
    # BƯỚC 5 — Tính IDF (TF-IDF)
    # ══════════════════════════════════════

    def _build_idf(self, all_tokens: List[List[str]]) -> None:
        """Tính IDF cho từng token trong vocab."""
        n_docs = len(all_tokens)
        df: Counter = Counter()
        for tokens in all_tokens:
            df.update(set(tokens))  # mỗi doc chỉ đếm 1 lần

        for tok in self._vocab:
            doc_freq = df.get(tok, 0)
            # Smoothed IDF: log((1 + N) / (1 + df)) + 1
            self._idf[tok] = math.log((1 + n_docs) / (1 + doc_freq)) + 1

    # ══════════════════════════════════════
    # BƯỚC 6 — Encode token → ID
    # ══════════════════════════════════════

    def _encode(self, tokens: List[str]) -> List[int]:
        """Chuyển danh sách token thành danh sách ID."""
        unk_id = self._vocab.get(self.config.unk_token, 1)
        return [self._vocab.get(tok, unk_id) for tok in tokens]

    # ══════════════════════════════════════
    # BƯỚC 7 — Padding / Truncate
    # ══════════════════════════════════════

    def _pad(self, ids: List[int]) -> List[int]:
        """Padding hoặc truncate danh sách ID theo pad_length."""
        cfg = self.config
        if cfg.pad_length is None:
            return ids
        pad_id = self._vocab.get(cfg.pad_token, 0)
        if cfg.truncate and len(ids) > cfg.pad_length:
            return ids[: cfg.pad_length]
        return ids + [pad_id] * max(0, cfg.pad_length - len(ids))

    # ══════════════════════════════════════
    # BƯỚC 8 — Bag of Words
    # ══════════════════════════════════════

    @staticmethod
    def _bow(tokens: List[str]) -> Dict[str, int]:
        """Trả về Bag-of-Words {token: count}."""
        return dict(Counter(tokens))

    # ══════════════════════════════════════
    # BƯỚC 9 — TF-IDF
    # ══════════════════════════════════════

    def _tfidf(self, tokens: List[str]) -> Dict[str, float]:
        """Tính TF-IDF vector cho một document."""
        if not self._idf:
            return {}
        n = len(tokens)
        if n == 0:
            return {}
        tf = Counter(tokens)
        return {
            tok: round((cnt / n) * self._idf.get(tok, 1.0), 6)
            for tok, cnt in tf.items()
        }

    # ══════════════════════════════════════
    # Public API
    # ══════════════════════════════════════

    def fit(self, corpus: List[str]) -> "NLPPipeline":
        """
        Xây dựng vocabulary và IDF từ corpus.
        Phải gọi trước transform() nếu cần token_ids / tfidf.

        Args:
            corpus: Danh sách văn bản thô.

        Returns:
            self (hỗ trợ method chaining).
        """
        print(f"[NLPPipeline] Đang fit corpus ({len(corpus)} văn bản)...")
        all_tokens = []
        for text in corpus:
            cleaned   = self._step1_preprocess(text)
            no_sw     = self._step2_remove_stopwords(cleaned)
            tokens    = self._step3_tokenize(no_sw)
            all_tokens.append(tokens)

        self._build_vocab(all_tokens)
        self._build_idf(all_tokens)
        self._is_fitted = True
        print(f"[NLPPipeline] Vocab size: {len(self._vocab)} token.")
        return self

    def transform(self, text: str) -> ProcessedSample:
        """
        Xử lý một chuỗi văn bản qua toàn bộ pipeline.

        Args:
            text: Chuỗi văn bản thô.

        Returns:
            ProcessedSample với đầy đủ các trường đầu ra.
        """
        # Ba bước chính
        cleaned      = self._step1_preprocess(text)
        no_stopwords = self._step2_remove_stopwords(cleaned)
        tokens       = self._step3_tokenize(no_stopwords)

        # Encode & padding (chỉ khi đã fit)
        token_ids, padded_ids = [], []
        if self._is_fitted:
            token_ids  = self._encode(tokens)
            padded_ids = self._pad(token_ids)

        return ProcessedSample(
            original     = text,
            cleaned      = cleaned,
            no_stopwords = no_stopwords,
            tokens       = tokens,
            token_ids    = token_ids,
            padded_ids   = padded_ids,
            bow          = self._bow(tokens),
            tfidf        = self._tfidf(tokens) if self._is_fitted else {},
        )

    def fit_transform(self, corpus: List[str]) -> List[ProcessedSample]:
        """
        Fit từ corpus rồi transform toàn bộ corpus.
        Tương đương gọi fit(corpus) rồi [transform(t) for t in corpus].

        Args:
            corpus: Danh sách văn bản thô.

        Returns:
            Danh sách ProcessedSample.
        """
        self.fit(corpus)
        return [self.transform(text) for text in corpus]

    def transform_batch(self, texts: List[str]) -> List[ProcessedSample]:
        """
        Transform danh sách văn bản (không fit lại vocab).

        Args:
            texts: Danh sách chuỗi văn bản.

        Returns:
            Danh sách ProcessedSample.
        """
        return [self.transform(t) for t in texts]

    # ──────────────────────────────────────
    # Xuất kết quả
    # ──────────────────────────────────────

    def to_token_matrix(self, samples: List[ProcessedSample]) -> List[List[int]]:
        """
        Trả về ma trận token_id dạng List[List[int]].
        Nếu pad_length được đặt, mỗi hàng có độ dài bằng nhau.

        Args:
            samples: Kết quả từ fit_transform hoặc transform_batch.

        Returns:
            Ma trận 2D (n_samples × seq_len).
        """
        if self.config.pad_length:
            return [s.padded_ids for s in samples]
        return [s.token_ids for s in samples]

    def to_bow_matrix(self, samples: List[ProcessedSample]) -> Tuple[List[str], List[List[int]]]:
        """
        Trả về ma trận Bag-of-Words dạng (feature_names, matrix).

        Args:
            samples: Kết quả từ fit_transform hoặc transform_batch.

        Returns:
            (feature_names, matrix) — feature_names là danh sách token trong vocab.
        """
        feature_names = [
            tok for tok in self._vocab
            if tok not in (self.config.pad_token, self.config.unk_token)
        ]
        matrix = []
        for s in samples:
            row = [s.bow.get(tok, 0) for tok in feature_names]
            matrix.append(row)
        return feature_names, matrix

    def to_tfidf_matrix(self, samples: List[ProcessedSample]) -> Tuple[List[str], List[List[float]]]:
        """
        Trả về ma trận TF-IDF dạng (feature_names, matrix).

        Args:
            samples: Kết quả từ fit_transform hoặc transform_batch.

        Returns:
            (feature_names, matrix) — feature_names là danh sách token trong vocab.
        """
        feature_names = [
            tok for tok in self._vocab
            if tok not in (self.config.pad_token, self.config.unk_token)
        ]
        matrix = []
        for s in samples:
            row = [round(s.tfidf.get(tok, 0.0), 6) for tok in feature_names]
            matrix.append(row)
        return feature_names, matrix

    # ──────────────────────────────────────
    # Lưu / nạp vocab
    # ──────────────────────────────────────

    def save_vocab(self, path: Union[str, Path]) -> None:
        """Lưu vocabulary ra file JSON."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)
        print(f"[NLPPipeline] Đã lưu vocab ({len(self._vocab)} token) → '{path}'")

    def load_vocab(self, path: Union[str, Path]) -> None:
        """Nạp vocabulary từ file JSON đã lưu."""
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            self._vocab = json.load(f)
        self._id2token = {idx: tok for tok, idx in self._vocab.items()}
        self._is_fitted = True
        print(f"[NLPPipeline] Đã nạp vocab ({len(self._vocab)} token) từ '{path}'")

    def save_results_csv(
        self,
        samples: List[ProcessedSample],
        path: Union[str, Path],
    ) -> None:
        """
        Lưu kết quả xử lý ra file CSV (phù hợp nạp vào pandas / sklearn).

        Các cột: original, cleaned, no_stopwords, tokens, token_ids, padded_ids.

        Args:
            samples: Danh sách ProcessedSample.
            path   : Đường dẫn file CSV đầu ra.
        """
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "original", "cleaned", "no_stopwords",
                "tokens", "token_ids", "padded_ids",
            ])
            for s in samples:
                writer.writerow([
                    s.original,
                    s.cleaned,
                    s.no_stopwords,
                    " ".join(s.tokens),
                    " ".join(map(str, s.token_ids)),
                    " ".join(map(str, s.padded_ids)),
                ])
        print(f"[NLPPipeline] Đã lưu {len(samples)} mẫu → '{path}'")

    # ──────────────────────────────────────
    # Thống kê corpus
    # ──────────────────────────────────────

    def corpus_stats(self, samples: List[ProcessedSample]) -> Dict[str, Any]:
        """
        Trả về thống kê tổng quan của corpus sau khi xử lý.

        Args:
            samples: Danh sách ProcessedSample.

        Returns:
            Dict chứa các chỉ số thống kê.
        """
        all_tokens = [tok for s in samples for tok in s.tokens]
        lengths    = [len(s.tokens) for s in samples]

        return {
            "n_documents"       : len(samples),
            "vocab_size"        : len(self._vocab),
            "total_tokens"      : len(all_tokens),
            "unique_tokens"     : len(set(all_tokens)),
            "avg_tokens_per_doc": round(sum(lengths) / len(lengths), 2) if lengths else 0,
            "max_tokens"        : max(lengths) if lengths else 0,
            "min_tokens"        : min(lengths) if lengths else 0,
            "top10_tokens"      : Counter(all_tokens).most_common(10),
        }

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"NLPPipeline("
            f"method='{self.config.tokenizer_method}', "
            f"vocab={len(self._vocab)}, "
            f"status={status})"
        )


# ══════════════════════════════════════════════════════════════
# Hàm tiện ích nhanh (không cần khởi tạo class)
# ══════════════════════════════════════════════════════════════

def quick_process(
    texts: List[str],
    stopwords_file: Optional[str] = None,
    tokenizer_method: str = "whitespace",
    pad_length: Optional[int] = None,
) -> List[ProcessedSample]:
    """
    Xử lý nhanh một danh sách văn bản với cấu hình mặc định.

    Args:
        texts            : Danh sách văn bản thô.
        stopwords_file   : Đường dẫn file stopwords (tùy chọn).
        tokenizer_method : Phương thức tokenize (mặc định 'whitespace').
        pad_length       : Độ dài padding (None = không padding).

    Returns:
        Danh sách ProcessedSample đã fit + transform.
    """
    cfg = PipelineConfig(
        stopwords_file=stopwords_file,
        tokenizer_method=tokenizer_method,
        pad_length=pad_length,
    )
    pipeline = NLPPipeline(cfg)
    return pipeline.fit_transform(texts)


# ══════════════════════════════════════════════════════════════
# Demo khi chạy trực tiếp
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    corpus = [
        "  Tôi VÀ Bạn Đang Học Máy Học 123!!!  ",
        "Tiền xử lý văn bản là bước QUAN TRỌNG #1 trong NLP!!!",
        "Python 3.11 ra mắt vào năm 2022 — tốc độ tăng ~60%.",
        "Machine Learning và Deep Learning rất quan trọng trong AI.",
        "Xử lý ngôn ngữ tự nhiên giúp máy hiểu được tiếng người.",
        "Tokenize là bước đầu tiên để chuyển văn bản thành số.",
        "Mô hình học máy cần dữ liệu sạch để huấn luyện tốt.",
        "Bag of Words và TF-IDF là hai phương pháp biểu diễn văn bản phổ biến.",
    ]

    sw_path = Path(__file__).parent / "stopwords.txt"

    SEP = "═" * 68

    # ── Cấu hình pipeline ──────────────────────────────────────
    cfg = PipelineConfig(
        stopwords_file  = str(sw_path) if sw_path.exists() else None,
        tokenizer_method= "whitespace",
        pad_length      = 12,
        min_freq        = 1,
        max_vocab_size  = 200,
    )

    pipeline = NLPPipeline(cfg)
    samples  = pipeline.fit_transform(corpus)

    print(f"\n{SEP}")
    print("  NLP PIPELINE — KẾT QUẢ XỬ LÝ TỪNG VĂN BẢN")
    print(SEP)

    for i, (text, s) in enumerate(zip(corpus, samples), 1):
        print(f"\n  [{i}] Gốc          : {s.original.strip()}")
        print(f"       Bước 1 (clean): {s.cleaned}")
        print(f"       Bước 2 (no SW): {s.no_stopwords}")
        print(f"       Bước 3 (token): {s.tokens}")
        print(f"       Token IDs      : {s.token_ids}")
        print(f"       Padded IDs     : {s.padded_ids}")
        print(f"       BoW            : {s.bow}")
        print(f"       TF-IDF         : { {k: round(v,4) for k,v in s.tfidf.items()} }")

    # ── Thống kê corpus ────────────────────────────────────────
    print(f"\n{SEP}")
    print("  THỐNG KÊ CORPUS")
    print(SEP)
    stats = pipeline.corpus_stats(samples)
    for key, val in stats.items():
        print(f"  {key:<25}: {val}")

    # ── Ma trận đầu ra cho học máy ─────────────────────────────
    print(f"\n{SEP}")
    print("  MA TRẬN TOKEN IDs (padded) — ĐẦU VÀO CHO HỌC MÁY")
    print(SEP)
    matrix = pipeline.to_token_matrix(samples)
    for i, row in enumerate(matrix, 1):
        print(f"  doc_{i:02d}: {row}")

    print(f"\n{SEP}")
    print("  MA TRẬN TF-IDF (5 feature đầu) — ĐẦU VÀO CHO HỌC MÁY")
    print(SEP)
    feat_names, tfidf_mat = pipeline.to_tfidf_matrix(samples)
    header = "  {:<6}  ".format("doc") + "  ".join(f"{f:<12}" for f in feat_names[:5])
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, row in enumerate(tfidf_mat, 1):
        line = f"  doc_{i:02d}  " + "  ".join(f"{v:<12.4f}" for v in row[:5])
        print(line)

    # ── Lưu kết quả ────────────────────────────────────────────
    out_dir = Path(__file__).parent
    pipeline.save_vocab(out_dir / "vocab.json")
    pipeline.save_results_csv(samples, out_dir / "processed_corpus.csv")

    print(f"\n{SEP}")
    print("  QUICK PROCESS (hàm tiện ích)")
    print(SEP)
    quick = quick_process(
        ["  Học máy rất thú vị!!!  ", "NLP giúp máy hiểu ngôn ngữ người."],
        stopwords_file=str(sw_path) if sw_path.exists() else None,
    )
    for s in quick:
        print(f"  Gốc  : {s.original.strip()}")
        print(f"  Token: {s.tokens}\n")

    print(f"{pipeline}\n")