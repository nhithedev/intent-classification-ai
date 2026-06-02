"""
tokenizer.py
------------
Module tokenize văn bản đã được chuẩn hóa (lowercase, loại bỏ ký tự đặc biệt,
số, và stopwords) từ các module:
  - text_preprocessing.py
  - stopwords_removal.py

Các phương thức tokenize:
  1. WhitespaceTokenizer  : Tách theo khoảng trắng (nhanh, đơn giản)
  2. CharTokenizer        : Tách theo từng ký tự
  3. NgramTokenizer       : Sinh n-gram (unigram, bigram, trigram, ...)
  4. SyllableTokenizer    : Tách âm tiết tiếng Việt (không cần thư viện ngoài)
  5. SubwordTokenizer     : Tách theo BPE-style đơn giản (dựa trên độ dài từ)
  6. Tokenizer (pipeline) : Kết hợp toàn bộ pipeline từ text_preprocessing
                            → stopwords_removal → tokenize
"""

import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple, Union


# ══════════════════════════════════════════════════════════════
# 1. WhitespaceTokenizer — Tách theo khoảng trắng
# ══════════════════════════════════════════════════════════════

class WhitespaceTokenizer:
    """
    Tokenizer đơn giản nhất: tách văn bản thành danh sách token
    dựa trên khoảng trắng.

    Phù hợp với văn bản đã được chuẩn hóa (đã qua text_preprocessing).

    Example:
        >>> tok = WhitespaceTokenizer()
        >>> tok.tokenize("học máy xử lý ngôn ngữ tự nhiên")
        ['học', 'máy', 'xử', 'lý', 'ngôn', 'ngữ', 'tự', 'nhiên']
    """

    def tokenize(self, text: str) -> List[str]:
        """
        Tách văn bản thành danh sách token theo khoảng trắng.

        Args:
            text: Chuỗi văn bản đã chuẩn hóa.

        Returns:
            Danh sách token.
        """
        _validate_str(text)
        return text.split()

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """Tokenize hàng loạt danh sách chuỗi."""
        _validate_list(texts)
        return [self.tokenize(t) for t in texts]

    def detokenize(self, tokens: List[str]) -> str:
        """Ghép danh sách token thành chuỗi."""
        return " ".join(tokens)


# ══════════════════════════════════════════════════════════════
# 2. CharTokenizer — Tách từng ký tự
# ══════════════════════════════════════════════════════════════

class CharTokenizer:
    """
    Tách văn bản thành từng ký tự đơn lẻ.
    Hữu ích cho các mô hình character-level.

    Example:
        >>> tok = CharTokenizer()
        >>> tok.tokenize("học máy")
        ['h', 'ọ', 'c', ' ', 'm', 'á', 'y']
        >>> tok.tokenize("học máy", include_space=False)
        ['h', 'ọ', 'c', 'm', 'á', 'y']
    """

    def tokenize(self, text: str, include_space: bool = True) -> List[str]:
        """
        Tách văn bản thành danh sách ký tự.

        Args:
            text         : Chuỗi văn bản đầu vào.
            include_space: Giữ lại ký tự khoảng trắng (mặc định True).

        Returns:
            Danh sách các ký tự.
        """
        _validate_str(text)
        if include_space:
            return list(text)
        return [ch for ch in text if ch != " "]

    def tokenize_batch(self, texts: List[str], include_space: bool = True) -> List[List[str]]:
        """Tokenize hàng loạt."""
        _validate_list(texts)
        return [self.tokenize(t, include_space) for t in texts]

    def detokenize(self, tokens: List[str]) -> str:
        """Ghép danh sách ký tự thành chuỗi."""
        return "".join(tokens)


# ══════════════════════════════════════════════════════════════
# 3. NgramTokenizer — Tạo n-gram
# ══════════════════════════════════════════════════════════════

class NgramTokenizer:
    """
    Sinh n-gram (cụm n token liên tiếp) từ văn bản.

    Hỗ trợ:
      - word n-gram  : n-gram ở cấp độ từ
      - char n-gram  : n-gram ở cấp độ ký tự

    Example:
        >>> tok = NgramTokenizer(n=2)
        >>> tok.tokenize("học máy xử lý")
        [('học', 'máy'), ('máy', 'xử'), ('xử', 'lý')]

        >>> tok = NgramTokenizer(n=3)
        >>> tok.tokenize("học máy xử lý")
        [('học', 'máy', 'xử'), ('máy', 'xử', 'lý')]
    """

    def __init__(self, n: int = 2, level: str = "word"):
        """
        Args:
            n    : Kích thước n-gram (mặc định bigram = 2).
            level: 'word' (n-gram từ) hoặc 'char' (n-gram ký tự).
        """
        if n < 1:
            raise ValueError(f"n phải >= 1, nhận được {n}")
        if level not in ("word", "char"):
            raise ValueError(f"level phải là 'word' hoặc 'char', nhận được '{level}'")
        self.n = n
        self.level = level

    def _build_ngrams(self, sequence: List[str]) -> List[Tuple[str, ...]]:
        """Sinh n-gram từ một sequence."""
        if len(sequence) < self.n:
            return []
        return [tuple(sequence[i : i + self.n]) for i in range(len(sequence) - self.n + 1)]

    def tokenize(self, text: str) -> List[Tuple[str, ...]]:
        """
        Sinh n-gram từ văn bản.

        Args:
            text: Chuỗi văn bản đầu vào.

        Returns:
            Danh sách n-gram (mỗi n-gram là một tuple).
        """
        _validate_str(text)
        if self.level == "word":
            units = text.split()
        else:
            units = list(text.replace(" ", ""))
        return self._build_ngrams(units)

    def tokenize_batch(self, texts: List[str]) -> List[List[Tuple[str, ...]]]:
        """Tokenize hàng loạt."""
        _validate_list(texts)
        return [self.tokenize(t) for t in texts]

    def frequency(self, text: str) -> Dict[Tuple[str, ...], int]:
        """
        Đếm tần suất xuất hiện của từng n-gram.

        Args:
            text: Chuỗi văn bản đầu vào.

        Returns:
            Dict ánh xạ n-gram → số lần xuất hiện, sắp xếp giảm dần.
        """
        ngrams = self.tokenize(text)
        return dict(Counter(ngrams).most_common())


# ══════════════════════════════════════════════════════════════
# 4. SyllableTokenizer — Tách âm tiết tiếng Việt
# ══════════════════════════════════════════════════════════════

class SyllableTokenizer:
    """
    Tách văn bản tiếng Việt thành từng âm tiết (syllable).

    Tiếng Việt là ngôn ngữ đơn âm: mỗi từ viết thường là một âm tiết
    phân tách bởi khoảng trắng. Tokenizer này tách chuỗi thành các âm
    tiết và tùy chọn nhóm thành từ ghép theo độ dài cho trước.

    Example:
        >>> tok = SyllableTokenizer()
        >>> tok.tokenize("xử lý ngôn ngữ tự nhiên")
        ['xử', 'lý', 'ngôn', 'ngữ', 'tự', 'nhiên']

        >>> tok.tokenize("xử lý ngôn ngữ tự nhiên", group_size=2)
        ['xử lý', 'ngôn ngữ', 'tự nhiên']
    """

    # Mẫu nhận diện âm tiết tiếng Việt (chữ cái Unicode + dấu thanh)
    _SYLLABLE_PATTERN = re.compile(r"[\w]+", re.UNICODE)

    def tokenize(self, text: str, group_size: int = 1) -> List[str]:
        """
        Tách văn bản thành danh sách âm tiết.

        Args:
            text      : Chuỗi văn bản đầu vào (đã lowercase).
            group_size: Nhóm bao nhiêu âm tiết thành một token (mặc định 1).

        Returns:
            Danh sách âm tiết hoặc cụm âm tiết.
        """
        _validate_str(text)
        syllables = self._SYLLABLE_PATTERN.findall(text)
        if group_size == 1:
            return syllables
        # Nhóm theo group_size
        grouped = []
        for i in range(0, len(syllables), group_size):
            grouped.append(" ".join(syllables[i : i + group_size]))
        return grouped

    def tokenize_batch(self, texts: List[str], group_size: int = 1) -> List[List[str]]:
        """Tokenize hàng loạt."""
        _validate_list(texts)
        return [self.tokenize(t, group_size) for t in texts]

    def detokenize(self, tokens: List[str]) -> str:
        """Ghép danh sách âm tiết thành chuỗi."""
        return " ".join(tokens)


# ══════════════════════════════════════════════════════════════
# 5. SubwordTokenizer — Tách subword đơn giản
# ══════════════════════════════════════════════════════════════

class SubwordTokenizer:
    """
    Tokenizer tách từ thành các đơn vị con (subword) dựa trên
    độ dài ký tự tối đa cho phép.

    Mô phỏng hành vi cơ bản của BPE / WordPiece: những từ dài
    hơn `max_len` bị cắt thành các mảnh có tiền tố '##'.

    Hữu ích khi không có thư viện ngoài (sentencepiece, tokenizers).

    Example:
        >>> tok = SubwordTokenizer(max_len=4)
        >>> tok.tokenize("preprocessing tokenization")
        ['prep', '##ro', '##ce', '##ss', '##in', '##g',
         'toke', '##ni', '##za', '##ti', '##on']
    """

    def __init__(self, max_len: int = 4):
        """
        Args:
            max_len: Độ dài tối đa mỗi subword (mặc định 4 ký tự).
        """
        if max_len < 1:
            raise ValueError(f"max_len phải >= 1, nhận được {max_len}")
        self.max_len = max_len

    def _split_word(self, word: str) -> List[str]:
        """Tách một từ thành các subword."""
        if len(word) <= self.max_len:
            return [word]
        subwords = []
        start = 0
        while start < len(word):
            end = min(start + self.max_len, len(word))
            chunk = word[start:end]
            subwords.append(chunk if start == 0 else f"##{chunk}")
            start = end
        return subwords

    def tokenize(self, text: str) -> List[str]:
        """
        Tách văn bản thành danh sách subword.

        Args:
            text: Chuỗi văn bản đầu vào.

        Returns:
            Danh sách subword token.
        """
        _validate_str(text)
        result = []
        for word in text.split():
            result.extend(self._split_word(word))
        return result

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """Tokenize hàng loạt."""
        _validate_list(texts)
        return [self.tokenize(t) for t in texts]

    def detokenize(self, tokens: List[str]) -> str:
        """Ghép subword token thành chuỗi gốc."""
        text = ""
        for tok in tokens:
            if tok.startswith("##"):
                text += tok[2:]
            else:
                text += (" " if text else "") + tok
        return text


# ══════════════════════════════════════════════════════════════
# 6. Tokenizer — Pipeline đầy đủ
# ══════════════════════════════════════════════════════════════

class Tokenizer:
    """
    Pipeline tokenize tổng hợp, tích hợp với:
      - text_preprocessing.py  (preprocess)
      - stopwords_removal.py   (StopwordsRemover)

    Thứ tự xử lý:
      1. preprocess()          — lowercase, bỏ ký tự đặc biệt & số
      2. StopwordsRemover      — loại bỏ stopwords (tùy chọn)
      3. Tokenize              — theo phương pháp được chọn

    Phương thức tokenize hỗ trợ: 'whitespace', 'char', 'syllable', 'subword'
    N-gram có thể bật thêm sau khi tokenize.

    Example:
        >>> tok = Tokenizer(stopwords_file="stopwords.txt", method="whitespace")
        >>> tok.tokenize("  Tôi VÀ Bạn Đang Học Python 123!!!  ")
        ['học', 'python']
    """

    METHODS = ("whitespace", "char", "syllable", "subword")

    def __init__(
        self,
        stopwords_file: Optional[Union[str, Path]] = None,
        method: str = "whitespace",
        subword_max_len: int = 4,
        ngram_n: Optional[int] = None,
        ngram_level: str = "word",
    ):
        """
        Args:
            stopwords_file: Đường dẫn file stopwords (None = bỏ qua bước này).
            method        : Phương thức tokenize ('whitespace','char','syllable','subword').
            subword_max_len: Độ dài subword tối đa (chỉ dùng khi method='subword').
            ngram_n       : Nếu đặt giá trị, áp dụng thêm n-gram sau tokenize.
            ngram_level   : 'word' hoặc 'char' cho n-gram.
        """
        # Import ở đây để tránh circular import nếu dùng riêng lẻ
        try:
            from text_process import preprocess as _preprocess
            self._preprocess = _preprocess
        except ImportError:
            self._preprocess = None
            print("[Tokenizer] Không tìm thấy text_preprocessing.py — bỏ qua bước preprocess.")

        try:
            from stopword_remove import StopwordsRemover as _Remover
            if stopwords_file:
                self._remover = _Remover(stopwords_file)
            else:
                self._remover = None
        except ImportError:
            self._remover = None
            print("[Tokenizer] Không tìm thấy stopwords_removal.py — bỏ qua bước stopwords.")

        if method not in self.METHODS:
            raise ValueError(f"method phải là một trong {self.METHODS}, nhận được '{method}'")

        self.method = method
        self._tok_map = {
            "whitespace": WhitespaceTokenizer(),
            "char"      : CharTokenizer(),
            "syllable"  : SyllableTokenizer(),
            "subword"   : SubwordTokenizer(max_len=subword_max_len),
        }
        self._ngram = NgramTokenizer(n=ngram_n, level=ngram_level) if ngram_n else None

    def tokenize(self, text: str) -> List:
        """
        Chạy toàn bộ pipeline: preprocess → remove stopwords → tokenize.

        Args:
            text: Chuỗi văn bản thô đầu vào.

        Returns:
            Danh sách token (hoặc n-gram nếu ngram_n được đặt).
        """
        _validate_str(text)

        # Bước 1: Tiền xử lý
        if self._preprocess:
            text = self._preprocess(text)

        # Bước 2: Loại bỏ stopwords
        if self._remover:
            text = self._remover.remove(text)

        # Bước 3: Tokenize
        tokens = self._tok_map[self.method].tokenize(text)

        # Bước 4 (tùy chọn): N-gram
        if self._ngram:
            return self._ngram._build_ngrams(tokens)

        return tokens

    def tokenize_batch(self, texts: List[str]) -> List[List]:
        """Chạy pipeline cho hàng loạt chuỗi."""
        _validate_list(texts)
        return [self.tokenize(t) for t in texts]

    def vocabulary(self, texts: List[str]) -> Dict[str, int]:
        """
        Xây dựng bộ từ vựng (vocabulary) từ danh sách văn bản.

        Args:
            texts: Danh sách chuỗi văn bản.

        Returns:
            Dict ánh xạ token → index (sắp xếp theo tần suất giảm dần).
        """
        all_tokens = []
        for t in texts:
            all_tokens.extend(self.tokenize(t))
        freq = Counter(all_tokens).most_common()
        return {tok: idx for idx, (tok, _) in enumerate(freq)}


# ══════════════════════════════════════════════════════════════
# Hàm kiểm tra kiểu dữ liệu dùng nội bộ
# ══════════════════════════════════════════════════════════════

def _validate_str(text) -> None:
    if not isinstance(text, str):
        raise TypeError(f"Đầu vào phải là str, nhận được {type(text).__name__}")


def _validate_list(texts) -> None:
    if not isinstance(texts, list):
        raise TypeError(f"Đầu vào phải là list, nhận được {type(texts).__name__}")


# ══════════════════════════════════════════════════════════════
# Demo khi chạy trực tiếp
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from pathlib import Path

    samples = [
        "  Tôi VÀ Bạn Đang Học Máy Học 123!!!  ",
        "Tiền xử lý văn bản là bước QUAN TRỌNG #1 trong NLP!!!",
        "Python 3.11 ra mắt vào năm 2022 — tốc độ tăng ~60%.",
        "machine learning and deep learning are very important fields",
    ]

    sep = "=" * 65

    # ── 1. WhitespaceTokenizer ──────────────────────────────
    print(f"\n{sep}")
    print("1. WHITESPACE TOKENIZER")
    print(sep)
    wt = WhitespaceTokenizer()
    for s in samples:
        tokens = wt.tokenize(s)
        print(f"  Gốc   : {s.strip()}")
        print(f"  Tokens: {tokens}\n")

    # ── 2. CharTokenizer ────────────────────────────────────
    print(f"{sep}")
    print("2. CHAR TOKENIZER")
    print(sep)
    ct = CharTokenizer()
    demo = "học máy"
    print(f"  Gốc             : '{demo}'")
    print(f"  Có khoảng trắng : {ct.tokenize(demo, include_space=True)}")
    print(f"  Không khoảng trắng: {ct.tokenize(demo, include_space=False)}")

    # ── 3. NgramTokenizer ───────────────────────────────────
    print(f"\n{sep}")
    print("3. NGRAM TOKENIZER")
    print(sep)
    text = "học máy xử lý ngôn ngữ tự nhiên"
    for n in (2, 3):
        nt = NgramTokenizer(n=n)
        print(f"  {n}-gram: {nt.tokenize(text)}")
    print(f"\n  Tần suất bigram: {NgramTokenizer(n=2).frequency(text)}")

    # ── 4. SyllableTokenizer ────────────────────────────────
    print(f"\n{sep}")
    print("4. SYLLABLE TOKENIZER (TIẾNG VIỆT)")
    print(sep)
    st = SyllableTokenizer()
    demo_vi = "xử lý ngôn ngữ tự nhiên tiếng việt"
    print(f"  Gốc              : '{demo_vi}'")
    print(f"  Âm tiết đơn      : {st.tokenize(demo_vi)}")
    print(f"  Nhóm 2 âm tiết  : {st.tokenize(demo_vi, group_size=2)}")

    # ── 5. SubwordTokenizer ─────────────────────────────────
    print(f"\n{sep}")
    print("5. SUBWORD TOKENIZER")
    print(sep)
    swt = SubwordTokenizer(max_len=4)
    demo_en = "preprocessing tokenization"
    tokens = swt.tokenize(demo_en)
    print(f"  Gốc       : '{demo_en}'")
    print(f"  Subwords  : {tokens}")
    print(f"  Detokenize: '{swt.detokenize(tokens)}'")

    # ── 6. Pipeline Tokenizer ───────────────────────────────
    print(f"\n{sep}")
    print("6. PIPELINE TOKENIZER (preprocess → stopwords → tokenize)")
    print(sep)
    sw_path = Path(__file__).parent / "stopwords.txt"
    if sw_path.exists():
        pipeline = Tokenizer(stopwords_file=sw_path, method="whitespace")
        print()
        for s in samples:
            result = pipeline.tokenize(s)
            print(f"  Gốc   : {s.strip()}")
            print(f"  Tokens: {result}\n")

        # Vocabulary
        vocab = pipeline.vocabulary(samples)
        print(f"  Vocabulary ({len(vocab)} từ): {dict(list(vocab.items())[:10])} ...")
    else:
        print(f"  [!] Không tìm thấy stopwords.txt tại {sw_path}")
        print("      Đặt file cùng thư mục để dùng Pipeline Tokenizer.")