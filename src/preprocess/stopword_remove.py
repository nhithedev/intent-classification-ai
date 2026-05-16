"""
stopwords_removal.py
--------------------
Module loại bỏ stopwords khỏi văn bản.

Tính năng:
  - Đọc danh sách stopwords từ file .txt bên ngoài
  - Hỗ trợ thêm / xoá stopwords động
  - Loại bỏ stopwords ở cấp độ từ (token-level)
  - Xử lý đơn lẻ và hàng loạt (batch)
  - Tương thích với pipeline text_preprocessing.py
"""

import re
from pathlib import Path
from typing import List, Optional, Set, Union


# ══════════════════════════════════════════════════════════════
# 1. Lớp StopwordsRemover
# ══════════════════════════════════════════════════════════════

class StopwordsRemover:
    """
    Đọc stopwords từ file văn bản và loại bỏ chúng khỏi chuỗi đầu vào.

    File stopwords hỗ trợ:
      - Mỗi từ / cụm từ trên một dòng.
      - Dòng bắt đầu bằng '#' được bỏ qua (comment).
      - Dòng trống được bỏ qua.

    Ví dụ khởi tạo:
        remover = StopwordsRemover("stopwords.txt")
        remover = StopwordsRemover("stopwords.txt", extra_stopwords=["hello","world"])
    """

    def __init__(
        self,
        stopwords_file: Union[str, Path],
        extra_stopwords: Optional[List[str]] = None,
        case_sensitive: bool = False,
    ):
        """
        Args:
            stopwords_file  : Đường dẫn tới file stopwords (.txt).
            extra_stopwords : Danh sách từ bổ sung (tuỳ chọn).
            case_sensitive  : Phân biệt hoa/thường khi so sánh (mặc định False).
        """
        self.case_sensitive = case_sensitive
        self._stopwords: Set[str] = set()

        self._load_from_file(stopwords_file)

        if extra_stopwords:
            self.add_stopwords(extra_stopwords)

    # ──────────────────────────────────────
    # Đọc file
    # ──────────────────────────────────────

    def _load_from_file(self, filepath: Union[str, Path]) -> None:
        """Đọc và nạp stopwords từ file .txt."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file stopwords: '{path}'")
        if path.suffix.lower() != ".txt":
            raise ValueError(f"File phải có định dạng .txt, nhận được: '{path.suffix}'")

        with path.open(encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if not word or word.startswith("#"):
                    continue
                self._add_word(word)

        print(f"[StopwordsRemover] Đã nạp {len(self._stopwords)} stopwords từ '{path.name}'")

    def _add_word(self, word: str) -> None:
        """Thêm một từ vào tập stopwords (có tính case_sensitive)."""
        self._stopwords.add(word if self.case_sensitive else word.lower())

    # ──────────────────────────────────────
    # Quản lý stopwords động
    # ──────────────────────────────────────

    def add_stopwords(self, words: List[str]) -> None:
        """
        Thêm danh sách từ mới vào tập stopwords hiện tại.

        Args:
            words: Danh sách từ cần thêm.

        Example:
            >>> remover.add_stopwords(["alo", "hello"])
        """
        for w in words:
            self._add_word(w.strip())

    def remove_stopwords_from_set(self, words: List[str]) -> None:
        """
        Xoá danh sách từ khỏi tập stopwords.

        Args:
            words: Danh sách từ cần xoá.

        Example:
            >>> remover.remove_stopwords_from_set(["và", "của"])
        """
        for w in words:
            key = w.strip() if self.case_sensitive else w.strip().lower()
            self._stopwords.discard(key)

    def get_stopwords(self) -> Set[str]:
        """Trả về bản sao tập stopwords hiện tại."""
        return set(self._stopwords)

    # ──────────────────────────────────────
    # Kiểm tra một token
    # ──────────────────────────────────────

    def _is_stopword(self, token: str) -> bool:
        key = token if self.case_sensitive else token.lower()
        return key in self._stopwords

    # ──────────────────────────────────────
    # Tokenize đơn giản (tách theo khoảng trắng)
    # ──────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tách văn bản thành danh sách token theo khoảng trắng."""
        return text.split()

    # ──────────────────────────────────────
    # Xử lý một chuỗi
    # ──────────────────────────────────────

    def remove(self, text: str) -> str:
        """
        Loại bỏ stopwords khỏi một chuỗi văn bản.

        Args:
            text: Chuỗi văn bản đầu vào.

        Returns:
            Chuỗi văn bản sau khi loại bỏ stopwords.

        Example:
            >>> remover.remove("tôi và bạn đang học python")
            'học python'
        """
        if not isinstance(text, str):
            raise TypeError(f"Đầu vào phải là str, nhận được {type(text).__name__}")

        tokens = self._tokenize(text)
        filtered = [tok for tok in tokens if not self._is_stopword(tok)]
        return " ".join(filtered)

    # ──────────────────────────────────────
    # Xử lý hàng loạt
    # ──────────────────────────────────────

    def remove_batch(self, texts: List[str]) -> List[str]:
        """
        Loại bỏ stopwords cho một danh sách chuỗi văn bản.

        Args:
            texts: Danh sách chuỗi văn bản đầu vào.

        Returns:
            Danh sách chuỗi đã loại bỏ stopwords.

        Example:
            >>> remover.remove_batch(["tôi và bạn", "hello world and"])
            ['học', 'hello world']
        """
        if not isinstance(texts, list):
            raise TypeError(f"Đầu vào phải là list, nhận được {type(texts).__name__}")
        return [self.remove(t) for t in texts]

    # ──────────────────────────────────────
    # Thống kê
    # ──────────────────────────────────────

    def stats(self, text: str) -> dict:
        """
        Trả về thống kê loại bỏ stopwords cho một chuỗi.

        Args:
            text: Chuỗi văn bản đầu vào.

        Returns:
            dict gồm:
              - total_tokens   : Tổng số token ban đầu
              - removed_count  : Số token bị loại
              - kept_count     : Số token giữ lại
              - removed_words  : Danh sách các từ bị loại
              - result         : Văn bản sau xử lý

        Example:
            >>> remover.stats("tôi và bạn đang học python")
            {'total_tokens': 6, 'removed_count': 3, ...}
        """
        if not isinstance(text, str):
            raise TypeError(f"Đầu vào phải là str, nhận được {type(text).__name__}")

        tokens = self._tokenize(text)
        removed = [tok for tok in tokens if self._is_stopword(tok)]
        kept = [tok for tok in tokens if not self._is_stopword(tok)]

        return {
            "total_tokens": len(tokens),
            "removed_count": len(removed),
            "kept_count": len(kept),
            "removed_words": removed,
            "result": " ".join(kept),
        }

    def __repr__(self) -> str:
        return f"StopwordsRemover(total_stopwords={len(self._stopwords)}, case_sensitive={self.case_sensitive})"


# ══════════════════════════════════════════════════════════════
# 2. Hàm tiện ích độc lập (không cần khởi tạo class)
# ══════════════════════════════════════════════════════════════

def load_stopwords(filepath: Union[str, Path]) -> Set[str]:
    """
    Đọc và trả về tập stopwords từ file .txt.

    Args:
        filepath: Đường dẫn tới file stopwords.

    Returns:
        Tập hợp (set) các stopwords.
    """
    path = Path(filepath)
    stopwords: Set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word and not word.startswith("#"):
                stopwords.add(word.lower())
    return stopwords


def remove_stopwords(
    text: str,
    stopwords: Set[str],
    case_sensitive: bool = False,
) -> str:
    """
    Loại bỏ stopwords khỏi văn bản sử dụng một set stopwords cho trước.

    Args:
        text          : Chuỗi văn bản đầu vào.
        stopwords     : Tập stopwords (set).
        case_sensitive: Phân biệt hoa/thường (mặc định False).

    Returns:
        Chuỗi văn bản sau khi loại bỏ stopwords.

    Example:
        >>> sw = load_stopwords("stopwords.txt")
        >>> remove_stopwords("tôi và bạn học python", sw)
        'học python'
    """
    tokens = text.split()
    if case_sensitive:
        filtered = [t for t in tokens if t not in stopwords]
    else:
        filtered = [t for t in tokens if t.lower() not in stopwords]
    return " ".join(filtered)


# ══════════════════════════════════════════════════════════════
# 3. Demo khi chạy trực tiếp
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os

    # Tìm file stopwords.txt cùng thư mục với script này
    script_dir = Path(__file__).parent
    sw_file = script_dir / "stopwords.txt"

    if not sw_file.exists():
        print(f"[!] Không tìm thấy '{sw_file}'. Hãy đặt file stopwords.txt cùng thư mục.")
    else:
        remover = StopwordsRemover(sw_file)
        print(f"\n{remover}\n")

        samples = [
            "tôi và bạn đang cùng nhau học lập trình python",
            "tiền xử lý văn bản là bước rất quan trọng trong nlp",
            "the quick brown fox jumps over the lazy dog",
            "machine learning and deep learning are very important fields",
            "chúng tôi cần phải xử lý dữ liệu trước khi huấn luyện mô hình",
        ]

        print("=" * 65)
        print("KẾT QUẢ LOẠI BỎ STOPWORDS")
        print("=" * 65)

        for i, text in enumerate(samples, 1):
            info = remover.stats(text)
            print(f"\n[{i}] Gốc    : {text}")
            print(f"    Kết quả: {info['result']}")
            print(f"    Đã loại: {info['removed_count']}/{info['total_tokens']} token "
                  f"→ {info['removed_words']}")

        print("\n" + "=" * 65)
        print("BATCH PROCESSING")
        print("=" * 65)
        results = remover.remove_batch(samples)
        for original, processed in zip(samples, results):
            print(f"  ✓ {processed}")