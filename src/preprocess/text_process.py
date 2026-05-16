"""
text_preprocessing.py
---------------------
Module tiền xử lý dataset dạng text.
Bao gồm các hàm:
  - lowercase           : chuyển văn bản về chữ thường
  - remove_special_chars: loại bỏ ký tự đặc biệt
  - remove_numbers      : loại bỏ chữ số
  - preprocess          : pipeline tổng hợp toàn bộ bước trên
"""

import re
import unicodedata
from typing import List, Union


# ─────────────────────────────────────────────
# 1. Lowercase
# ─────────────────────────────────────────────

def lowercase(text: str) -> str:
    """
    Chuyển toàn bộ ký tự trong chuỗi về chữ thường.

    Args:
        text: Chuỗi văn bản đầu vào.

    Returns:
        Chuỗi văn bản đã lowercase.

    Example:
        >>> lowercase("Hello World! 123")
        'hello world! 123'
    """
    if not isinstance(text, str):
        raise TypeError(f"Đầu vào phải là str, nhận được {type(text).__name__}")
    return text.lower()


# ─────────────────────────────────────────────
# 2. Loại bỏ ký tự đặc biệt
# ─────────────────────────────────────────────

def remove_special_chars(text: str, keep_whitespace: bool = True) -> str:
    """
    Loại bỏ các ký tự đặc biệt (không phải chữ cái Unicode và không phải khoảng trắng).

    Hỗ trợ đầy đủ tiếng Việt và các ngôn ngữ Unicode khác.

    Args:
        text          : Chuỗi văn bản đầu vào.
        keep_whitespace: Nếu True, giữ lại khoảng trắng (mặc định True).

    Returns:
        Chuỗi văn bản sau khi loại bỏ ký tự đặc biệt.

    Example:
        >>> remove_special_chars("Xin chào! @#$%^&*()")
        'Xin chào '
        >>> remove_special_chars("Hello---World!!!", keep_whitespace=False)
        'HelloWorld'
    """
    if not isinstance(text, str):
        raise TypeError(f"Đầu vào phải là str, nhận được {type(text).__name__}")

    if keep_whitespace:
        # Giữ lại chữ cái Unicode (\w bao gồm cả tiếng Việt) và khoảng trắng
        pattern = r"[^\w\s]"
    else:
        pattern = r"[^\w]"

    cleaned = re.sub(pattern, "", text, flags=re.UNICODE)
    return cleaned


# ─────────────────────────────────────────────
# 3. Loại bỏ chữ số
# ─────────────────────────────────────────────

def remove_numbers(text: str) -> str:
    """
    Loại bỏ toàn bộ chữ số (0-9 và chữ số Unicode) khỏi văn bản.

    Args:
        text: Chuỗi văn bản đầu vào.

    Returns:
        Chuỗi văn bản sau khi loại bỏ chữ số.

    Example:
        >>> remove_numbers("Năm 2024 có 365 ngày")
        'Năm  có  ngày'
    """
    if not isinstance(text, str):
        raise TypeError(f"Đầu vào phải là str, nhận được {type(text).__name__}")

    return re.sub(r"\d+", "", text, flags=re.UNICODE)


# ─────────────────────────────────────────────
# 4. Chuẩn hóa khoảng trắng
# ─────────────────────────────────────────────

def normalize_whitespace(text: str) -> str:
    """
    Thu gọn nhiều khoảng trắng liên tiếp thành một khoảng trắng duy nhất
    và cắt bỏ khoảng trắng đầu/cuối chuỗi.

    Args:
        text: Chuỗi văn bản đầu vào.

    Returns:
        Chuỗi đã chuẩn hóa khoảng trắng.

    Example:
        >>> normalize_whitespace("  Xin   chào   thế   giới  ")
        'Xin chào thế giới'
    """
    if not isinstance(text, str):
        raise TypeError(f"Đầu vào phải là str, nhận được {type(text).__name__}")

    return re.sub(r"\s+", " ", text).strip()


# ─────────────────────────────────────────────
# 5. Pipeline tổng hợp
# ─────────────────────────────────────────────

def preprocess(
    text: str,
    apply_lowercase: bool = True,
    apply_remove_special: bool = True,
    apply_remove_numbers: bool = True,
    apply_normalize_whitespace: bool = True,
    keep_whitespace: bool = True,
) -> str:
    """
    Pipeline tiền xử lý văn bản tổng hợp.

    Thực hiện tuần tự các bước (có thể bật/tắt từng bước):
      1. Lowercase
      2. Loại bỏ ký tự đặc biệt
      3. Loại bỏ chữ số
      4. Chuẩn hóa khoảng trắng

    Args:
        text                      : Chuỗi văn bản đầu vào.
        apply_lowercase           : Bật/tắt bước lowercase (mặc định True).
        apply_remove_special      : Bật/tắt loại bỏ ký tự đặc biệt (mặc định True).
        apply_remove_numbers      : Bật/tắt loại bỏ chữ số (mặc định True).
        apply_normalize_whitespace: Bật/tắt chuẩn hóa khoảng trắng (mặc định True).
        keep_whitespace           : Truyền vào remove_special_chars (mặc định True).

    Returns:
        Chuỗi văn bản đã qua tiền xử lý.

    Example:
        >>> preprocess("  Hello World!!! 123  ")
        'hello world'
    """
    if apply_lowercase:
        text = lowercase(text)
    if apply_remove_special:
        text = remove_special_chars(text, keep_whitespace=keep_whitespace)
    if apply_remove_numbers:
        text = remove_numbers(text)
    if apply_normalize_whitespace:
        text = normalize_whitespace(text)
    return text


# ─────────────────────────────────────────────
# 6. Xử lý hàng loạt (batch)
# ─────────────────────────────────────────────

def preprocess_batch(
    texts: List[str],
    **kwargs,
) -> List[str]:
    """
    Áp dụng pipeline preprocess cho một danh sách các chuỗi văn bản.

    Args:
        texts : Danh sách các chuỗi văn bản đầu vào.
        **kwargs: Các tham số truyền thẳng vào hàm preprocess().

    Returns:
        Danh sách các chuỗi đã tiền xử lý.

    Example:
        >>> preprocess_batch(["Hello World!", "Xin chào 2024!"])
        ['hello world', 'xin chào']
    """
    if not isinstance(texts, list):
        raise TypeError(f"Đầu vào phải là list, nhận được {type(texts).__name__}")

    return [preprocess(t, **kwargs) for t in texts]


# ─────────────────────────────────────────────
# Demo nhanh khi chạy trực tiếp
# ─────────────────────────────────────────────

if __name__ == "__main__":
    samples = [
        "  Hello, World!!!  ",
        "Năm 2024 có 365 ngày & rất nhiều sự kiện @#$%!",
        "Tiền xử lý (Preprocessing) là bước QUAN TRỌNG #1 trong NLP!!!",
        "Python 3.11 ra mắt vào 10/2022 — tốc độ tăng ~60%.",
        "   nhiều    khoảng   trắng   liên   tiếp   ",
    ]

    print("=" * 60)
    print("DEMO TIỀN XỬ LÝ VĂN BẢN")
    print("=" * 60)

    for i, text in enumerate(samples, 1):
        result = preprocess(text)
        print(f"\n[{i}] Gốc   : {repr(text)}")
        print(f"    Kết quả: {repr(result)}")

    print("\n" + "=" * 60)
    print("BATCH PROCESSING")
    print("=" * 60)
    batch_results = preprocess_batch(samples)
    for original, processed in zip(samples, batch_results):
        print(f"  {repr(original):<55} → {repr(processed)}")