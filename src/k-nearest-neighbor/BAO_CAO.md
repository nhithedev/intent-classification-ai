# Báo cáo chi tiết: Route KNN & toàn bộ luồng xử lý dự án

> Tài liệu này giải thích **mọi file** liên quan tới bài toán Intent Classification
> trên **branch hiện tại** (branch gốc, trước khi có nhánh KNN riêng), tập trung
> vào route KNN nhưng bao quát cả `dataset/`, `src/input/`, `src/utils/`,
> `src/preprocess/` và `main.py`. Mục tiêu: đọc xong là hiểu **file nào đang được
> dùng, file nào không, và tại sao**.

---

## 0. Bản đồ nhanh — File nào ĐANG DÙNG, file nào MỒ CÔI

| File / Thư mục | Trạng thái | Vai trò |
|----------------|-----------|---------|
| `dataset/data/data_full.json` | ✅ **ĐANG DÙNG** | Dữ liệu gốc CLINC150 |
| `dataset/data/data_{small,imbalanced,oos_plus}.json` | ⚪ Không dùng | Các phiên bản dataset khác |
| `dataset/stopwords.txt` | ⚪ **KHÔNG dùng** (trên branch này) | split.py không import |
| `src/utils/split.py` | ✅ **ĐANG DÙNG** (chạy thủ công) | Sinh ra `src/input/*.txt` |
| `src/utils/tfidfcal.py` | ✅ **CORE** | Nguồn DUY NHẤT của `TfIdfVectorizer` |
| `src/utils/labelEncode.py` | ✅ **CORE** | Nguồn DUY NHẤT của `LabelEncoder` |
| `src/utils/dataio.py` | ✅ **CORE** | `load_split` + `INPUT_DIR`/`MODEL_DIR` |
| `src/input/*.txt` (6 file) | ✅ **ĐANG DÙNG** | Dữ liệu train/val/test |
| `src/preprocess/*` (5 file) | 🔴 **MỒ CÔI** (cả cụm) | Bộ tiền xử lý không nối vào pipeline thật |
| `src/logistic-regression/mrl.py` | ✅ Model | Logistic Regression (import shared từ `utils/`) |
| `src/naive-bayes/mnb.py` | ✅ Model | Naive Bayes |
| `src/decision-tree/mdt.py` | ✅ Model | Decision Tree (TF-IDF riêng) |
| `src/k-nearest-neighbor/mknn.py` | ✅ Model | **KNN (file chính của route này)** |
| `src/k-nearest-neighbor/testMKNN.py` | ✅ Eval | Đo metrics KNN trên test |
| `src/random-forest/mrf.py` | ✅ Model | **Random Forest** (bagging + Gini, numpy) |
| `src/random-forest/testMRF.py` | ✅ Eval | Đo metrics RF trên test |
| `src/model/*.pkl` | ✅ Output | 5 model đã train |
| `main.py` | ✅ CLI | predict / chat / eval cho cả 5 model |

> 🔴 **"Mồ côi"** = file tồn tại trong repo nhưng **không file nào import/chạy nó**
> trong luồng huấn luyện thực tế. Chúng là bản nháp/thư viện dự phòng. Xem mục 4 & 5.

---

## 1. Luồng dữ liệu end-to-end (cái gì → cái gì)

```
dataset/data/data_full.json          (dữ liệu gốc, JSON)
        │
        │  ❶ chạy: python src/utils/split.py
        ▼
src/input/{train,val,test}_corpus.txt   ← text RAW (chỉ .strip(), KHÔNG preprocessing)
src/input/{train,val,test}_labels.txt   ← nhãn tương ứng
        │
        │  ❷ chạy train: python src/k-nearest-neighbor/mknn.py
        │     (tương tự cho mrl.py / mnb.py / mdt.py)
        │     - TfIdfVectorizer:  doc.lower().split()  → vector TF-IDF
        │     - LabelEncoder:     nhãn chữ → số
        │     - KNN.fit():        L2-normalize + lưu training data
        ▼
src/model/KNN_model.pkl   = {"vectorizer", "label_encoder", "model"}
        │
        ├─ ❸ đánh giá: python src/k-nearest-neighbor/testMKNN.py  → metrics test
        └─ ❹ dùng thật: python main.py chat                       → so sánh 5 model
```

**Điểm cốt lõi cần nhớ:** trên branch này **không có bước preprocessing riêng**.
"Tiền xử lý" duy nhất là `doc.lower().split()` nằm bên trong `TfIdfVectorizer`
(lowercase + tách theo khoảng trắng). Stopwords, ký tự đặc biệt, chữ số đều **được
giữ nguyên** — và TF-IDF tự động hạ trọng số các từ phổ biến qua IDF.

---

## 2. `dataset/` — Dữ liệu gốc

### 2.1 `dataset/data/data_full.json` ✅ (file đang dùng)

Dataset **CLINC150** (xem `meta.txt`): 150 nhãn ý định in-scope (banking, travel,
auto...) + 1 lớp out-of-scope (OOS). Cấu trúc JSON gồm **6 khối**:

| Key trong JSON | Số mẫu | branch này dùng? |
|----------------|--------|------------------|
| `train`     | 15 000 | ✅ → train (in-scope) |
| `val`       | 3 000  | ✅ → val (in-scope) |
| `test`      | 4 500  | ✅ → test (in-scope) |
| `oos_val`   | 100    | ✅ → trộn vào val |
| `oos_test`  | 1 000  | ✅ → trộn vào test |
| `oos_train` | 100    | ❌ **bị bỏ** (không học nhãn OOS) |

Mỗi mẫu có dạng `[text, label]`, ví dụ `["what time does the bank open", "banking_hours"]`.

### 2.2 Các file khác trong `dataset/`

- `data_small.json`, `data_imbalanced.json`, `data_oos_plus.json` — biến thể khác của
  CLINC150 (ít mẫu hơn / mất cân bằng / nhiều OOS hơn). **Không dùng** trong dự án này.
- `meta.txt` — mô tả dataset (nguồn, trích dẫn, thống kê). Chỉ để tham khảo.
- `LICENSE` — giấy phép dataset.
- `stopwords.txt` — danh sách ~91 stopwords tiếng Anh. **KHÔNG được dùng trên branch
  này** vì `split.py` ở đây không xóa stopwords (xem mục 4.1).

---

## 3. `src/input/` — Dữ liệu đã tách (6 file)

Đây là đầu vào trực tiếp cho mọi model. Gồm **3 cặp** (corpus + labels):

| File corpus | File labels | Số dòng | Nội dung |
|-------------|-------------|---------|----------|
| `train_corpus.txt` | `train_labels.txt` | 15 000 | Chỉ in-scope — để **học** |
| `val_corpus.txt`   | `val_labels.txt`   | 3 100  | 3 000 in-scope + 100 OOS — để **tune** (chọn k, ngưỡng) |
| `test_corpus.txt`  | `test_labels.txt`  | 5 500  | 4 500 in-scope + 1 000 OOS — để **đánh giá cuối** |

**Tại sao cần chia 3 tập?**
- **train**: model học từ đây. Không chứa OOS để model chỉ học 150 nhãn thật.
- **val** (validation): dùng để **chọn siêu tham số** (KNN chọn `k`, chọn ngưỡng OOS)
  mà không "nhìn trộm" test. Có OOS để đo khả năng phát hiện OOS.
- **test**: chỉ chạm vào **một lần** ở cuối để báo cáo. Có nhiều OOS (1 000) để kiểm
  tra mô hình có nhận ra câu ngoài phạm vi không.

**Quan hệ corpus ↔ labels:** dòng thứ `i` của `*_corpus.txt` là câu, dòng thứ `i`
của `*_labels.txt` là nhãn của câu đó. Hai file luôn cùng số dòng, ghép theo chỉ số.

**Định dạng text:** RAW, ví dụ dòng 1 của `train_corpus.txt`:
```
what expression would i use to say i love you if i were an italian
```
→ còn nguyên "what", "would", "i" (stopwords) vì branch này không xóa chúng.

**File này sinh ra từ đâu?** Chạy `python src/utils/split.py` (mục 4.1).

---

## 4. `src/utils/`

### 4.1 `split.py` ✅ (ĐANG DÙNG — chạy thủ công)

Script sinh ra 6 file trong `src/input/`. Logic:
1. Đọc `dataset/data/data_full.json`.
2. Ghép các khối: `train` = `train`; `test` = `test` + `oos_test`; `val` = `val` + `oos_val`.
   (Bỏ `oos_train` — không cho model học lớp OOS.)
3. Ghi thẳng `text.strip()` ra file — **KHÔNG lowercase, KHÔNG xóa ký tự/số/stopwords**.

> ⚠️ **Đây là điểm khác biệt quan trọng giữa branch.** Trên branch `nhipham/knn`,
> `split.py` có thêm bước xóa stopwords + ký tự đặc biệt → vocab nhỏ hơn nhưng
> **điểm số THẤP hơn** (LR 0.79, NB 0.77). Trên branch hiện tại text để raw →
> giữ được signal từ function words → **điểm cao hơn** (LR 0.89, NB 0.87).
> Kết luận thực nghiệm: với CLINC150, xóa stopwords làm **giảm** độ chính xác.

### 4.2 `tfidfcal.py` ✅ (CORE — nguồn chung)

Định nghĩa `TfIdfVectorizer` — **nguồn DUY NHẤT** cho cả 4 model dùng chung
(LR, NB, KNN, RF). Sau refactor, `mrl.py`/`mnb.py`/`mknn.py`/`mrf.py` đều
`from tfidfcal import TfIdfVectorizer` thay vì copy hoặc import từ nhau.
(`mdt.py` vẫn giữ bản TF-IDF riêng vì có `min_df` + sublinear scaling — biến thể
khác có chủ đích.)

### 4.3 `labelEncode.py` ✅ (CORE — nguồn chung)

Định nghĩa `LabelEncoder` — nguồn DUY NHẤT cho LR/NB/KNN. Các model
`from labelEncode import LabelEncoder`.

### 4.4 `dataio.py` ✅ (CORE — nguồn chung, mới thêm)

Chứa `load_split()` (đọc cặp corpus/labels) và đường dẫn chuẩn `INPUT_DIR`,
`MODEL_DIR`. Trước đây những thứ này nằm trong `mrl.py`; nay tách ra `utils/` để
đúng vai trò "thư viện chung".

> **Kiến trúc sau refactor** (loại bỏ phụ thuộc ngược NB/KNN → LR):
> ```
> utils/tfidfcal.py   (TfIdfVectorizer)  ─┐
> utils/labelEncode.py (LabelEncoder)    ─┼─► mrl.py, mnb.py, mknn.py  (đều import từ utils)
> utils/dataio.py     (load_split, paths)─┘
> ```
> `mrl.py` vẫn re-export các tên này (vì nó cũng import) nên `testMRL.py`/`testMNB.py`
> /`main.py` cũ không cần sửa nhiều. `mdt.py` độc lập (vectorizer riêng).

---

## 5. `src/preprocess/` — Bộ tiền xử lý KHÔNG được dùng (cả cụm 🔴)

Đây là một **bộ công cụ NLP hoàn chỉnh và độc lập**, gồm 5 file gọi lẫn nhau:

```
data_processor.py          (CLI: python data_processor.py --data ... --stopwords ...)
      │ import
      ▼
preprocess_pipeline.py     (class NLPPipeline + PipelineConfig — điều phối)
      │ import
      ├──► text_process.py     (lowercase, xóa ký tự đặc biệt, xóa số, chuẩn hóa space)
      ├──► stopword_remove.py  (class StopwordsRemover — xóa stopwords từ file .txt)
      └──► tokenizer.py        (Whitespace / Ngram / Subword / Syllable tokenizer)
```

**Từng file làm gì:**
- `text_process.py` — các hàm làm sạch text: `lowercase()`, `remove_special_chars()`,
  `remove_numbers()`, `normalize_whitespace()`, và `preprocess()` gộp chung.
- `stopword_remove.py` — class `StopwordsRemover` đọc file stopwords và lọc token.
- `tokenizer.py` — nhiều kiểu tokenizer (tách khoảng trắng, n-gram, subword, âm tiết).
- `preprocess_pipeline.py` — `NLPPipeline` ghép 3 module trên thành 1 pipeline cấu hình
  được, xuất ra token list / token ids / BoW / TF-IDF / padded sequences.
- `data_processor.py` — CLI: đọc JSON → chạy pipeline → ghi `corpus.txt` + `labels.txt`.

**Tại sao gọi là "mồ côi"?** Cụm này **chỉ import lẫn nhau**, không có file nào ngoài
cụm (split.py, các model, main.py) import nó. Pipeline thật sinh `src/input/` bằng
`split.py` (ghi raw), còn việc tách từ + TF-IDF do `TfIdfVectorizer` trong `mrl.py`
đảm nhiệm. Nói cách khác: `src/preprocess/` là một **phương án tiền xử lý thay thế
chưa bao giờ được nối vào** luồng huấn luyện hiện tại.

> Nếu sau này muốn dùng: có thể thay `split.py` bằng `data_processor.py` để sinh
> `src/input/`. Nhưng theo thực nghiệm ở mục 4.1, xóa stopwords làm giảm điểm, nên
> hiện tại để raw là lựa chọn tốt hơn.

---

## 6. `src/k-nearest-neighbor/` — Route KNN (trọng tâm)

### 6.1 `mknn.py` — Toàn bộ logic KNN + pipeline train

File này **import** `TfIdfVectorizer`, `LabelEncoder`, `load_split` từ `mrl.py`
(KHÔNG sửa `mrl.py` → không ảnh hưởng LR/NB/DT). Các thành phần:

**a) Class `KNearestNeighbors`** — KNN tự viết bằng NumPy:
- `__init__(k, weighted, normalize, batch_size)` — cấu hình.
- `_l2(X)` — **L2 normalization**: chia mỗi vector cho độ dài của nó. Đây là **đòn bẩy
  accuracy lớn nhất** cho KNN: trên vector đã chuẩn hóa L2, khoảng cách Euclidean
  tương đương cosine similarity → giảm tác hại của "curse of dimensionality" (không
  gian 5863 chiều). **Chỉ nằm trong class KNN**, không lan sang model khác.
- `fit(X, y)` — lazy learner: chỉ L2-normalize rồi **lưu** training data (KNN không
  "học" gì lúc fit).
- `_distances(X_batch)` — tính khoảng cách Euclidean bằng **công thức khai triển**
  `||a-b||² = ||a||² + ||b||² - 2a·b` → biến thành phép nhân ma trận (BLAS) rất nhanh.
- `kneighbors(X, n)` — tìm `n` hàng xóm gần nhất, **đã sort**. Dùng `argpartition`
  (nhanh hơn sort toàn bộ) + xử lý **theo batch 256 câu** để tiết kiệm RAM (ma trận
  khoảng cách đầy đủ 5500×15000 ≈ 660 MB; batch chỉ ~30 MB/lần).
- `_vote(dist, idx, k, weighted)` — bỏ phiếu:
  - `weighted=False`: mỗi hàng xóm 1 phiếu (đa số thắng).
  - `weighted=True`: phiếu = `1/(distance+ε)` → hàng xóm gần đóng góp nhiều hơn.
  - `confidence` = (tổng phiếu nhãn thắng) / (tổng phiếu) → dùng cho OOS.
- `predict_with_oos(X, threshold)` — trả `(nhãn, confidence)`; nếu confidence < threshold
  → trả `-1` (OUT OF SCOPE).
- `predict(X)` — gọi `predict_with_oos(threshold=0.0)` → luôn trả nhãn (không lọc OOS).

**b) `evaluate(...)`** — in In-scope Accuracy / OOS Recall / False Rejection trên val.

**c) `tune_hyperparams(...)`** — **tự động chọn k và chế độ vote tốt nhất**:
- Tính `kneighbors` với k lớn nhất **MỘT LẦN**, rồi quét mọi `k ∈ {1,3,5,7,9,11,15,21,31}`
  và cả 2 chế độ vote (tái dùng kết quả → rất nhanh).
- Chọn cấu hình cho **in-scope accuracy** cao nhất trên val.
- Kết quả tune trên branch này: **k=11, weighted=True** (val in-scope acc 80.27%).

**d) Pipeline `main`** — đọc data → lọc OOS khỏi train → TF-IDF → tune → train model
cuối → lưu `src/model/KNN_model.pkl` dạng `{"vectorizer", "label_encoder", "model"}`.

### 6.2 `testMKNN.py` — Đánh giá trên test set

- Load `KNN_model.pkl`, lọc các mẫu nhãn lạ (OOS không có trong label_encoder),
  dự đoán với `threshold=0.0` (phân loại thuần, không lọc OOS), tính macro
  Accuracy/Precision/Recall/F1 bằng `sklearn.metrics`.
- Có xử lý **pickle `__main__`**: vì `mknn.py` được chạy trực tiếp khi train, class
  `KNearestNeighbors` được pickle dưới module `__main__`. File test đăng ký lại class
  vào `__main__` để `pickle.load` tìm thấy (xem mục 11).

### 6.3 `BAO_CAO.md` (file này) + `KNN_model.pkl`

- `BAO_CAO.md` — tài liệu bạn đang đọc.
- `src/model/KNN_model.pkl` — model KNN đã train (k=11, weighted, L2-norm).

---

## 7. Các model khác (tóm tắt vai trò)

- `src/logistic-regression/mrl.py` — **chỉ còn** model Logistic Regression. Sau
  refactor, nó import `TfIdfVectorizer`/`LabelEncoder`/`load_split` từ `utils/`
  (giống NB, KNN, RF) thay vì tự định nghĩa. Không còn là "thư viện chung".
- `src/naive-bayes/mnb.py` — Multinomial Naive Bayes (import shared từ `utils/`).
- `src/decision-tree/mdt.py` — Decision Tree phân cấp, **tự định nghĩa `TfIdfVectorizer`
  riêng** (có `min_df`, sublinear scaling) — độc lập với `mrl.py`.
- `src/random-forest/mrf.py` — **Random Forest** tự viết bằng NumPy (xem mục 7.1).

### 7.1 `src/random-forest/mrf.py` — Random Forest (song song với DT)

RF được thêm vào **song song với DT** (không thay thế), vừa tăng bộ so sánh vừa cho
thấy cách bagging giải quyết nhược điểm overfit của 1 cây đơn.

**Ý tưởng cốt lõi:** 1 cây quyết định = variance cao, dễ overfit.
Random Forest = trung bình nhiều cây nhờ 2 nguồn ngẫu nhiên:
- **Bagging**: mỗi cây học trên bootstrap sample (lấy có hoàn lại) → các cây thấy dữ liệu khác nhau.
- **Random features**: mỗi node chỉ xét √(số chiều) đặc trưng ngẫu nhiên → các cây ít tương quan → trung bình mạnh hơn.

**Quyết định thiết kế:**
- **Split nhị phân "từ có/vắng"** (`X[:, f] > 0`): với TF-IDF thưa, câu hỏi phân biệt chủ yếu là "có chứa từ X không" — tránh quét ngưỡng liên tục, nhanh hơn ~100×.
- **Tiêu chí split: Gini impurity** — Gini = 1 − Σp² = đo độ hỗn tạp nút con.
- **OOB score**: mỗi cây bỏ qua ~36% mẫu bootstrap → đánh giá miễn phí (OOB ≈ 68.9%).
- **Confidence = tỉ lệ phiếu** (`votes_winner / n_trees`): vì 80 cây và 150 lớp, confidence thường nằm 0.15–0.35 → ngưỡng OOS mặc định của RF là **0.15** (thấp hơn LR/NB/KNN).
- **Lưu ý bộ nhớ**: cài bằng index-based recursion — `_build(X, y, indices, depth)` truyền mảng chỉ số nhỏ thay vì sao chép mảng X to (363 MB) ở mỗi tầng đệ quy.

**Cấu hình chốt:** `n_estimators=80, max_depth=50, min_samples_split=2, max_features="sqrt", min_df=2`

---

## 8. `main.py` — CLI tích hợp 5 model

3 lệnh con:

| Lệnh | Công dụng | Ví dụ |
|------|-----------|-------|
| `predict` | Đoán 1 câu | `python main.py predict "what time is it" -m knn` |
| `chat`    | Chat liên tục, **mặc định hiện cả 5 model** | `python main.py chat` |
| `eval`    | Đo metrics test, mặc định cả 5 | `python main.py eval -m all` |

**Chế độ chat (yêu cầu chính):** mặc định `-m all` → load cả 5 model, mỗi câu nhập
vào in **bảng so sánh**:
```
| Model                  | Intent                   | Confidence |
| Logistic Regression    | weather                  | 0.92       |
| Naive Bayes            | weather                  | 1.00       |
| Decision Tree          | weather                  | 1.00       |
| K-Nearest Neighbors    | weather                  | 0.64       |
| Random Forest          | weather                  | 0.30       |
```

**Cơ chế quan trọng trong main.py:**
- Mỗi model có **ngưỡng OOS mặc định riêng** (LR/NB=0.50, DT=0.35, KNN=0.30, RF=0.15) vì
  thang confidence khác nhau (RF dùng tỉ lệ phiếu, nên thường thấp hơn). Có thể ghi đè bằng `-t`.
- Xử lý **pickle `__main__`**: import `KNearestNeighbors`, `HierarchicalDecisionTree`, `RandomForest`
  vào namespace của `main.py`, đồng thời `sys.modules['mknn']=mknn`, `sys.modules['mrf']=mrf` v.v.
  để `pickle.load` hoạt động dù model được train bằng cách chạy file trực tiếp.

---

## 9. Cách KNN đạt accuracy tốt nhất (0.667 → 0.813)

| Kỹ thuật | Vì sao giúp |
|----------|-------------|
| **L2 normalization** | Euclidean trên vector chuẩn hóa ≈ cosine → giảm curse of dimensionality (đòn bẩy lớn nhất). |
| **Distance-weighted voting** | Hàng xóm gần đáng tin hơn hàng xóm xa; mọi k đều cho điểm ≥ vote đều. |
| **Tune k trên val** | Quét k và chọn k=11 (tốt hơn k=15 mặc định). Tính hàng xóm 1 lần rồi tái dùng → tune nhanh. |
| **Batch + công thức khai triển** | Chạy được trên RAM thường + tận dụng BLAS. |

Tất cả các tối ưu trên **chỉ nằm trong `mknn.py`** → không đụng tới LR/NB/DT.

---

## 10. Kết quả cuối — Bảng xếp hạng 5 model (test set, macro F1)

| Model | Accuracy | Precision | Recall | F1 macro |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.8911 | 0.8965 | 0.8911 | **0.8905** |
| Naive Bayes | 0.8742 | 0.8797 | 0.8742 | **0.8722** |
| **K-Nearest Neighbors** | 0.8162 | 0.8270 | 0.8162 | **0.8127** |
| **Random Forest** | 0.7167 | 0.7958 | 0.7167 | **0.7162** |
| Decision Tree | 0.5389 | 0.7326 | 0.5224 | **0.5892** |

> RF xếp thứ 4, cao hơn DT đơn lẻ — đúng kỳ vọng lý thuyết: bagging giảm variance
> của cây đơn. KNN vẫn nhỉnh hơn RF trên CLINC150 vì L2-normalized TF-IDF rất phù hợp
> với cosine distance trong không gian text ngắn.

---

## 11. Các điểm "gotcha" / lưu ý kỹ thuật

1. **Pickle `__main__`**: train bằng `python mknn.py` → class lưu dưới `__main__`.
   Khi load ở file khác phải đưa class vào `__main__` (đã xử lý trong `testMKNN.py`
   và `main.py`). Nếu không sẽ lỗi `AttributeError: Can't get attribute`.
2. **Lệch tiền xử lý train ↔ chat**: text train trong `src/input/` là raw; câu nhập
   ở `chat` chỉ qua `.lower().split()` trong vectorizer. Nếu gõ dấu câu/số (vd "open?"),
   token "open?" sẽ không khớp "open" trong vocab → bị bỏ. Đây là hành vi **chung cho
   cả 4 model** (không riêng KNN). Muốn chuẩn hơn có thể làm sạch input trước khi transform.
3. **File mồ côi**: chỉ còn `src/preprocess/*` (cụm 5 file) là không nằm trong luồng
   chạy. (`utils/tfidfcal.py` & `labelEncode.py` đã trở thành nguồn chung sau refactor.)
4. **Encoding Windows**: PowerShell mặc định `cp1258` không in được tiếng Việt → mọi
   file thêm `sys.stdout.reconfigure(encoding='utf-8')`, hoặc đặt `PYTHONUTF8=1`.
5. **Không dùng `sklearn` cho thuật toán**: KNN tự viết bằng NumPy; `sklearn.metrics`
   chỉ dùng để **đo** (accuracy/precision/recall/f1), không phải để phân loại.

---

## 12. Lệnh thường dùng (cheat sheet)

```bash
# (1) Sinh lại dữ liệu input từ JSON gốc
python src/utils/split.py

# (2) Train từng model và lưu pkl
python src/logistic-regression/mrl.py
python src/naive-bayes/mnb.py
python src/decision-tree/mdt.py
python src/k-nearest-neighbor/mknn.py
python src/random-forest/mrf.py

# (3) Đánh giá từng model trên test set
python src/k-nearest-neighbor/testMKNN.py
python src/random-forest/testMRF.py

# (4) Chat so sánh cả 5 model cùng lúc
python main.py chat

# (5) Đánh giá cả 5 model trên test (cần PYTHONUTF8=1 trên Windows)
PYTHONUTF8=1 python main.py eval -m all

# (6) Đoán 1 câu với model cụ thể
python main.py predict "what is the weather today" -m rf
```
