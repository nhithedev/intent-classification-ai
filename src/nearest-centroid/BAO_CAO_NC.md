# 4. Thuật toán Nearest Centroid (Rocchio Classifier)

## 4.1. Tổng quan

Nearest Centroid (hay Rocchio Classifier) là thuật toán phân loại dựa trên nguyên mẫu (prototype-based): mỗi lớp được đại diện bởi **một vector trung tâm duy nhất** (centroid), tính bằng trung bình các vector đặc trưng của các mẫu thuộc lớp đó. Khi phân loại một câu mới, thuật toán tính độ tương đồng của câu đó với tất cả centroid và chọn lớp có centroid gần nhất.

Thuật toán được cài đặt thuần NumPy, không dùng thư viện học máy bên ngoài.

---

## 4.2. Cơ sở lý thuyết

### 4.2.1. Vector hóa văn bản (TF-IDF)

Mỗi câu văn được biểu diễn bằng một vector TF-IDF có số chiều bằng kích thước từ vựng (trong thực nghiệm này là 5.219 chiều). Giá trị mỗi chiều phản ánh tầm quan trọng của một từ đối với câu, tính bằng:

$$\text{TF-IDF}(t, d) = \text{count}(t, d) \times \ln\frac{N}{\text{DF}(t)}$$

trong đó:
- $\text{count}(t, d)$ — số lần từ $t$ xuất hiện trong câu $d$ 
- $N$ — tổng số câu trong tập huấn luyện
- $\text{DF}(t)$ — số câu chứa từ $t$
- $\ln$ — logarithm tự nhiên (cơ số $e$), không có smoothing

### 4.2.2. Chuẩn hóa L2 (L2 Normalization)

Trước khi tính toán, mỗi vector được chuẩn hóa về độ dài đơn vị:

$$\hat{x} = \frac{x}{\|x\|_2}, \quad \|x\|_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}$$

**Mục đích:** loại bỏ ảnh hưởng của độ dài câu. Một câu dài tự nhiên có các giá trị TF lớn hơn, dẫn đến vector "dài" hơn. Nếu không chuẩn hóa, các câu dài sẽ chi phối phép tính trung bình. Sau khi chuẩn hóa L2, mọi câu đều nằm trên một mặt cầu đơn vị, và phép so sánh chỉ còn phụ thuộc vào **hướng** của vector (nội dung ngữ nghĩa), không phụ thuộc độ lớn.

### 4.2.3. Cosine Similarity

Độ tương đồng giữa hai vector được đo bằng cosine của góc giữa chúng:

$$\text{cosine}(a, b) = \frac{a \cdot b}{\|a\|_2 \cdot \|b\|_2}$$

Khi cả hai vector đã được chuẩn hóa L2 (độ dài = 1), công thức rút gọn thành tích vô hướng đơn giản:

$$\text{cosine}(\hat{a}, \hat{b}) = \hat{a} \cdot \hat{b}$$

Giá trị cosine nằm trong khoảng $[0, 1]$ đối với vector TF-IDF (vì mọi thành phần không âm). Cosine càng gần 1 nghĩa là hai câu càng giống nhau về nội dung.

---

## 4.3. Thuật toán chi tiết

### 4.3.1. Giai đoạn huấn luyện (Training)

**Đầu vào:** ma trận TF-IDF $X$ kích thước $(n\_\text{mẫu} \times n\_\text{đặc trưng})$ và mảng nhãn $y$.

**Các bước:**

1. Chuẩn hóa L2 toàn bộ ma trận $X$, thu được $\hat{X}$.
2. Với mỗi lớp $c$ trong 150 lớp, lấy tập con $D_c = \{\hat{x}_i \mid y_i = c\}$ và tính centroid:

$$\mu_c = \frac{1}{|D_c|} \sum_{\hat{x}_i \in D_c} \hat{x}_i$$

3. Chuẩn hóa L2 các centroid:

$$\hat{\mu}_c = \frac{\mu_c}{\|\mu_c\|_2}$$

**Kết quả:** ma trận centroid $\hat{M}$ kích thước $(150 \times n\_\text{đặc trưng})$, mỗi hàng là $\hat{\mu}_c$ của một lớp.

> **Lưu ý:** centroid được tính từ các vector đã chuẩn hóa $\hat{x}_i$, không phải vector thô $x_i$. Sau đó centroid được chuẩn hóa lần hai để đảm bảo tích vô hướng tại bước dự đoán tương đương cosine similarity.

### 4.3.2. Giai đoạn dự đoán (Prediction)

Với mỗi câu mới $x$:

1. Chuẩn hóa L2: $\hat{x} = x / \|x\|_2$
2. Tính cosine similarity tới toàn bộ 150 centroid (phép nhân ma trận):

$$\text{similarities} = \hat{x} \cdot \hat{M}^\top \in \mathbb{R}^{150}$$

3. Chọn lớp có cosine cao nhất:

$$\hat{y} = \arg\max_c \; \text{cosine}(\hat{x}, \hat{\mu}_c)$$

---

## 4.4. Cơ chế phát hiện Out-of-Scope (OOS)

Một ưu điểm tự nhiên của Nearest Centroid là khả năng phát hiện câu ngoài phạm vi. Độ tin cậy (confidence) của dự đoán chính là giá trị cosine similarity tới centroid gần nhất:

$$\text{confidence} = \max_c \; \text{cosine}(\hat{x}, \hat{\mu}_c)$$

Một câu OOS sẽ xa tất cả các centroid, dẫn đến cosine thấp. Bằng cách đặt một ngưỡng $\theta$, quyết định phân loại trở thành:

$$\hat{y} = \begin{cases} \arg\max_c \; \text{cosine}(\hat{x}, \hat{\mu}_c) & \text{nếu } \text{confidence} \geq \theta \\ \text{OOS} & \text{nếu } \text{confidence} < \theta \end{cases}$$

Tín hiệu này có ý nghĩa hình học rõ ràng (khoảng cách tới nguyên mẫu), nên đáng tin cậy hơn so với confidence của Naive Bayes vốn thường bão hòa gần 1.
