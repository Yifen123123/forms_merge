# Meeting Form Classification System

本專案目標為：

> 根據客服通話紀錄（STT `.txt`），自動預測應開立的會辦單類別。

系統採用：

- **Embedding Model：** `multilingual-e5-small`
- **Classifier：** `SVM (scikit-learn)`
- **Similarity Search：** `FAISS`
- **Evaluation：** `LOOCV`

---

# System Architecture

## Offline Training Pipeline

```text
forms_type_mapping.json
        +
calls/*.txt
        ↓
classifier_dataset.json
        ↓
Long Text Chunking
        ↓
E5 Embedding
        ↓
Weighted Pooling
        ↓
Call-level Vector
        ↓
 ┌──────────────┬──────────────┬──────────────┐
 ↓              ↓              ↓
SVM Training   LOOCV Eval     Vector Index
 ↓                              ↓
svm_model                       FAISS
```

---

## Online Prediction Pipeline

```text
new_call.txt
    ↓
Long Text Chunking
    ↓
E5 Embedding
    ↓
Weighted Pooling
    ↓
Call Vector
    ↓
┌──────────────┬──────────────┐
↓              ↓
SVM            FAISS
↓              ↓
分類結果        Top-K 相似案例
        ↓
Final Decision
```

---

# Detailed Workflow

## 1. Dataset Preparation

輸入資料：

### `forms_type_mapping.json`

定義：

- `unit`（會辦單位或代碼）
- `category`（會辦單類別）
- `call_id`

例如：

```json
[
  {
    "doc_type": "行政會辦單",
    "unit": "保全部",
    "category": "地址變更",
    "calls": [
      {
        "call_id": "123456789"
      }
    ]
  }
]
```

---

### `calls/*.txt`

每一筆客服通話紀錄。

例如：

```text
calls/
├── 123456789.txt
├── 987654321.txt
```

---

轉換後產生：

### `classifier_dataset.json`

每一筆通話對應一筆 training sample。

---

# 2. Long Text Chunking

由於 embedding model 有 token 限制，因此長文本需切 chunk。

採用：

## Sliding Window Chunking

設定：

```python
chunk_size = 400
stride = 200
```

例如：

```text
chunk1: token 0~400
chunk2: token 200~600
chunk3: token 400~800
...
```

目的：

- 完整覆蓋全文
- 避免 chunk boundary information loss

---

# 3. E5 Embedding

每個 chunk 轉換成語意向量。

使用：

```text
multilingual-e5-small
```

輸入格式：

```text
passage: <chunk_text>
```

輸出：

```text
384-dimensional vector
```

---

# 4. Weighted Pooling

每個 chunk embedding 合併成一個通話向量。

採用：

## Position Weighting

由於客服通話：

- 前段：開場、身份驗證
- 中段：主要問題描述
- 後段：處理確認

權重設計：

```python
prefix = 0.9
middle = 1.2
suffix = 1.0
```

---

計算方式：

## Weighted Average Pooling

\[
v_{call}
=
\frac{
\sum_i w_i v_i
}{
\sum_i w_i
}
\]

輸出：

```text
Call-level Vector (384 dimensions)
```

---

# 5. Model Training

## SVM Classification

使用：

```text
scikit-learn SVC(kernel="linear")
```

用途：

將：

```text
Call Vector
```

映射至：

```text
Meeting Form Category
```

即：

\[
X \rightarrow y
\]

---

# 6. Model Evaluation

採用：

## Leave-One-Out Cross Validation (LOOCV)

方法：

每次：

```text
N-1 samples → training
1 sample → testing
```

目的：

- 評估小樣本資料效果
- 找出容易混淆類別

輸出：

- Accuracy
- Classification Report
- Confusion Matrix

---

# 7. Vector Retrieval

使用：

```text
FAISS
```

用途：

建立歷史通話向量索引。

---

輸入：

```text
Call Vector
```

輸出：

```text
Top-K Similar Cases
```

例如：

```json
[
  {
    "call_id": "123456789",
    "label_name": "保全部__地址變更",
    "similarity": 0.91
  }
]
```

---

# Online Prediction

輸入：

```text
new_call.txt
```

系統流程：

```text
Text
↓
Chunking
↓
E5 Embedding
↓
Weighted Pooling
↓
Call Vector
```

---

並行執行：

## SVM

輸出：

```json
{
  "label_name": "保全部__地址變更",
  "confidence": 0.82
}
```

---

## FAISS

輸出：

```text
Top-K Similar Cases
```

---

# Final Decision

綜合：

- SVM Prediction
- SVM Confidence
- FAISS Similar Cases

輸出最終分類結果。
