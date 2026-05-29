# Customer Service Call Classification System

## 專案目標

本專案旨在建立一套基於客服通話逐字稿（STT Transcript）的自動化會辦分類系統。

系統透過：

1. 通話資料清洗
2. 分類標籤建立
3. 類別整併
4. 訓練/測試資料切分
5. LLM 分類知識庫建構
6. 向量資料庫建立
7. Two-Stage RAG 分類評估

完成客服通話到會辦類別的自動分類流程。

---

# 系統架構

```text
uncleaned_data
        │
        ▼
00_clean_data.py
        │
        ▼
data
        │
        ▼
01_build_label_json.py
        │
        ▼
classifier_dataset.json
        │
        ▼
02_merge_class_rebuild_json.py
        │
        ▼
data_merged
        │
        ▼
03_split_train_test.py
        │
        ├── train_data
        └── test_data
        │
        ▼
04_build_category_knowledge_base.py
        │
        ▼
category_knowledge_base.json
        │
        ▼
05_build_category_faiss_index.py
        │
        ▼
FAISS Vector Database
        │
        ▼
06_test_two_stage_category_rag.py
        │
        ▼
two_stage_rag_test_results.json
two_stage_rag_test_summary.json
```

---

# 原始資料格式

資料以資料夾結構表示分類標籤：

```text
data/
├── 行政會辦單/
│   ├── 單位A/
│   │   ├── 類別1/
│   │   │   ├── 001.txt
│   │   │   ├── 002.txt
│   │   │   └── ...
│   │   └── 類別2/
│   │
│   └── 單位B/
│
└── 業務會辦單/
```

其中：

```text
doc_type = 行政會辦單 / 業務會辦單
unit      = 會辦單位
category  = 會辦類別
call_id   = txt 檔名
```

---

# 00_clean_data.py

## 目的

清洗 STT 原始逐字稿。

---

## 處理內容

移除：

```text
.wav 檔名
日期資訊
出席者資訊
會議紀錄標頭
時間戳記
空白列
```

統一說話者格式：

```text
左音軌:
→
L:

右音軌:
→
R:
```

並將：

```text
L:
您好

R:
我要補發保單
```

轉換為：

```text
L: 您好
R: 我要補發保單
```

---

## 輸出

```text
data/
```

---

# 01_build_label_json.py

## 目的

將資料夾結構轉換為分類標籤資料。

---

## 處理邏輯

掃描：

```text
doc_type
unit
category
call_id
```

建立：

```json
{
    "call_id": "001",
    "doc_type": "行政會辦單",
    "unit": "契約服務部",
    "category": "保單補發",
    "label_name": "契約服務部_保單補發"
}
```

---

## 輸出

```text
processed/classifier_dataset.json
```

---

# 02_merge_class_rebuild_json.py

## 目的

處理實務上的類別整併需求。

---

## 問題

實務上可能存在：

```text
單位A / 類別A
單位A / 類別B
```

實際上應視為同一類。

甚至：

```text
單位B / 類別A
↓
單位A / 類別B
```

---

## 處理邏輯

讀取：

```text
merge_rules.csv
```

依照規則：

```text
(old_unit, old_category)
↓
(new_unit, new_category)
```

重新建立：

```text
data_merged/
```

並同步更新標籤資料。

---

## 輸出

```text
data_merged/
processed/classifier_dataset.json
```

---

# 03_split_train_test.py

## 目的

建立訓練集與測試集。

---

## 原因

分類規則建立與分類評估不能使用同一批資料。

否則測試結果將失去意義。

---

## 處理邏輯

以：

```text
label_name
```

為單位進行分層切分。

例如：

```text
100筆
↓
80 Train
20 Test
```

確保各分類都保留測試樣本。

---

## 輸出

```text
train_data/
test_data/

processed/train_dataset.json
processed/test_dataset.json
```

---

# 04_build_category_knowledge_base.py

## 目的

利用 LLM 從訓練資料中整理分類規則。

---

## 核心概念

系統不直接使用通話資料進行分類。

而是先建立：

```text
分類知識庫
```

作為後續檢索與推理依據。

---

## 處理邏輯

對每個分類：

```text
label_name
```

抽取多筆通話案例。

交由 LLM 整理：

* 分類定義
* 關鍵詞
* 客戶意圖
* 判斷規則
* 必要證據
* 不應歸類情況

---

## 輸出

```text
processed/category_knowledge_base.json
```

---

# 05_build_category_faiss_index.py

## 目的

建立分類規則向量資料庫。

---

## 問題

當分類數量增加時：

```text
50+
100+
200+
```

不適合每次都將所有分類規則交給 LLM。

---

## 處理邏輯

將：

```text
category_knowledge_base.json
```

轉換為：

```text
Embedding Vector
```

使用模型：

```text
intfloat/multilingual-e5-small
```

建立：

```text
FAISS Index
```

供後續 RAG 搜尋使用。

---

## 輸出

```text
processed/category_faiss/

├── category_rules.index
├── category_rules_metadata.json
├── category_rule_vectors.npy
└── category_faiss_config.json
```

---

# 06_test_two_stage_category_rag.py

## 目的

評估整個分類系統。

---

# 第一階段：Unit Classification

先判斷：

```text
doc_type
unit
```

例如：

```text
行政會辦單
契約服務部
```

---

## 為什麼要先判斷 Unit

若直接從所有類別搜尋：

```text
全部類別
↓
直接分類
```

將導致：

* Prompt 過長
* 類別混淆
* 檢索範圍過大

因此先縮小搜尋空間：

```text
通話
↓
Unit
```

---

# 第二階段：Category Classification

在預測出的 Unit 範圍內：

```text
契約服務部
├── 保單補發
├── 契約變更
├── 地址修改
```

再進行細分類。

---

## 處理流程

```text
通話內容
↓
Embedding
↓
FAISS Retrieval
↓
Top-K Candidates
↓
LLM Re-ranking
↓
Final Category
```

---

# 評估結果

系統紀錄：

```text
真實分類
預測分類
Confidence
檢索候選
推理依據
```

並計算：

---

## Unit Accuracy

評估：

```text
(doc_type, unit)
```

是否正確。

---

## Category Accuracy

評估：

```text
category
```

是否正確。

---

## Final Accuracy

評估：

```text
label_name
```

是否完全正確。

---

## 輸出

詳細結果：

```text
processed/two_stage_rag_test_results.json
```

統計結果：

```text
processed/two_stage_rag_test_summary.json
```

例如：

```json
{
    "unit_accuracy": 0.87,
    "category_accuracy": 0.79,
    "final_accuracy": 0.76
}
```

---

# 核心設計理念

本系統採用：

```text
Rule Generation
+
Vector Retrieval
+
LLM Reasoning
```

三層架構。

```text
Train Data
↓
LLM
↓
Category Knowledge Base
↓
Embedding
↓
FAISS
↓
Top-K Retrieval
↓
LLM Re-ranking
↓
Final Classification
```

目標並非讓 LLM 直接猜測分類結果，而是透過知識庫與向量檢索先縮小搜尋空間，再利用 LLM 完成最終判斷，提高分類穩定性、可解釋性與可維護性。
