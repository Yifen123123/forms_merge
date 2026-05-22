# 通話紀錄會辦分類系統

本專案目標是根據客服通話紀錄，自動判斷該通電話應歸類到哪一個會辦單位與會辦類別。

目前系統分成兩條主線：

1. LLM 分類流程
2. Embedding + FAISS 分類規則檢索流程

---

# 專案核心概念

本專案並不是單純做：

```text
文字 → 相似度分類
```

而是：

```text
通話內容
→ 建立分類知識庫
→ 用分類規則理解類別
→ 利用 Embedding 做 Retrieval
→ 利用 LLM 做 Final Reasoning
```

因此本專案本質上比較接近：

```text
RAG-style Classification System
```

而不是傳統的：

```text
Cosine Similarity Classifier
```

---

# 專案資料來源

## calls/

放置所有通話紀錄 `.txt` 檔案。

檔名需對應 `classifier_dataset.json` 中的 `call_id`。

範例：

```text
calls/
├── 001.txt
├── 002.txt
└── ...
```

---

## processed/classifier_dataset.json

紀錄每一筆通話對應的人工標註分類。

格式：

```json
{
  "call_id": "通話紀錄 id",
  "doc_type": "行政會辦單 / 業務會辦單",
  "unit": "會辦單位",
  "category": "會辦類別",
  "label_name": "會辦單位_會辦類別"
}
```

---

# 專案整體架構

```text
calls/
        +
classifier_dataset.json
        ↓

01_build_category_knowledge_base.py
        ↓
category_knowledge_base.json

==================================================

02_predict_call_with_llm_two_stage.py
        ↓
兩階段 LLM 分類

==================================================

03_build_category_faiss_index.py
        ↓
建立 category rule embeddings
        ↓
建立 FAISS index

==================================================

04_test_category_retrieval.py
        ↓
測試 Top-K category retrieval
```

---

# 目前資料規模

目前：

```text
通話資料：91 筆
分類數量：64 類
```

部分分類只有 1 筆資料。

因此本專案特別強調：

```text
1. 分類知識庫
2. Retrieval
3. LLM reasoning
```

而不是只依靠：

```text
純 supervised classifier
```

---

# 01_build_category_knowledge_base.py

# 功能

根據：

```text
calls/
processed/classifier_dataset.json
```

建立：

```text
processed/category_knowledge_base.json
```

也就是：

```text
分類知識庫
```

---

# 為什麼需要分類知識庫？

原始通話只有：

```text
客服與客戶的對話
```

但 LLM 在分類時需要：

```text
1. 類別定義
2. 類別規則
3. 判斷依據
4. 不應歸類情況
5. 關鍵證據
```

因此先讓 LLM 根據歷史通話整理：

```text
category-level knowledge
```

這會比：

```text
直接拿原始通話做分類
```

更穩定。

---

# 設計邏輯

程式會先根據：

```python
label_name
```

將資料分組。

例如：

```text
保全部_契約變更
├── call_001.txt
├── call_012.txt
└── call_045.txt
```

接著將：

```text
同一分類底下的通話案例
```

交給 LLM。

LLM 會整理出：

```text
1. 此類別通常處理什麼問題
2. 客戶常見需求
3. 判斷規則
4. 關鍵證據
5. 不應該歸類的情況
```

---

# category_knowledge_base.json 結構

每個分類會整理成：

```json
{
  "label_name": "",
  "doc_type": "",
  "unit": "",
  "category": "",

  "definition": "",

  "main_customer_intents": [],
  "keywords": [],
  "decision_rules": [],
  "negative_rules": [],
  "required_evidence_from_call": [],

  "possible_confusing_categories": [],

  "example_call_ids": [],

  "num_examples": 0,
  "data_sufficiency": "",

  "limitations": ""
}
```

---

# 欄位說明

## definition

此類別的整體定義。

---

## main_customer_intents

客戶常見需求。

例如：

```text
修改保單
調整保費
契約變更
```

---

## keywords

常見關鍵詞。

但後續分類不能只靠 keywords。

---

## decision_rules

後續分類時的主要判斷規則。

例如：

```text
若客戶要求修改保單內容
且客服確認契約異動流程
則可歸類為契約變更
```

---

## negative_rules

哪些情況不應該分到此類。

---

## required_evidence_from_call

通話中至少需要出現哪些證據。

例如：

```text
客戶明確要求變更保單
客服確認異動內容
```

---

## data_sufficiency

根據資料量判定：

```text
low    = 1 筆
medium = 2~3 筆
high   = 4 筆以上
```

這在後續分類時非常重要。

因為：

```text
低資料量類別
→ LLM 不應該過度自信
```

---

# 執行方式

```bash
python 01_build_category_knowledge_base.py
```

---

# 輸出

```text
processed/category_knowledge_base.json
```

---

# 02_predict_call_with_llm_two_stage.py

# 功能

輸入一通新的客服電話，使用：

```text
兩階段 LLM 分類
```

判斷它應該屬於哪一個會辦分類。

---

# 為什麼要做兩階段？

原本如果一次把：

```text
全部 64 類
```

都丟給 LLM：

prompt 會非常長。

曾經測到：

```text
75086 字
```

造成：

```text
1. timeout
2. JSON parsing failure
3. hallucination
4. 回傳不存在的分類
```

因此改成：

```text
Stage 1：先判斷 unit
Stage 2：只在該 unit 底下判斷 category
```

---

# Stage 1：判斷會辦單位

程式會從：

```text
category_knowledge_base.json
```

整理出所有 unit：

```json
[
  {
    "unit_index": 0,
    "unit": "保全部"
  },
  {
    "unit_index": 1,
    "unit": "客服部"
  }
]
```

接著 LLM 只能輸出：

```json
{
  "pred_unit_index": 0
}
```

而不是直接輸出：

```text
保全部
```

---

# 為什麼不用 label_name？

因為 LLM 很容易：

```text
改字
縮寫
翻譯
加空格
```

例如：

```text
保全部_契約變更
```

可能變：

```text
保全部-契約變更
契約變更
```

所以現在設計是：

```text
LLM 只選 index
程式負責反查正式 label
```

這樣可以保證：

```text
最後輸出的分類一定存在於 knowledge base 中
```

---

# Stage 2：判斷 category

Stage 1 得到：

```python
pred_unit
```

之後：

```python
candidate_kb = filter_kb_by_unit(kb, pred_unit)
```

只保留該 unit 底下的分類。

例如：

```text
原本 64 類
↓
只剩 4 類
```

然後讓 LLM：

```text
只在這 4 類中選一類
```

---

# retry 機制

由於：

```text
gpt-oss:20b
```

不一定穩定輸出 JSON。

因此：

```python
call_llm_json()
```

內部有 retry 機制。

如果第一次：

```text
不是合法 JSON
```

就會：

```text
自動重新要求模型只輸出 JSON
```

---

# timing 機制

程式使用：

```text
utils/timing_utils.py
```

紀錄：

```text
1. prompt build 時間
2. stage1 llm 時間
3. stage2 llm 時間
4. validation 時間
5. total script 時間
```

輸出：

```json
{
  "timing": {
    "stage1_llm_seconds": 4.2,
    "stage2_llm_seconds": 5.1,
    "total_script_seconds": 9.5
  }
}
```

---

# 執行方式

```bash
python 02_predict_call_with_llm_two_stage.py \
  --input new_calls/new_call.txt
```

輸出 JSON：

```bash
python 02_predict_call_with_llm_two_stage.py \
  --input new_calls/new_call.txt \
  --output processed/predictions/result.json
```

---

# 03_build_category_faiss_index.py

# 功能

將：

```text
category_knowledge_base.json
```

轉成 embedding。

並建立：

```text
FAISS category index
```

---

# 為什麼 embedding 的是 knowledge base？

不是 embedding 原始通話。

而是 embedding：

```text
分類規則
```

也就是：

```text
category-level semantic knowledge
```

例如：

```text
定義
判斷規則
客戶意圖
negative rules
evidence
```

這比：

```text
直接 embedding 單一通話
```

更穩定。

---

# category text 建構方式

每個 category 會被整理成：

```text
會辦單位：...
會辦類別：...
定義：...
主要客戶意圖：...
判斷規則：...
不應歸類情況：...
通話證據：...
```

之後再 embedding。

---

# Embedding Model

目前推薦：

```python
MODEL_NAME = "intfloat/multilingual-e5-base"
```

---

# 為什麼使用 E5？

因為 E5 是：

```text
retrieval-oriented embedding model
```

適合：

```text
semantic retrieval
RAG
category routing
```

而不是單純 sentence similarity。

---

# small / base / large 差異

## multilingual-e5-small

```text
384 維
速度快
適合 prototype
```

---

## multilingual-e5-base

```text
768 維
semantic quality 較好
目前最推薦
```

---

## multilingual-e5-large

```text
1024 維
成本高
目前資料量下可能 overkill
```

---

# FAISS 設計

目前使用：

```python
faiss.IndexFlatIP
```

並對向量做 normalize：

```python
inner product ≈ cosine similarity
```

---

# 注意事項

只要更換 embedding model：

```text
small → base
```

就必須重新建立：

```text
1. category_rules.index
2. category_rule_vectors.npy
3. metadata
```

否則會出現：

```text
AssertionError: assert d == self.d
```

因為：

```text
embedding 維度不同
```

---

# 執行方式

```bash
python 03_build_category_faiss_index.py
```

---

# 輸出

```text
processed/category_faiss/
├── category_rules.index
├── category_rules_metadata.json
└── category_rule_vectors.npy
```

---

# 04_test_category_retrieval.py

# 功能

測試：

```text
輸入一通新電話
↓
能否從 category knowledge base 找回相關分類規則
```

---

# Retrieval 流程

```text
new_call.txt
        ↓
E5 embedding
        ↓
FAISS search
        ↓
Top-K category rules
```

---

# 為什麼 retrieval 很重要？

目前系統：

```text
LLM + knowledge base
```

如果 knowledge base 太大：

```text
prompt 會爆炸
```

因此：

```text
Embedding retrieval
```

的目的不是直接分類。

而是：

```text
幫 LLM 縮小候選範圍
```

也就是：

```text
coarse retrieval
↓
fine reasoning
```

---

# 輸出結果

```json
{
  "rank": 1,
  "score": 0.82,
  "label_name": "",
  "unit": "",
  "category": ""
}
```

---

# timing 統計

會記錄：

```json
{
  "load_model_seconds": 1.2,
  "embedding_seconds": 0.08,
  "faiss_search_seconds": 0.0003,
  "total_script_seconds": 1.3
}
```

---

# 執行方式

```bash
python 04_test_category_retrieval.py \
  --input new_calls/new_call.txt \
  --top-k 5
```

---

# utils/timing_utils.py

# 功能

統一管理：

```text
執行時間紀錄
```

---

# 使用方式

```python
from utils.timing_utils import TimingRecorder

timer = TimingRecorder()

with timer.measure("stage1_llm_seconds"):
    result = call_llm()
```

---

# 整體系統架構

# LLM 分類流程

```text
new_call.txt
        ↓
讀取 category_knowledge_base.json
        ↓
Stage 1：判斷 unit
        ↓
Stage 2：判斷 category
        ↓
final_prediction
```

---

# Retrieval 流程

```text
category_knowledge_base.json
        ↓
embedding
        ↓
FAISS

================================

new_call.txt
        ↓
embedding
        ↓
FAISS search
        ↓
Top-K category candidates
```

---

# 下一步建議

目前：

```text
02_ 與 04_ 還是分開的
```

下一步可以整合成：

```text
05_predict_call_with_category_rag.py
```

流程：

```text
new_call.txt
→ embedding
→ FAISS retrieve Top-K categories
→ LLM final classification
→ output
```

這會真正變成：

```text
RAG-style category classification system
```
