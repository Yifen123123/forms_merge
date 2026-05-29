# test_two_stage_category_rag.py

## 功能說明

`test_two_stage_category_rag.py` 用於評估會辦分類系統在測試資料集上的表現。

此程式採用 **Two-Stage RAG Classification（兩階段檢索增強分類）** 架構，透過向量檢索（FAISS）與大型語言模型（LLM）結合，完成會辦單位與會辦類別的預測。

---

## 輸入資料

### 測試資料

```text
processed/test_dataset.json
test_data/
```

每筆資料包含：

```json
{
  "call_id": "001",
  "doc_type": "行政會辦單",
  "unit": "契約服務部",
  "category": "保單補發",
  "label_name": "契約服務部_保單補發"
}
```

### 分類知識庫

```text
processed/category_knowledge_base.json
```

由 `build_category_knowledge_base.py` 產生。

### 分類向量資料庫

```text
processed/category_faiss/
├── category_rules.index
├── category_rules_metadata.json
└── category_rule_vectors.npy
```

由 `build_category_faiss_index.py` 產生。

---

## 系統流程

### Stage 1：會辦單位分類（Unit Classification）

首先將所有類別規則依照：

```text
doc_type + unit
```

進行聚合，建立 Unit-Level 知識文件。

例如：

```text
行政會辦單 / 契約服務部
    ├─ 保單補發
    ├─ 契約變更
    └─ 地址修改
```

接著：

1. 將測試通話轉換為向量
2. 使用 FAISS 搜尋最相近的 Unit
3. 取得 Top-K Unit Candidates
4. 交由 LLM 從候選單位中選出最適合的單位

輸出：

```json
{
  "doc_type": "...",
  "unit": "...",
  "confidence": 0.85
}
```

---

### Stage 2：會辦類別分類（Category Classification）

在取得預測單位後：

```text
doc_type = 行政會辦單
unit = 契約服務部
```

系統只保留該單位底下的類別：

```text
契約服務部_保單補發
契約服務部_契約變更
契約服務部_地址修改
...
```

再執行：

1. FAISS Category Retrieval
2. 取得 Top-K Category Candidates
3. 交由 LLM 選出最終類別

輸出：

```json
{
  "label_name": "契約服務部_保單補發",
  "confidence": 0.91
}
```

---

## 為何採用 Two-Stage 分類

相較於直接在所有類別中搜尋：

```text
通話
↓
所有類別
↓
分類結果
```

Two-Stage 架構：

```text
通話
↓
Unit
↓
Category
↓
分類結果
```

具有以下優點：

* 降低 Prompt 長度
* 降低類別數量
* 降低類別混淆
* 提高 Retrieval 命中率
* 提高 LLM 分類穩定度
* 更容易分析錯誤來源

---

## 評估指標

程式會輸出三種 Accuracy。

### Unit Accuracy

第一階段單位分類正確率：

```text
(doc_type, unit)
```

是否預測正確。

---

### Category Accuracy

第二階段類別分類正確率：

```text
category
```

是否預測正確。

---

### Final Accuracy

完整分類正確率：

```text
label_name
```

是否完全正確。

例如：

```text
True:
契約服務部_保單補發

Pred:
契約服務部_保單補發
```

則：

```text
Final Accuracy = Correct
```

---

## 輸出檔案

### 詳細結果

```text
processed/two_stage_rag_test_results.json
```

紀錄每筆測試資料：

* 真實標籤
* 預測標籤
* Unit 預測結果
* Category 預測結果
* Retrieval 候選類別
* Confidence
* Reason

---

### 統計結果

```text
processed/two_stage_rag_test_summary.json
```

包含：

```json
{
  "total": 100,
  "unit_accuracy": 0.87,
  "category_accuracy": 0.79,
  "final_accuracy": 0.76
}
```

---

## 執行方式

```bash
python test_two_stage_category_rag.py
```

---

## 核心概念

此程式並非直接讓 LLM 自由分類，而是：

```text
Embedding
↓
FAISS Retrieval
↓
Top-K Candidates
↓
LLM Re-ranking
↓
Final Prediction
```

透過 Retrieval + LLM 的方式降低幻覺（Hallucination）與分類錯誤風險，並提升分類結果的可解釋性。
