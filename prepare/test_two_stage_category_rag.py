from pathlib import Path
import json
import re
import time
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report


# =========================
# Config
# =========================

OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL_NAME = "gpt-oss:20b"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

TEST_DATASET_PATH = Path("processed/test_dataset.json")
TEST_DATA_DIR = Path("test_data")

CATEGORY_INDEX_PATH = Path("processed/category_faiss/category_rules.index")
CATEGORY_METADATA_PATH = Path("processed/category_faiss/category_rules_metadata.json")

OUTPUT_RESULT_PATH = Path("processed/two_stage_rag_test_results.json")
OUTPUT_SUMMARY_PATH = Path("processed/two_stage_rag_test_summary.json")

TOP_K_UNIT = 5
TOP_K_CATEGORY = 5

MAX_CALL_CHARS = 5000
TIMEOUT_SECONDS = 300
SLEEP_SECONDS = 1
CONFIDENCE_THRESHOLD = 0.65


# =========================
# I/O
# =========================

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_test_call(item: dict) -> str:
    path = (
        TEST_DATA_DIR
        / item["doc_type"]
        / item["unit"]
        / item["category"]
        / f"{item['call_id']}.txt"
    )

    if not path.exists():
        raise FileNotFoundError(f"找不到測試通話檔案：{path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"測試通話檔案是空的：{path}")

    return text[:MAX_CALL_CHARS]


# =========================
# JSON / LLM Utils
# =========================

def extract_json(text: str) -> dict:
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM 回傳內容找不到 JSON：\n{text}")

    return json.loads(text[start:end + 1])


def call_ollama(prompt: str) -> str:
    url = f"{OLLAMA_HOST}/api/generate"

    payload = {
        "model": LLM_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "top_p": 0.8,
            "num_ctx": 8192,
        },
    }

    response = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()

    data = response.json()

    if "response" not in data:
        raise RuntimeError(f"Ollama 回傳格式異常：{data}")

    return data["response"].strip()


def call_llm_json(prompt: str, retry_hint: str) -> dict:
    raw = call_ollama(prompt)

    try:
        return extract_json(raw)
    except Exception:
        retry_prompt = f"""
你剛剛的輸出不是合法 JSON。

請重新輸出。
嚴格限制：
1. 只能輸出一個 JSON object。
2. 第一個字元必須是 {{。
3. 最後一個字元必須是 }}。
4. 不准輸出 markdown。
5. 不准輸出 JSON 以外的任何文字。

請輸出格式範例：
{retry_hint}

原始任務：
{prompt}
""".strip()

        raw2 = call_ollama(retry_prompt)
        return extract_json(raw2)


# =========================
# Embedding / Retrieval
# =========================

def encode_query(model: SentenceTransformer, text: str) -> np.ndarray:
    return model.encode(
        [f"query: {text}"],
        normalize_embeddings=True,
    ).astype("float32")


def build_unit_documents(category_metadata: list[dict]) -> list[dict]:
    """
    不另外建立 unit FAISS。
    直接從 category metadata 聚合出 unit-level 文件。
    """

    unit_map = {}

    for item in category_metadata:
        doc_type = item.get("doc_type", "")
        unit = item.get("unit", "")
        unit_key = f"{doc_type}_{unit}"

        if unit_key not in unit_map:
            unit_map[unit_key] = {
                "unit_index": len(unit_map),
                "unit_key": unit_key,
                "doc_type": doc_type,
                "unit": unit,
                "categories": [],
                "unit_text_parts": [],
            }

        unit_map[unit_key]["categories"].append({
            "label_name": item.get("label_name", ""),
            "category": item.get("category", ""),
            "definition": item.get("definition", ""),
            "data_sufficiency": item.get("data_sufficiency", ""),
        })

        unit_map[unit_key]["unit_text_parts"].append(
            item.get("category_text", "")
        )

    unit_docs = []

    for unit_key, data in unit_map.items():
        categories_text = "\n".join(data["unit_text_parts"])

        unit_text = f"""
doc_type：{data["doc_type"]}
會辦單位：{data["unit"]}
此單位底下的會辦類別與規則：
{categories_text}
""".strip()

        unit_docs.append({
            "unit_index": data["unit_index"],
            "unit_key": unit_key,
            "doc_type": data["doc_type"],
            "unit": data["unit"],
            "categories": data["categories"],
            "unit_text": unit_text,
        })

    return unit_docs


def build_unit_faiss(
    unit_docs: list[dict],
    model: SentenceTransformer,
):
    texts = [
        f"passage: {item['unit_text']}"
        for item in unit_docs
    ]

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    return index, vectors


def retrieve_units(
    call_text: str,
    model: SentenceTransformer,
    unit_index,
    unit_docs: list[dict],
    top_k: int,
) -> list[dict]:
    query_vector = encode_query(model, call_text)
    scores, indices = unit_index.search(query_vector, top_k)

    candidates = []

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx == -1:
            continue

        item = unit_docs[idx]

        candidates.append({
            "candidate_index": len(candidates),
            "rank": rank,
            "retrieval_score": float(score),
            "doc_type": item["doc_type"],
            "unit": item["unit"],
            "unit_key": item["unit_key"],
            "categories": item["categories"],
            "unit_text": item["unit_text"],
        })

    return candidates


def retrieve_categories_within_unit(
    call_text: str,
    pred_doc_type: str,
    pred_unit: str,
    model: SentenceTransformer,
    category_index,
    category_metadata: list[dict],
    top_k: int,
) -> list[dict]:

    query_vector = encode_query(model, call_text)

    # 先全域取多一點，再過濾 unit
    search_k = min(len(category_metadata), max(top_k * 10, top_k))
    scores, indices = category_index.search(query_vector, search_k)

    candidates = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        item = category_metadata[idx]

        if item.get("doc_type", "") != pred_doc_type:
            continue

        if item.get("unit", "") != pred_unit:
            continue

        candidates.append({
            "candidate_index": len(candidates),
            "rank": len(candidates) + 1,
            "retrieval_score": float(score),
            "category_index": item.get("category_index", idx),
            "label_name": item.get("label_name", ""),
            "doc_type": item.get("doc_type", ""),
            "unit": item.get("unit", ""),
            "category": item.get("category", ""),
            "definition": item.get("definition", ""),
            "num_examples": item.get("num_examples", 0),
            "data_sufficiency": item.get("data_sufficiency", ""),
            "category_text": item.get("category_text", ""),
        })

        if len(candidates) >= top_k:
            break

    # 如果全域搜尋後沒有找到該 unit 的類別，就直接從 metadata 裡補候選
    if not candidates:
        fallback_items = [
            item for item in category_metadata
            if item.get("doc_type", "") == pred_doc_type
            and item.get("unit", "") == pred_unit
        ]

        for item in fallback_items[:top_k]:
            candidates.append({
                "candidate_index": len(candidates),
                "rank": len(candidates) + 1,
                "retrieval_score": 0.0,
                "category_index": item.get("category_index", -1),
                "label_name": item.get("label_name", ""),
                "doc_type": item.get("doc_type", ""),
                "unit": item.get("unit", ""),
                "category": item.get("category", ""),
                "definition": item.get("definition", ""),
                "num_examples": item.get("num_examples", 0),
                "data_sufficiency": item.get("data_sufficiency", ""),
                "category_text": item.get("category_text", ""),
            })

    return candidates


# =========================
# Prompts
# =========================

def build_unit_prompt(call_text: str, unit_candidates: list[dict]) -> str:
    simplified = []

    for item in unit_candidates:
        simplified.append({
            "candidate_index": item["candidate_index"],
            "retrieval_rank": item["rank"],
            "retrieval_score": round(item["retrieval_score"], 6),
            "doc_type": item["doc_type"],
            "unit": item["unit"],
            "categories_under_unit": [
                c["category"] for c in item["categories"]
            ],
            "unit_text": item["unit_text"],
        })

    max_index = len(simplified) - 1

    return f"""
你是一個保險客服通話的兩段式會辦分類系統。

Stage 1 任務：
請先判斷這通電話最可能屬於哪一個會辦單位。

重要限制：
1. 只能從 unit_candidates 選擇一個 candidate_index。
2. candidate_index 必須是 0 到 {max_index} 的整數。
3. 不可以自行新增單位。
4. 請根據通話中的主要需求、處理方向、明確證據判斷。
5. 只能輸出 JSON，不要輸出 markdown 或說明文字。

unit_candidates:
{json.dumps(simplified, ensure_ascii=False, indent=2)}

新通話內容：
\"\"\"
{call_text}
\"\"\"

請只輸出以下 JSON：

{{
  "pred_candidate_index": 0,
  "confidence": 0.0,
  "reason": "",
  "evidence_from_call": [],
  "need_human_review": false
}}
""".strip()


def build_category_prompt(call_text: str, category_candidates: list[dict]) -> str:
    simplified = []

    for item in category_candidates:
        simplified.append({
            "candidate_index": item["candidate_index"],
            "retrieval_rank": item["rank"],
            "retrieval_score": round(item["retrieval_score"], 6),
            "label_name": item["label_name"],
            "doc_type": item["doc_type"],
            "unit": item["unit"],
            "category": item["category"],
            "definition": item.get("definition", ""),
            "num_examples": item.get("num_examples", 0),
            "data_sufficiency": item.get("data_sufficiency", ""),
            "category_text": item.get("category_text", ""),
        })

    max_index = len(simplified) - 1

    return f"""
你是一個保險客服通話的兩段式會辦分類系統。

Stage 2 任務：
目前已經先判斷出會辦單位。
請你只在 category_candidates 中選出最適合的會辦類別。

重要限制：
1. 只能從 category_candidates 選擇一個 candidate_index。
2. candidate_index 必須是 0 到 {max_index} 的整數。
3. 不可以自行新增、改寫、翻譯 label_name。
4. retrieval_score 只能參考，不可以只因為分數最高就選它。
5. 若信心不足，仍需選出最可能的一類，但 confidence 要降低。
6. 只能輸出 JSON，不要輸出 markdown 或說明文字。

category_candidates:
{json.dumps(simplified, ensure_ascii=False, indent=2)}

新通話內容：
\"\"\"
{call_text}
\"\"\"

請只輸出以下 JSON：

{{
  "pred_candidate_index": 0,
  "confidence": 0.0,
  "reason": "",
  "evidence_from_call": [],
  "possible_alternatives": [],
  "need_human_review": false
}}
""".strip()


# =========================
# Validation
# =========================

def validate_unit_prediction(result: dict, candidates: list[dict]) -> dict:
    candidate_map = {
        item["candidate_index"]: item
        for item in candidates
    }

    pred_index = int(result.get("pred_candidate_index"))

    if pred_index not in candidate_map:
        raise ValueError(f"unit pred_candidate_index 不在候選範圍內：{pred_index}")

    matched = candidate_map[pred_index]

    confidence = float(result.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    evidence = result.get("evidence_from_call", [])
    if not isinstance(evidence, list):
        evidence = []

    return {
        "pred_candidate_index": pred_index,
        "retrieval_rank": matched["rank"],
        "retrieval_score": matched["retrieval_score"],
        "doc_type": matched["doc_type"],
        "unit": matched["unit"],
        "confidence": confidence,
        "reason": result.get("reason", ""),
        "evidence_from_call": evidence,
        "need_human_review": bool(result.get("need_human_review", False)) or confidence < CONFIDENCE_THRESHOLD,
    }


def validate_category_prediction(result: dict, candidates: list[dict]) -> dict:
    candidate_map = {
        item["candidate_index"]: item
        for item in candidates
    }

    pred_index = int(result.get("pred_candidate_index"))

    if pred_index not in candidate_map:
        raise ValueError(f"category pred_candidate_index 不在候選範圍內：{pred_index}")

    matched = candidate_map[pred_index]

    confidence = float(result.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    evidence = result.get("evidence_from_call", [])
    if not isinstance(evidence, list):
        evidence = []

    return {
        "pred_candidate_index": pred_index,
        "retrieval_rank": matched["rank"],
        "retrieval_score": matched["retrieval_score"],
        "label_name": matched["label_name"],
        "doc_type": matched["doc_type"],
        "unit": matched["unit"],
        "category": matched["category"],
        "confidence": confidence,
        "reason": result.get("reason", ""),
        "evidence_from_call": evidence,
        "possible_alternatives": result.get("possible_alternatives", []),
        "need_human_review": bool(result.get("need_human_review", False)) or confidence < CONFIDENCE_THRESHOLD,
    }


# =========================
# Evaluation
# =========================

def evaluate_results(results: list[dict]) -> dict:
    y_true_final = [r["true_label_name"] for r in results]
    y_pred_final = [r["final_prediction"]["label_name"] for r in results]

    final_accuracy = accuracy_score(y_true_final, y_pred_final)

    unit_correct_count = sum(r["unit_correct"] for r in results)
    category_correct_count = sum(r["category_correct"] for r in results)
    final_correct_count = sum(r["final_correct"] for r in results)

    return {
        "total": len(results),
        "unit_correct": unit_correct_count,
        "category_correct": category_correct_count,
        "final_correct": final_correct_count,
        "unit_accuracy": unit_correct_count / len(results) if results else 0,
        "category_accuracy": category_correct_count / len(results) if results else 0,
        "final_accuracy": final_accuracy,
        "classification_report": classification_report(
            y_true_final,
            y_pred_final,
            output_dict=True,
            zero_division=0,
        ),
    }


# =========================
# Main
# =========================

def main():
    test_dataset = load_json(TEST_DATASET_PATH)

    if not isinstance(test_dataset, list):
        raise ValueError("test_dataset.json 應該是一個 list")

    print("載入 embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("載入 category FAISS...")
    category_index = faiss.read_index(str(CATEGORY_INDEX_PATH))
    category_metadata = load_json(CATEGORY_METADATA_PATH)

    print("建立 unit-level FAISS...")
    unit_docs = build_unit_documents(category_metadata)
    unit_index, _ = build_unit_faiss(unit_docs, embedding_model)

    results = []
    errors = []

    print(f"測試資料數：{len(test_dataset)}")
    print(f"Unit Top-K：{TOP_K_UNIT}")
    print(f"Category Top-K：{TOP_K_CATEGORY}")

    for idx, item in enumerate(test_dataset, start=1):
        print("=" * 80)
        print(f"[{idx}/{len(test_dataset)}] call_id={item['call_id']}")
        print(f"True：{item['label_name']}")

        try:
            call_text = read_test_call(item)

            # ---------- Stage 1：Unit ----------
            unit_candidates = retrieve_units(
                call_text=call_text,
                model=embedding_model,
                unit_index=unit_index,
                unit_docs=unit_docs,
                top_k=TOP_K_UNIT,
            )

            if not unit_candidates:
                raise ValueError("Stage 1 沒有取回任何 unit candidates")

            unit_prompt = build_unit_prompt(call_text, unit_candidates)
            unit_raw = call_llm_json(
                unit_prompt,
                retry_hint='{"pred_candidate_index": 0, "confidence": 0.0, "reason": "", "evidence_from_call": [], "need_human_review": true}',
            )
            unit_pred = validate_unit_prediction(unit_raw, unit_candidates)

            # ---------- Stage 2：Category ----------
            category_candidates = retrieve_categories_within_unit(
                call_text=call_text,
                pred_doc_type=unit_pred["doc_type"],
                pred_unit=unit_pred["unit"],
                model=embedding_model,
                category_index=category_index,
                category_metadata=category_metadata,
                top_k=TOP_K_CATEGORY,
            )

            if not category_candidates:
                raise ValueError(
                    f"Stage 2 沒有取回任何 category candidates："
                    f"{unit_pred['doc_type']} / {unit_pred['unit']}"
                )

            category_prompt = build_category_prompt(call_text, category_candidates)
            category_raw = call_llm_json(
                category_prompt,
                retry_hint='{"pred_candidate_index": 0, "confidence": 0.0, "reason": "", "evidence_from_call": [], "possible_alternatives": [], "need_human_review": true}',
            )
            final_pred = validate_category_prediction(category_raw, category_candidates)

            unit_correct = (
                item["doc_type"] == unit_pred["doc_type"]
                and item["unit"] == unit_pred["unit"]
            )

            category_correct = (
                item["category"] == final_pred["category"]
            )

            final_correct = (
                item["label_name"] == final_pred["label_name"]
            )

            result = {
                "call_id": item["call_id"],
                "true_label_name": item["label_name"],
                "true_doc_type": item["doc_type"],
                "true_unit": item["unit"],
                "true_category": item["category"],

                "unit_prediction": unit_pred,
                "final_prediction": final_pred,

                "unit_correct": unit_correct,
                "category_correct": category_correct,
                "final_correct": final_correct,

                "unit_candidates": [
                    {
                        "candidate_index": c["candidate_index"],
                        "rank": c["rank"],
                        "retrieval_score": c["retrieval_score"],
                        "doc_type": c["doc_type"],
                        "unit": c["unit"],
                    }
                    for c in unit_candidates
                ],

                "category_candidates": [
                    {
                        "candidate_index": c["candidate_index"],
                        "rank": c["rank"],
                        "retrieval_score": c["retrieval_score"],
                        "label_name": c["label_name"],
                        "doc_type": c["doc_type"],
                        "unit": c["unit"],
                        "category": c["category"],
                    }
                    for c in category_candidates
                ],
            }

            results.append(result)
            save_json(results, OUTPUT_RESULT_PATH)

            print(f"Pred unit：{unit_pred['doc_type']} / {unit_pred['unit']}")
            print(f"Pred label：{final_pred['label_name']}")
            print(f"Final correct：{final_correct}")

        except Exception as e:
            error_item = {
                "call_id": item.get("call_id", ""),
                "true_label_name": item.get("label_name", ""),
                "error": str(e),
            }

            errors.append(error_item)
            print("失敗")
            print(e)

        time.sleep(SLEEP_SECONDS)

    summary = evaluate_results(results)

    if errors:
        summary["errors"] = errors

    save_json(results, OUTPUT_RESULT_PATH)
    save_json(summary, OUTPUT_SUMMARY_PATH)

    print("=" * 80)
    print("兩段式 RAG 測試完成")
    print(f"成功測試數：{len(results)}")
    print(f"錯誤數：{len(errors)}")
    print(f"Unit Accuracy：{summary['unit_accuracy']:.4f}")
    print(f"Category Accuracy：{summary['category_accuracy']:.4f}")
    print(f"Final Accuracy：{summary['final_accuracy']:.4f}")
    print(f"結果輸出：{OUTPUT_RESULT_PATH}")
    print(f"摘要輸出：{OUTPUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
