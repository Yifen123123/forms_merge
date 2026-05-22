from pathlib import Path
import argparse
import json
import time
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from utils.timing_utils import TimingRecorder


# =========================
# Config
# =========================

OLLAMA_HOST = ""
LLM_MODEL_NAME = "gpt-oss:20b"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

KB_PATH = Path("processed/category_knowledge_base.json")
FAISS_INDEX_PATH = Path("processed/category_faiss/category_rules.index")
FAISS_METADATA_PATH = Path("processed/category_faiss/category_rules_metadata.json")

MAX_CALL_CHARS = 6000
TOP_K = 5

TIMEOUT_SECONDS = 180
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


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"找不到通話檔案：{path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"通話檔案是空的：{path}")

    return text[:MAX_CALL_CHARS]


# =========================
# Vector Utils
# =========================

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


# =========================
# Retrieval
# =========================

def retrieve_candidate_categories(
    call_text: str,
    top_k: int,
    model: SentenceTransformer,
    index,
    metadata: list[dict]
) -> list[dict]:
    query_text = f"query: {call_text}"

    query_vector = model.encode(
        [query_text],
        normalize_embeddings=True
    ).astype("float32")

    query_vector[0] = normalize_vector(query_vector[0])

    scores, indices = index.search(query_vector, top_k)

    candidates = []

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx == -1:
            continue

        item = metadata[idx]

        candidates.append({
            "candidate_index": len(candidates),
            "rank": rank,
            "retrieval_score": float(score),
            "category_index": item.get("category_index", idx),
            "label_name": item.get("label_name", ""),
            "doc_type": item.get("doc_type", ""),
            "unit": item.get("unit", ""),
            "category": item.get("category", ""),
            "definition": item.get("definition", ""),
            "num_examples": item.get("num_examples", 0),
            "data_sufficiency": item.get("data_sufficiency", ""),
            "category_text": item.get("category_text", "")
        })

    return candidates


# =========================
# Ollama
# =========================

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
            "num_ctx": 8192
        }
    }

    response = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()

    data = response.json()

    if "response" not in data:
        raise RuntimeError(f"Ollama 回傳格式異常：{data}")

    return data["response"].strip()


def extract_json(text: str) -> dict:
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        print("\n========== RAW LLM OUTPUT ==========\n")
        print(repr(text))
        raise ValueError("LLM 回傳內容找不到 JSON")

    json_text = text[start:end + 1]

    try:
        return json.loads(json_text)

    except json.JSONDecodeError:
        print("\n========== RAW LLM OUTPUT ==========\n")
        print(repr(text))
        print("\n========== JSON TEXT ==========\n")
        print(json_text)
        raise


def call_llm_json(prompt: str, retry_prompt_hint: str = "") -> dict:
    raw_output = call_ollama(prompt)

    try:
        return extract_json(raw_output)

    except Exception:
        print("\n第一次輸出不是合法 JSON，開始 retry...\n")
        print("第一次 raw output:")
        print(repr(raw_output))

        retry_prompt = f"""
你剛剛的輸出不是合法 JSON。

請重新輸出。
嚴格限制：
1. 只能輸出一個 JSON object。
2. 第一個字元必須是 {{。
3. 最後一個字元必須是 }}。
4. 不准輸出任何推理、說明、英文分析、markdown。
5. 不准輸出 JSON 以外的任何文字。

{retry_prompt_hint}

原始任務如下：
{prompt}
""".strip()

        raw_output_2 = call_ollama(retry_prompt)

        print("\n第二次 raw output:")
        print(repr(raw_output_2))

        return extract_json(raw_output_2)


# =========================
# Prompt
# =========================

def simplify_candidates_for_prompt(candidates: list[dict]) -> list[dict]:
    simplified = []

    for item in candidates:
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
            "category_text": item.get("category_text", "")
        })

    return simplified


def build_rag_prompt(call_text: str, candidates: list[dict]) -> str:
    candidate_options = simplify_candidates_for_prompt(candidates)
    max_index = len(candidate_options) - 1

    return f"""
系統規則：
你不是聊天助手。
你不是解釋助手。
你是 JSON API。
你只能回傳一個 JSON object。
禁止輸出分析過程。
禁止輸出 reasoning。
禁止輸出任何 JSON 以外的文字。

任務：
你是一個保險客服通話的會辦分類系統。

目前系統已經先用 Embedding + FAISS 從分類知識庫中找出 Top-K 候選分類。
請你根據「新通話內容」與「候選分類規則」，只在候選分類中選出最適合的一個分類。

重要限制：
1. 你只能從 candidate_categories 中選擇一個 candidate_index。
2. candidate_index 必須是整數，範圍只能是 0 到 {max_index}。
3. 禁止自行新增、改寫、翻譯、縮寫任何 label_name。
4. 最後 label_name 會由程式根據 candidate_index 自動反查，因此你不需要輸出 label_name。
5. 如果信心不足，也必須選出最可能的一個 candidate_index，但 confidence 要降低。
6. 如果某類 data_sufficiency = low，請不要過度自信。
7. retrieval_score 只能作為參考，不可以只因為分數最高就選它。
8. 請根據通話中的主要需求、處理方向、明確證據判斷。
9. 第一個字元必須是 {{。
10. 最後一個字元必須是 }}。
11. 不要輸出 markdown。
12. 不要輸出說明文字。

candidate_categories:
{json.dumps(candidate_options, ensure_ascii=False, indent=2)}

新通話內容：
\"\"\"
{call_text}
\"\"\"

請只輸出以下 JSON 格式：

{{
  "pred_candidate_index": 0,
  "confidence": 0.0,
  "reason": "",
  "evidence_from_call": [],
  "possible_alternatives": [
    {{
      "candidate_index": 0,
      "reason": ""
    }}
  ],
  "need_human_review": false
}}
""".strip()


# =========================
# Validation
# =========================

def validate_rag_prediction(result: dict, candidates: list[dict]) -> dict:
    candidate_map = {
        item["candidate_index"]: item
        for item in candidates
    }

    try:
        pred_index = int(result.get("pred_candidate_index"))
    except Exception:
        raise ValueError("LLM 沒有輸出合法的 pred_candidate_index")

    if pred_index not in candidate_map:
        raise ValueError(f"pred_candidate_index 不在候選範圍內：{pred_index}")

    matched = candidate_map[pred_index]

    try:
        confidence = float(result.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    evidence = result.get("evidence_from_call", [])
    if not isinstance(evidence, list):
        evidence = []

    alternatives = result.get("possible_alternatives", [])
    cleaned_alternatives = []

    if isinstance(alternatives, list):
        for alt in alternatives:
            if not isinstance(alt, dict):
                continue

            try:
                alt_index = int(alt.get("candidate_index"))
            except Exception:
                continue

            if alt_index not in candidate_map:
                continue

            alt_item = candidate_map[alt_index]

            cleaned_alternatives.append({
                "candidate_index": alt_index,
                "rank": alt_item.get("rank"),
                "retrieval_score": alt_item.get("retrieval_score"),
                "label_name": alt_item.get("label_name", ""),
                "doc_type": alt_item.get("doc_type", ""),
                "unit": alt_item.get("unit", ""),
                "category": alt_item.get("category", ""),
                "reason": alt.get("reason", "")
            })

    return {
        "pred_candidate_index": pred_index,
        "retrieval_rank": matched.get("rank"),
        "retrieval_score": matched.get("retrieval_score"),
        "label_name": matched.get("label_name", ""),
        "doc_type": matched.get("doc_type", ""),
        "unit": matched.get("unit", ""),
        "category": matched.get("category", ""),
        "confidence": confidence,
        "reason": result.get("reason", ""),
        "evidence_from_call": evidence,
        "possible_alternatives": cleaned_alternatives,
        "need_human_review": bool(result.get("need_human_review", False)) or confidence < CONFIDENCE_THRESHOLD
    }


# =========================
# CLI
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG-style LLM classifier using category FAISS retrieval."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="新通話 txt 檔案路徑，例如 new_calls/new_call.txt"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="FAISS 取回前幾個候選分類，預設 5"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="輸出 JSON 路徑，例如 processed/predictions/rag_result.json"
    )

    return parser.parse_args()


# =========================
# Main
# =========================

def main():
    timer = TimingRecorder()
    total_start = time.perf_counter()

    args = parse_args()
    input_path = Path(args.input)

    with timer.measure("read_input_seconds"):
        call_text = read_text(input_path)

    with timer.measure("load_kb_seconds"):
        kb = load_json(KB_PATH)

    if not isinstance(kb, list):
        raise ValueError("category_knowledge_base.json 應該是一個 list")

    print("=" * 80)
    print("Stage 0：Embedding + FAISS Retrieval")

    with timer.measure("load_embedding_model_seconds"):
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    with timer.measure("load_faiss_index_seconds"):
        index = faiss.read_index(str(FAISS_INDEX_PATH))

    with timer.measure("load_faiss_metadata_seconds"):
        metadata = load_json(FAISS_METADATA_PATH)

    with timer.measure("category_retrieval_seconds"):
        candidates = retrieve_candidate_categories(
            call_text=call_text,
            top_k=args.top_k,
            model=embedding_model,
            index=index,
            metadata=metadata
        )

    if not candidates:
        raise ValueError("FAISS 沒有取回任何候選分類")

    print(f"Top-K：{args.top_k}")
    print("候選分類：")
    for item in candidates:
        print(
            f"[{item['candidate_index']}] "
            f"rank={item['rank']} "
            f"score={item['retrieval_score']:.6f} "
            f"{item['label_name']}"
        )

    print("=" * 80)
    print("Stage 1：LLM Final Classification on Retrieved Candidates")

    with timer.measure("rag_prompt_build_seconds"):
        prompt = build_rag_prompt(call_text, candidates)

    print(f"RAG prompt 長度：{len(prompt)} 字")
    print("開始呼叫 Ollama...")

    with timer.measure("rag_llm_seconds"):
        llm_result = call_llm_json(
            prompt,
            retry_prompt_hint='請輸出：{"pred_candidate_index": 0, "confidence": 0.0, "reason": "", "evidence_from_call": [], "possible_alternatives": [], "need_human_review": false}'
        )

    with timer.measure("rag_validation_seconds"):
        prediction = validate_rag_prediction(llm_result, candidates)

    timer.add(
        "total_script_seconds",
        time.perf_counter() - total_start
    )

    final_result = {
        "timing": timer.get_timings(),
        "input_file": str(input_path),
        "method": "category_rag_llm_classification",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": LLM_MODEL_NAME,
        "top_k": args.top_k,
        "retrieved_candidates": [
            {
                "candidate_index": item["candidate_index"],
                "rank": item["rank"],
                "retrieval_score": item["retrieval_score"],
                "label_name": item["label_name"],
                "doc_type": item["doc_type"],
                "unit": item["unit"],
                "category": item["category"],
                "num_examples": item.get("num_examples", 0),
                "data_sufficiency": item.get("data_sufficiency", "")
            }
            for item in candidates
        ],
        "final_prediction": prediction
    }

    print("=" * 80)
    print("Final Result")
    print(json.dumps(final_result, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        save_json(final_result, output_path)
        print(f"\n結果已儲存至：{output_path}")


if __name__ == "__main__":
    main()
