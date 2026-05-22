from pathlib import Path
import argparse
import json
import re
import requests


# =========================
# 基本設定
# =========================

OLLAMA_HOST = ""
MODEL_NAME = "gpt-oss:20b"

KB_PATH = Path("processed/category_knowledge_base.json")

MAX_CALL_CHARS = 6000
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


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"找不到通話檔案：{path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"通話檔案是空的：{path}")

    return text[:MAX_CALL_CHARS]


# =========================
# JSON parsing
# =========================

def clean_llm_json_text(text: str) -> str:
    text = text.strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError(f"LLM 回傳內容找不到 JSON：\n{text}")

    return match.group(0)


def extract_json(text: str) -> dict:
    json_text = clean_llm_json_text(text)

    try:
        return json.loads(json_text)

    except json.JSONDecodeError:
        print("\n========== RAW LLM OUTPUT ==========\n")
        print(text)
        print("\n========== JSON TEXT ==========\n")
        print(json_text)
        raise


# =========================
# Ollama
# =========================

def call_ollama(prompt: str) -> str:
    url = f"{OLLAMA_HOST}/api/generate"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 8192
        }
    }

    response = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()

    data = response.json()

    if "response" not in data:
        raise RuntimeError(f"Ollama 回傳格式異常：{data}")

    return data["response"]


# =========================
# Knowledge base
# =========================

def simplify_kb_for_prompt(kb: list[dict]) -> list[dict]:
    simplified = []

    for idx, item in enumerate(kb):
        simplified.append({
            "index": idx,
            "label_name": item.get("label_name", ""),
            "doc_type": item.get("doc_type", ""),
            "unit": item.get("unit", ""),
            "category": item.get("category", ""),
            "definition": item.get("definition", ""),
            "main_customer_intents": item.get("main_customer_intents", []),
            "keywords": item.get("keywords", []),
            "decision_rules": item.get("decision_rules", []),
            "negative_rules": item.get("negative_rules", []),
            "required_evidence_from_call": item.get("required_evidence_from_call", []),
            "num_examples": item.get("num_examples", 0),
            "data_sufficiency": item.get("data_sufficiency", "unknown")
        })

    return simplified


def build_label_options(kb: list[dict]) -> list[dict]:
    options = []

    for idx, item in enumerate(kb):
        options.append({
            "index": idx,
            "label_name": item.get("label_name", ""),
            "doc_type": item.get("doc_type", ""),
            "unit": item.get("unit", ""),
            "category": item.get("category", "")
        })

    return options


# =========================
# Prompt
# =========================

def build_prompt(call_text: str, kb: list[dict]) -> str:
    kb_for_prompt = simplify_kb_for_prompt(kb)
    label_options = build_label_options(kb)

    max_index = len(label_options) - 1

    prompt = f"""
你是一個保險客服通話的會辦分類系統。

你的任務：
根據「新通話內容」與「分類知識庫」，判斷這通電話最應該被分到哪一個會辦分類。

非常重要的限制：
1. 你只能輸出 provided_label_options 中存在的 pred_label_index。
2. pred_label_index 必須是整數，範圍只能是 0 到 {max_index}。
3. 禁止輸出 label_name 作為分類結果。
4. 禁止自行新增、改寫、翻譯、縮寫任何分類名稱。
5. 最後的 label_name 會由程式根據 pred_label_index 自動反查，因此你不需要輸出 pred_label_name。
6. 如果信心不足，也必須選出最可能的一個 pred_label_index，但 confidence 要降低。
7. 如果某類 data_sufficiency = low，請不要過度自信。
8. 請根據通話中的具體語句、主要需求、處理方向判斷，不要只靠關鍵字。
9. 請輸出合法 JSON，不要輸出 markdown，不要加任何解釋文字。

provided_label_options:
{json.dumps(label_options, ensure_ascii=False, indent=2)}

分類知識庫：
{json.dumps(kb_for_prompt, ensure_ascii=False, indent=2)}

新通話內容：
\"\"\"
{call_text}
\"\"\"

請只輸出以下 JSON 格式，欄位名稱不可更改：

{{
  "pred_label_index": 0,
  "confidence": 0.0,
  "reason": "",
  "evidence_from_call": [],
  "possible_alternatives": [
    {{
      "label_index": 0,
      "reason": ""
    }}
  ],
  "need_human_review": false
}}
"""
    return prompt.strip()


# =========================
# Validation
# =========================

def validate_prediction(result: dict, kb: list[dict]) -> dict:
    label_map = {
        idx: item
        for idx, item in enumerate(kb)
    }

    pred_index = result.get("pred_label_index")

    try:
        pred_index = int(pred_index)
    except Exception:
        result["pred_label_index"] = None
        result["pred_label_name"] = ""
        result["pred_doc_type"] = ""
        result["pred_unit"] = ""
        result["pred_category"] = ""
        result["confidence"] = 0.0
        result["need_human_review"] = True
        result["validation_warning"] = "LLM 沒有輸出合法的 pred_label_index。"
        return result

    if pred_index not in label_map:
        result["pred_label_index"] = pred_index
        result["pred_label_name"] = ""
        result["pred_doc_type"] = ""
        result["pred_unit"] = ""
        result["pred_category"] = ""
        result["confidence"] = 0.0
        result["need_human_review"] = True
        result["validation_warning"] = "LLM 輸出的 pred_label_index 不在候選範圍內。"
        return result

    matched = label_map[pred_index]

    result["pred_label_index"] = pred_index
    result["pred_label_name"] = matched.get("label_name", "")
    result["pred_doc_type"] = matched.get("doc_type", "")
    result["pred_unit"] = matched.get("unit", "")
    result["pred_category"] = matched.get("category", "")

    try:
        confidence = float(result.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))
    result["confidence"] = confidence

    if "need_human_review" not in result:
        result["need_human_review"] = False

    if confidence < CONFIDENCE_THRESHOLD:
        result["need_human_review"] = True

    # 整理 possible_alternatives
    alternatives = result.get("possible_alternatives", [])

    cleaned_alternatives = []

    if isinstance(alternatives, list):
        for alt in alternatives:
            if not isinstance(alt, dict):
                continue

            alt_index = alt.get("label_index")

            try:
                alt_index = int(alt_index)
            except Exception:
                continue

            if alt_index not in label_map:
                continue

            alt_item = label_map[alt_index]

            cleaned_alternatives.append({
                "label_index": alt_index,
                "label_name": alt_item.get("label_name", ""),
                "doc_type": alt_item.get("doc_type", ""),
                "unit": alt_item.get("unit", ""),
                "category": alt_item.get("category", ""),
                "reason": alt.get("reason", "")
            })

    result["possible_alternatives"] = cleaned_alternatives

    return result


# =========================
# CLI
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Use local Ollama LLM to classify a new call into meeting-form category."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="新通話 txt 檔案路徑，例如 new_calls/new_call.txt"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="可選：輸出 JSON 路徑，例如 processed/predictions/result.json"
    )

    return parser.parse_args()


# =========================
# Main
# =========================

def main():
    args = parse_args()

    input_path = Path(args.input)

    kb = load_json(KB_PATH)

    if not isinstance(kb, list):
        raise ValueError("category_knowledge_base.json 應該是一個 list。")

    call_text = read_text(input_path)
    prompt = build_prompt(call_text, kb)

    print(f"分類知識庫類別數：{len(kb)}")
    print(f"新通話長度：{len(call_text)} 字")
    print(f"Prompt 長度：{len(prompt)} 字")
    print("開始呼叫 Ollama...")

    raw_output = call_ollama(prompt)

    print("Ollama 回傳完成，開始解析 JSON...")

    result = extract_json(raw_output)
    result = validate_prediction(result, kb)

    result["input_file"] = str(input_path)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n結果已儲存至：{output_path}")


if __name__ == "__main__":
    main()
