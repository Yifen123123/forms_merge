from pathlib import Path
import argparse
import json
import re
import requests


OLLAMA_HOST = ""
MODEL_NAME = "gpt-oss:20b"

KB_PATH = Path("processed/category_knowledge_base.json")

MAX_CALL_CHARS = 6000
TIMEOUT_SECONDS = 180


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
    text = text.strip()

    # 移除 markdown code block
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # 抓最外層 JSON
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError(f"找不到 JSON：\n{text}")

    json_text = match.group(0)

    # 修正常見錯誤
    json_text = json_text.replace("\n", " ")

    # 單引號 → 雙引號
    json_text = re.sub(r"'", '"', json_text)

    # True False None 修正
    json_text = json_text.replace("True", "true")
    json_text = json_text.replace("False", "false")
    json_text = json_text.replace("None", "null")

    try:
        return json.loads(json_text)

    except json.JSONDecodeError as e:
        print("\n========== JSON PARSE FAILED ==========\n")
        print(json_text)
        raise e


def call_ollama(prompt: str) -> str:
    url = f"{OLLAMA_HOST}/api/generate"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
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


def simplify_kb_for_prompt(kb: list[dict]) -> list[dict]:
    """
    壓縮 knowledge base，避免 prompt 過長。
    """
    simplified = []

    for item in kb:
        simplified.append({
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


def build_prompt(call_text: str, kb: list[dict]) -> str:
    kb_for_prompt = simplify_kb_for_prompt(kb)

    label_names = [item["label_name"] for item in kb_for_prompt]

    prompt = f"""
你是一個保險客服通話的會辦分類系統。

你的任務：
根據「新通話內容」與「分類知識庫」，判斷這通電話最應該被分到哪一個會辦分類。

重要限制：
1. 你只能從 provided_label_names 裡面選擇一個 label_name。
2. 不可以創造新的 label_name。
3. 不可以輸出不存在於分類知識庫的分類。
4. 如果信心不足，也必須選出最可能的一類，但 confidence 要降低。
5. 如果某類 data_sufficiency = low，請不要過度自信。
6. 請根據通話中的具體語句或需求判斷，不要只靠關鍵字。
7. 請輸出合法 JSON，不要輸出 markdown，不要加任何解釋文字。

provided_label_names:
{json.dumps(label_names, ensure_ascii=False, indent=2)}

分類知識庫：
{json.dumps(kb_for_prompt, ensure_ascii=False, indent=2)}

新通話內容：
\"\"\"
{call_text}
\"\"\"

請輸出以下 JSON 格式，欄位名稱不可更改：

{{
  "pred_label_name": "",
  "pred_doc_type": "",
  "pred_unit": "",
  "pred_category": "",
  "confidence": 0.0,
  "reason": "",
  "evidence_from_call": [],
  "possible_alternatives": [
    {{
      "label_name": "",
      "reason": ""
    }}
  ],
  "need_human_review": false
}}
"""
    return prompt.strip()


def validate_prediction(result: dict, kb: list[dict]) -> dict:
    label_map = {
        item.get("label_name"): item
        for item in kb
    }

    pred_label = result.get("pred_label_name", "")

    if pred_label not in label_map:
        result["need_human_review"] = True
        result["validation_warning"] = "LLM 輸出了不存在於 category_knowledge_base.json 的 label_name。"
        return result

    matched = label_map[pred_label]

    result["pred_doc_type"] = matched.get("doc_type", "")
    result["pred_unit"] = matched.get("unit", "")
    result["pred_category"] = matched.get("category", "")

    confidence = result.get("confidence", 0.0)

    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))
    result["confidence"] = confidence

    if confidence < 0.65:
        result["need_human_review"] = True

    return result


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


def main():
    args = parse_args()

    input_path = Path(args.input)

    kb = load_json(KB_PATH)
    call_text = read_text(input_path)

    prompt = build_prompt(call_text, kb)

    print(f"分類知識庫類別數：{len(kb)}")
    print(f"新通話長度：{len(call_text)} 字")
    print(f"Prompt 長度：{len(prompt)} 字")
    print("開始呼叫 Ollama...")

    raw_output = call_ollama(prompt)

    print("Ollama 回傳完成，開始解析 JSON...")

    print("\n========== RAW LLM OUTPUT ==========\n")
    print(raw_output)

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
