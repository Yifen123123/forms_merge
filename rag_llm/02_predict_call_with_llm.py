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
CONFIDENCE_THRESHOLD = 0.65


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


def extract_json(text: str) -> dict:
    text = text.strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        print("\n========== RAW LLM OUTPUT ==========\n")
        print(repr(text))
        raise ValueError("LLM 回傳內容找不到 JSON")

    return json.loads(match.group(0))


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

    if not data["response"].strip():
        raise ValueError("Ollama response 是空字串，通常是 prompt 太長或模型中斷。")

    return data["response"]


# ============================================================
# Stage 1: Unit Classification
# ============================================================

def build_unit_options(kb: list[dict]) -> list[dict]:
    units = sorted(set(item.get("unit", "") for item in kb if item.get("unit", "")))

    return [
        {
            "unit_index": idx,
            "unit": unit
        }
        for idx, unit in enumerate(units)
    ]


def build_unit_prompt(call_text: str, unit_options: list[dict]) -> str:
    max_index = len(unit_options) - 1

    return f"""
你是一個保險客服通話的會辦單位分類系統。

任務：
根據新通話內容，判斷這通電話最應該送到哪一個會辦單位。

重要限制：
1. 你只能從 provided_unit_options 中選擇一個 unit_index。
2. unit_index 必須是整數，範圍只能是 0 到 {max_index}。
3. 禁止自行新增、改寫、翻譯任何單位名稱。
4. 如果信心不足，也必須選出最可能的一個 unit_index，但 confidence 要降低。
5. 請輸出合法 JSON，不要輸出 markdown，不要加解釋文字。

provided_unit_options:
{json.dumps(unit_options, ensure_ascii=False, indent=2)}

新通話內容：
\"\"\"
{call_text}
\"\"\"

請只輸出以下 JSON：

{{
  "pred_unit_index": 0,
  "confidence": 0.0,
  "reason": "",
  "evidence_from_call": [],
  "need_human_review": false
}}
""".strip()


def validate_unit_prediction(result: dict, unit_options: list[dict]) -> dict:
    unit_map = {
        item["unit_index"]: item["unit"]
        for item in unit_options
    }

    try:
        pred_index = int(result.get("pred_unit_index"))
    except Exception:
        raise ValueError("LLM 沒有輸出合法的 pred_unit_index")

    if pred_index not in unit_map:
        raise ValueError(f"pred_unit_index 不在範圍內：{pred_index}")

    confidence = float(result.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    return {
        "pred_unit_index": pred_index,
        "pred_unit": unit_map[pred_index],
        "unit_confidence": confidence,
        "unit_reason": result.get("reason", ""),
        "unit_evidence_from_call": result.get("evidence_from_call", []),
        "unit_need_human_review": bool(result.get("need_human_review", False)) or confidence < CONFIDENCE_THRESHOLD
    }


# ============================================================
# Stage 2: Category Classification within Unit
# ============================================================

def filter_kb_by_unit(kb: list[dict], unit: str) -> list[dict]:
    return [
        item for item in kb
        if item.get("unit", "") == unit
    ]


def simplify_candidate_kb(candidate_kb: list[dict]) -> list[dict]:
    simplified = []

    for idx, item in enumerate(candidate_kb):
        simplified.append({
            "label_index": idx,
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


def build_category_prompt(call_text: str, pred_unit: str, candidate_kb: list[dict]) -> str:
    candidate_rules = simplify_candidate_kb(candidate_kb)
    max_index = len(candidate_rules) - 1

    return f"""
你是一個保險客服通話的會辦類別分類系統。

目前第一階段已判斷此通話的會辦單位為：

{pred_unit}

任務：
請你只在該單位底下的候選會辦類別中，選出最適合的一個分類。

重要限制：
1. 你只能從 candidate_categories 中選擇一個 label_index。
2. label_index 必須是整數，範圍只能是 0 到 {max_index}。
3. 禁止自行新增、改寫、翻譯、縮寫任何 label_name。
4. 最後 label_name 會由程式根據 label_index 自動反查，因此你不需要輸出 label_name。
5. 如果信心不足，也必須選出最可能的一個 label_index，但 confidence 要降低。
6. 如果某類 data_sufficiency = low，請不要過度自信。
7. 請根據通話中的主要需求、處理方向、明確證據判斷，不要只靠關鍵字。
8. 請輸出合法 JSON，不要輸出 markdown，不要加解釋文字。

candidate_categories:
{json.dumps(candidate_rules, ensure_ascii=False, indent=2)}

新通話內容：
\"\"\"
{call_text}
\"\"\"

請只輸出以下 JSON：

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
""".strip()


def validate_category_prediction(result: dict, candidate_kb: list[dict]) -> dict:
    label_map = {
        idx: item
        for idx, item in enumerate(candidate_kb)
    }

    try:
        pred_index = int(result.get("pred_label_index"))
    except Exception:
        raise ValueError("LLM 沒有輸出合法的 pred_label_index")

    if pred_index not in label_map:
        raise ValueError(f"pred_label_index 不在候選類別範圍內：{pred_index}")

    matched = label_map[pred_index]

    confidence = float(result.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    alternatives = result.get("possible_alternatives", [])
    cleaned_alternatives = []

    if isinstance(alternatives, list):
        for alt in alternatives:
            if not isinstance(alt, dict):
                continue

            try:
                alt_index = int(alt.get("label_index"))
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

    return {
        "pred_label_index": pred_index,
        "pred_label_name": matched.get("label_name", ""),
        "pred_doc_type": matched.get("doc_type", ""),
        "pred_unit": matched.get("unit", ""),
        "pred_category": matched.get("category", ""),
        "category_confidence": confidence,
        "category_reason": result.get("reason", ""),
        "category_evidence_from_call": result.get("evidence_from_call", []),
        "possible_alternatives": cleaned_alternatives,
        "category_need_human_review": bool(result.get("need_human_review", False)) or confidence < CONFIDENCE_THRESHOLD
    }


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage LLM classifier for meeting-form category."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="新通話 txt 檔案路徑，例如 new_calls/new_call.txt"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="輸出 JSON 路徑，例如 processed/predictions/result.json"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)

    kb = load_json(KB_PATH)

    if not isinstance(kb, list):
        raise ValueError("category_knowledge_base.json 應該是一個 list")

    call_text = read_text(input_path)

    print("=" * 80)
    print("Stage 1：判斷會辦單位")

    unit_options = build_unit_options(kb)
    unit_prompt = build_unit_prompt(call_text, unit_options)

    print(f"單位數量：{len(unit_options)}")
    print(f"Stage 1 prompt 長度：{len(unit_prompt)} 字")

    raw_unit_output = call_ollama(unit_prompt)
    unit_result = extract_json(raw_unit_output)
    unit_prediction = validate_unit_prediction(unit_result, unit_options)

    pred_unit = unit_prediction["pred_unit"]

    print(f"預測單位：{pred_unit}")
    print(f"單位信心：{unit_prediction['unit_confidence']}")

    print("=" * 80)
    print("Stage 2：判斷該單位底下的會辦類別")

    candidate_kb = filter_kb_by_unit(kb, pred_unit)

    if not candidate_kb:
        raise ValueError(f"找不到 unit={pred_unit} 底下的候選類別")

    category_prompt = build_category_prompt(
        call_text=call_text,
        pred_unit=pred_unit,
        candidate_kb=candidate_kb
    )

    print(f"候選類別數：{len(candidate_kb)}")
    print(f"Stage 2 prompt 長度：{len(category_prompt)} 字")

    raw_category_output = call_ollama(category_prompt)
    category_result = extract_json(raw_category_output)
    category_prediction = validate_category_prediction(category_result, candidate_kb)

    final_result = {
        "input_file": str(input_path),
        "method": "two_stage_llm_classification",
        "stage_1_unit_prediction": unit_prediction,
        "stage_2_category_prediction": category_prediction,
        "final_prediction": {
            "label_name": category_prediction["pred_label_name"],
            "doc_type": category_prediction["pred_doc_type"],
            "unit": category_prediction["pred_unit"],
            "category": category_prediction["pred_category"],
            "confidence": min(
                unit_prediction["unit_confidence"],
                category_prediction["category_confidence"]
            ),
            "need_human_review": (
                unit_prediction["unit_need_human_review"]
                or category_prediction["category_need_human_review"]
            )
        }
    }

    print("=" * 80)
    print("Final Result")
    print(json.dumps(final_result, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)

        print(f"\n結果已儲存至：{output_path}")


if __name__ == "__main__":
    main()
