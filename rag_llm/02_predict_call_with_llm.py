from pathlib import Path
import argparse
import json
import requests
import time

from utils.timing_utils import TimingRecorder


OLLAMA_HOST = "http://10.67.75.157:11434"
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


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def call_ollama(prompt: str) -> str:
    url = f"{OLLAMA_HOST}/api/generate"

    payload = {
        "model": MODEL_NAME,
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


def build_unit_options(kb: list[dict]) -> list[dict]:
    units = sorted(
        set(
            item.get("unit", "")
            for item in kb
            if item.get("unit", "")
        )
    )

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
系統規則：
你不是聊天助手。
你不是解釋助手。
你是 JSON API。
你只能回傳一個 JSON object。
禁止輸出分析過程。
禁止輸出 reasoning。
禁止輸出任何 JSON 以外的文字。

任務：
你是一個保險客服通話的會辦單位分類系統。
請根據新通話內容，判斷這通電話最應該送到哪一個會辦單位。

重要限制：
1. 你只能從 provided_unit_options 中選擇一個 unit_index。
2. unit_index 必須是整數，範圍只能是 0 到 {max_index}。
3. 禁止自行新增、改寫、翻譯任何單位名稱。
4. 如果信心不足，也必須選出最可能的一個 unit_index，但 confidence 要降低。
5. 第一個字元必須是 {{。
6. 最後一個字元必須是 }}。
7. 不要輸出 markdown。
8. 不要輸出說明文字。

provided_unit_options:
{json.dumps(unit_options, ensure_ascii=False, indent=2)}

新通話內容：
\"\"\"
{call_text}
\"\"\"

請只輸出以下 JSON 格式：

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

    try:
        confidence = float(result.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    evidence = result.get("evidence_from_call", [])
    if not isinstance(evidence, list):
        evidence = []

    return {
        "pred_unit_index": pred_index,
        "pred_unit": unit_map[pred_index],
        "unit_confidence": confidence,
        "unit_reason": result.get("reason", ""),
        "unit_evidence_from_call": evidence,
        "unit_need_human_review": bool(result.get("need_human_review", False)) or confidence < CONFIDENCE_THRESHOLD
    }


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
系統規則：
你不是聊天助手。
你不是解釋助手。
你是 JSON API。
你只能回傳一個 JSON object。
禁止輸出分析過程。
禁止輸出 reasoning。
禁止輸出任何 JSON 以外的文字。

任務：
你是一個保險客服通話的會辦類別分類系統。

目前第一階段已判斷此通話的會辦單位為：
{pred_unit}

請只在該單位底下的候選會辦類別中，選出最適合的一個分類。

重要限制：
1. 你只能從 candidate_categories 中選擇一個 label_index。
2. label_index 必須是整數，範圍只能是 0 到 {max_index}。
3. 禁止自行新增、改寫、翻譯、縮寫任何 label_name。
4. 最後 label_name 會由程式根據 label_index 自動反查，因此你不需要輸出 label_name。
5. 如果信心不足，也必須選出最可能的一個 label_index，但 confidence 要降低。
6. 如果某類 data_sufficiency = low，請不要過度自信。
7. 請根據通話中的主要需求、處理方向、明確證據判斷，不要只靠關鍵字。
8. 第一個字元必須是 {{。
9. 最後一個字元必須是 }}。
10. 不要輸出 markdown。
11. 不要輸出說明文字。

candidate_categories:
{json.dumps(candidate_rules, ensure_ascii=False, indent=2)}

新通話內容：
\"\"\"
{call_text}
\"\"\"

請只輸出以下 JSON 格式：

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
        "category_evidence_from_call": evidence,
        "possible_alternatives": cleaned_alternatives,
        "category_need_human_review": bool(result.get("need_human_review", False)) or confidence < CONFIDENCE_THRESHOLD
    }


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
    timer = TimingRecorder()
    total_start = time.perf_counter()

    args = parse_args()
    input_path = Path(args.input)

    with timer.measure("load_kb_seconds"):
        kb = load_json(KB_PATH)

    if not isinstance(kb, list):
        raise ValueError("category_knowledge_base.json 應該是一個 list")

    with timer.measure("read_input_seconds"):
        call_text = read_text(input_path)

    print("=" * 80)
    print("Stage 1：判斷會辦單位")

    with timer.measure("stage1_build_unit_options_seconds"):
        unit_options = build_unit_options(kb)

    with timer.measure("stage1_prompt_build_seconds"):
        unit_prompt = build_unit_prompt(call_text, unit_options)

    print(f"單位數量：{len(unit_options)}")
    print(f"Stage 1 prompt 長度：{len(unit_prompt)} 字")
    print("開始呼叫 Ollama...")

    with timer.measure("stage1_llm_seconds"):
        unit_result = call_llm_json(
            unit_prompt,
            retry_prompt_hint='請輸出：{"pred_unit_index": 0, "confidence": 0.0, "reason": "", "evidence_from_call": [], "need_human_review": false}'
        )

    with timer.measure("stage1_validation_seconds"):
        unit_prediction = validate_unit_prediction(unit_result, unit_options)

    pred_unit = unit_prediction["pred_unit"]

    print(f"預測單位：{pred_unit}")
    print(f"單位信心：{unit_prediction['unit_confidence']}")

    print("=" * 80)
    print("Stage 2：判斷該單位底下的會辦類別")

    with timer.measure("stage2_candidate_filter_seconds"):
        candidate_kb = filter_kb_by_unit(kb, pred_unit)

    if not candidate_kb:
        raise ValueError(f"找不到 unit={pred_unit} 底下的候選類別")

    with timer.measure("stage2_prompt_build_seconds"):
        category_prompt = build_category_prompt(
            call_text=call_text,
            pred_unit=pred_unit,
            candidate_kb=candidate_kb
        )

    print(f"候選類別數：{len(candidate_kb)}")
    print(f"Stage 2 prompt 長度：{len(category_prompt)} 字")
    print("開始呼叫 Ollama...")

    with timer.measure("stage2_llm_seconds"):
        category_result = call_llm_json(
            category_prompt,
            retry_prompt_hint='請輸出：{"pred_label_index": 0, "confidence": 0.0, "reason": "", "evidence_from_call": [], "possible_alternatives": [], "need_human_review": false}'
        )

    with timer.measure("stage2_validation_seconds"):
        category_prediction = validate_category_prediction(
            category_result,
            candidate_kb
        )

    final_confidence = min(
        unit_prediction["unit_confidence"],
        category_prediction["category_confidence"]
    )

    timer.add(
        "total_script_seconds",
        time.perf_counter() - total_start
    )

    final_result = {
        "timing": timer.get_timings(),
        "input_file": str(input_path),
        "method": "two_stage_llm_classification_with_retry",
        "stage_1_unit_prediction": unit_prediction,
        "stage_2_category_prediction": category_prediction,
        "final_prediction": {
            "label_name": category_prediction["pred_label_name"],
            "doc_type": category_prediction["pred_doc_type"],
            "unit": category_prediction["pred_unit"],
            "category": category_prediction["pred_category"],
            "confidence": final_confidence,
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
        save_json(final_result, output_path)
        print(f"\n結果已儲存至：{output_path}")


if __name__ == "__main__":
    main()
