from pathlib import Path
from collections import defaultdict
import json
import re
import time
import requests


# =========================
# 基本設定
# =========================

OLLAMA_HOST = ""
MODEL_NAME = "gpt-oss:20b"

DATASET_PATH = Path("processed/train_dataset.json")
CALLS_DIR = Path("train_data")

OUTPUT_PATH = Path("processed/category_knowledge_base.json")
ERROR_LOG_PATH = Path("processed/category_knowledge_base_errors.json")

MAX_EXAMPLES_PER_CATEGORY = 5
MAX_CALL_CHARS = 3500
SLEEP_SECONDS = 1


# =========================
# 工具函式
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


def read_call_text(item: dict) -> str:
    """
    依照 train_data/doc_type/unit/category/call_id.txt 讀取通話內容
    """
    path = (
        CALLS_DIR
        / item["doc_type"]
        / item["unit"]
        / item["category"]
        / f"{item['call_id']}.txt"
    )

    if not path.exists():
        raise FileNotFoundError(f"找不到通話紀錄：{path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"通話紀錄是空的：{path}")

    return text[:MAX_CALL_CHARS]


def group_by_label(dataset: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)

    required_keys = [
        "call_id",
        "doc_type",
        "unit",
        "category",
        "label_name",
    ]

    for item in dataset:
        for key in required_keys:
            if key not in item:
                raise KeyError(f"train_dataset.json 缺少欄位：{key}")

        grouped[item["label_name"]].append(item)

    return dict(grouped)


def get_data_sufficiency(num_examples: int) -> str:
    if num_examples == 1:
        return "low"
    elif num_examples <= 3:
        return "medium"
    return "high"


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
    return json.loads(json_text)


def call_ollama(prompt: str) -> str:
    url = f"{OLLAMA_HOST}/api/generate"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_ctx": 8192,
        },
    }

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()

    data = response.json()

    if "response" not in data:
        raise RuntimeError(f"Ollama 回傳格式異常：{data}")

    return data["response"]


def build_prompt(
    label_name: str,
    items: list[dict],
    data_sufficiency: str,
) -> str:
    first = items[0]
    examples = []

    for item in items[:MAX_EXAMPLES_PER_CATEGORY]:
        call_id = item["call_id"]
        call_text = read_call_text(item)

        examples.append({
            "call_id": call_id,
            "call_text": call_text,
        })

    num_examples = len(items)

    prompt = f"""
你是一位保險客服會辦分類知識庫整理助手。

現在我要建立 category_knowledge_base.json。
用途是：之後輸入一通新的客服通話紀錄時，LLM 可以根據這份知識庫判斷它應該屬於哪一個會辦分類。

請你根據「已知分類資料」與「通話案例」整理此分類的分類規則。

重要限制：
1. 請只根據提供的通話案例整理，不要編造不存在的業務流程。
2. 請輸出合法 JSON，不要輸出 markdown，不要加解釋文字。
3. label_name、doc_type、unit、category 必須完全照我提供的內容。
4. decision_rules 要寫成之後可以用來分類新通話的判斷準則。
5. negative_rules 要寫「什麼情況不應該分到這一類」。
6. required_evidence_from_call 要寫「通話中至少需要出現哪些證據，才適合判定為此類」。
7. 如果此分類只有少量案例，請不要把單一案例中的偶然細節當成通用規則。
8. 請只整理穩定、可泛化的分類依據。
9. 如果案例不足，請在 limitations 裡清楚說明。

已知分類資料：
{{
  "label_name": "{label_name}",
  "doc_type": "{first.get("doc_type", "")}",
  "unit": "{first.get("unit", "")}",
  "category": "{first.get("category", "")}",
  "num_examples": {num_examples},
  "data_sufficiency": "{data_sufficiency}"
}}

通話案例：
{json.dumps(examples, ensure_ascii=False, indent=2)}

請輸出以下 JSON 格式，欄位名稱不可更改：

{{
  "label_name": "{label_name}",
  "doc_type": "{first.get("doc_type", "")}",
  "unit": "{first.get("unit", "")}",
  "category": "{first.get("category", "")}",
  "definition": "",
  "main_customer_intents": [],
  "keywords": [],
  "decision_rules": [],
  "negative_rules": [],
  "required_evidence_from_call": [],
  "possible_confusing_categories": [],
  "example_call_ids": [],
  "num_examples": {num_examples},
  "data_sufficiency": "{data_sufficiency}",
  "limitations": ""
}}
"""
    return prompt.strip()


def validate_kb_item(
    kb_item: dict,
    source_item: dict,
    num_examples: int,
    data_sufficiency: str,
) -> dict:
    """
    強制修正關鍵欄位，避免 LLM 改掉 label_name / unit / category。
    """
    kb_item["label_name"] = source_item["label_name"]
    kb_item["doc_type"] = source_item["doc_type"]
    kb_item["unit"] = source_item["unit"]
    kb_item["category"] = source_item["category"]
    kb_item["num_examples"] = num_examples
    kb_item["data_sufficiency"] = data_sufficiency

    list_fields = [
        "main_customer_intents",
        "keywords",
        "decision_rules",
        "negative_rules",
        "required_evidence_from_call",
        "possible_confusing_categories",
        "example_call_ids",
    ]

    for field in list_fields:
        if field not in kb_item or not isinstance(kb_item[field], list):
            kb_item[field] = []

    string_fields = [
        "definition",
        "limitations",
    ]

    for field in string_fields:
        if field not in kb_item or not isinstance(kb_item[field], str):
            kb_item[field] = ""

    return kb_item


# =========================
# 主程式
# =========================

def main():
    dataset = load_json(DATASET_PATH)

    # 保險：如果 JSON 裡有 split 欄位，只保留 train
    dataset = [
        item for item in dataset
        if item.get("split", "train") == "train"
    ]

    grouped = group_by_label(dataset)

    knowledge_base = []
    errors = []

    print(f"總 train 通話資料筆數：{len(dataset)}")
    print(f"總分類數：{len(grouped)}")

    for idx, (label_name, items) in enumerate(grouped.items(), start=1):
        num_examples = len(items)
        data_sufficiency = get_data_sufficiency(num_examples)

        print("=" * 80)
        print(f"[{idx}/{len(grouped)}] {label_name}")
        print(f"案例數：{num_examples}")
        print(f"資料充足度：{data_sufficiency}")

        try:
            prompt = build_prompt(
                label_name=label_name,
                items=items,
                data_sufficiency=data_sufficiency,
            )

            raw_output = call_ollama(prompt)
            kb_item = extract_json(raw_output)

            kb_item = validate_kb_item(
                kb_item=kb_item,
                source_item=items[0],
                num_examples=num_examples,
                data_sufficiency=data_sufficiency,
            )

            knowledge_base.append(kb_item)
            save_json(knowledge_base, OUTPUT_PATH)

            print("完成")

        except Exception as e:
            error_item = {
                "label_name": label_name,
                "num_examples": num_examples,
                "data_sufficiency": data_sufficiency,
                "error": str(e),
            }

            errors.append(error_item)
            save_json(errors, ERROR_LOG_PATH)

            print("失敗")
            print(e)

        time.sleep(SLEEP_SECONDS)

    save_json(knowledge_base, OUTPUT_PATH)

    if errors:
        save_json(errors, ERROR_LOG_PATH)

    print("=" * 80)
    print("全部處理完成")
    print(f"成功分類數：{len(knowledge_base)}")
    print(f"失敗分類數：{len(errors)}")
    print(f"輸出檔案：{OUTPUT_PATH}")

    if errors:
        print(f"錯誤紀錄：{ERROR_LOG_PATH}")


if __name__ == "__main__":
    main()
