from pathlib import Path
from collections import defaultdict
import json
import re
import requests


OLLAMA_HOST = ""
MODEL_NAME = "gpt-oss:20b"

DATASET_PATH = Path("processed/classifier_dataset.json")
CALLS_DIR = Path("calls")

# 等等把這裡改成第 24 類的 label_name
TARGET_LABEL_NAME = "請填入第24類的label_name"

MAX_EXAMPLES_PER_CATEGORY = 3
MAX_CALL_CHARS = 2000


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_call_text(call_id: str) -> str:
    path = CALLS_DIR / f"{call_id}.txt"

    if not path.exists():
        raise FileNotFoundError(f"找不到通話紀錄：{path}")

    return path.read_text(encoding="utf-8").strip()[:MAX_CALL_CHARS]


def group_by_label(dataset: list[dict]) -> dict:
    grouped = defaultdict(list)

    for item in dataset:
        grouped[item["label_name"]].append(item)

    return grouped


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
    return json.loads(clean_llm_json_text(text))


def call_ollama(prompt: str) -> str:
    url = f"{OLLAMA_HOST}/api/generate"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_ctx": 8192
        }
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()

    return response.json()["response"]


def build_prompt(label_name: str, items: list[dict], data_sufficiency: str) -> str:
    first = items[0]
    num_examples = len(items)

    examples = []

    for item in items[:MAX_EXAMPLES_PER_CATEGORY]:
        call_id = item["call_id"]
        examples.append({
            "call_id": call_id,
            "call_text": read_call_text(call_id)
        })

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


def validate_kb_item(kb_item: dict, source_item: dict, num_examples: int, data_sufficiency: str):
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
        "example_call_ids"
    ]

    for field in list_fields:
        if field not in kb_item or not isinstance(kb_item[field], list):
            kb_item[field] = []

    if "definition" not in kb_item:
        kb_item["definition"] = ""

    if "limitations" not in kb_item:
        kb_item["limitations"] = ""

    return kb_item


def main():
    dataset = load_json(DATASET_PATH)
    grouped = group_by_label(dataset)

    if TARGET_LABEL_NAME not in grouped:
        print(f"找不到這個 label_name：{TARGET_LABEL_NAME}")
        print("\n目前可用的 label_name：")
        for label in grouped.keys():
            print("-", label)
        return

    items = grouped[TARGET_LABEL_NAME]

    num_examples = len(items)
    data_sufficiency = get_data_sufficiency(num_examples)

    print(f"目標類別：{TARGET_LABEL_NAME}")
    print(f"案例數：{num_examples}")
    print(f"資料充足度：{data_sufficiency}")

    prompt = build_prompt(
        label_name=TARGET_LABEL_NAME,
        items=items,
        data_sufficiency=data_sufficiency
    )

    print(f"Prompt 長度：{len(prompt)} 字")
    print("開始呼叫 Ollama...")

    raw_output = call_ollama(prompt)

    print("Ollama 回傳完成，開始解析 JSON...")

    kb_item = extract_json(raw_output)

    kb_item = validate_kb_item(
        kb_item=kb_item,
        source_item=items[0],
        num_examples=num_examples,
        data_sufficiency=data_sufficiency
    )

    print("\n請將以下 JSON 貼回 processed/category_knowledge_base.json：\n")
    print(json.dumps(kb_item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
