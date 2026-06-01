import json
from pathlib import Path
from typing import Optional


def load_json(path: str | Path) -> list[dict]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到範例檔案：{path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Gold examples 檔案格式應該是 list[dict]")

    return data


def build_examples_text(
    examples_path: str | Path,
    doc_type: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> str:
    """
    將 call_extraction_gold_examples.json 轉成 Prompt 可直接使用的 few-shot examples 文字。

    預期每筆 example 至少包含：
    - doc_type
    - category
    - unit
    - dialogue
    - corrected_problem_description
    - corrected_request_content
    """

    examples = load_json(examples_path)

    if doc_type:
        examples = [
            item for item in examples
            if item.get("doc_type") == doc_type
        ]

    if max_examples:
        examples = examples[:max_examples]

    if not examples:
        return "目前無可參考之需求方確認範例。"

    blocks = []

    for idx, item in enumerate(examples, start=1):
        block = f"""
Example {idx}

會辦類型：
{item.get("doc_type", "未提供")}

會辦分類：
{item.get("category", "未提供")}

承辦單位：
{item.get("unit", "未提供")}

通話內容：
{item.get("dialogue", "").strip()}

正確輸出：
{{
    "problem_description": "{item.get("corrected_problem_description", "").strip()}",
    "request_content": "{item.get("corrected_request_content", "").strip()}"
}}
""".strip()

        blocks.append(block)

    return "\n\n" + ("=" * 60).join(blocks) + "\n"


if __name__ == "__main__":
    examples_text = build_examples_text(
        "data/call_extraction_examples.json"
    )
    print(examples_text)
