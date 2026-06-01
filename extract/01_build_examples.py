import json
from pathlib import Path


INPUT_FILE = "data/call_extraction_examples.json"
OUTPUT_FILE = "data/examples.txt"


def load_examples(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_examples_text(examples: list[dict]) -> str:
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

    return "\n\n" + ("\n" + "=" * 80 + "\n").join(blocks)


def save_examples_text(text: str, output_path: str):
    output_file = Path(output_path)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)


def main():

    examples = load_examples(INPUT_FILE)

    examples_text = build_examples_text(examples)

    save_examples_text(
        examples_text,
        OUTPUT_FILE
    )

    print(f"已產生：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
