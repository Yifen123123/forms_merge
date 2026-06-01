import json
import csv
from pathlib import Path

from utils.ollama_client import OllamaClient
from utils.json_validator import (
    extract_json,
    validate_result
)


# =========================
# 設定
# =========================

MODEL_NAME = "qwen3:8b"

PROMPT_FILE = (
    "prompts/extract_call_form.prompt"
)

EXAMPLE_FILE = (
    "call_extraction_examples.json"
)

CALLS_DIR = "calls"

OUTPUT_JSON = (
    "results/extracted_calls.json"
)

OUTPUT_CSV = (
    "results/extracted_calls.csv"
)


# =========================
# 讀 Prompt
# =========================

def load_prompt():

    with open(
        PROMPT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


# =========================
# Few Shot
# =========================

def build_examples():

    with open(
        EXAMPLE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        examples_data = json.load(f)

    example_blocks = []

    for item in examples_data:

        block = f"""
【範例】

通話識別碼:
{item.get("call_id", "")}

正確輸出:

{{
    "problem_description":
    "{item.get('corrected_problem_description','')}",

    "request_content":
    "{item.get('corrected_request_content','')}"
}}
"""
        example_blocks.append(block)

    return "\n".join(example_blocks)


# =========================
# 讀通話
# =========================

def load_dialogue(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


# =========================
# LLM 擷取
# =========================

def extract_one_call(
    client,
    prompt_template,
    examples,
    dialogue
):

    prompt = prompt_template.format(
        examples=examples,
        dialogue=dialogue
    )

    raw_output = client.generate(prompt)

    parsed = extract_json(raw_output)

    if parsed is None:

        return {
            "problem_description": "",
            "request_content": "",
            "status": "json_error"
        }

    if not validate_result(parsed):

        return {
            "problem_description": "",
            "request_content": "",
            "status": "schema_error"
        }

    return {
        "problem_description":
            parsed["problem_description"],

        "request_content":
            parsed["request_content"],

        "status": "success"
    }


# =========================
# Main
# =========================

def main():

    Path("results").mkdir(
        exist_ok=True
    )

    client = OllamaClient(
        model=MODEL_NAME
    )

    prompt_template = load_prompt()

    examples = build_examples()

    call_files = sorted(
        Path(CALLS_DIR).glob("*.txt")
    )

    total = len(call_files)

    results = []

    print(f"共找到 {total} 筆通話")

    for idx, call_file in enumerate(
        call_files,
        start=1
    ):

        call_id = call_file.stem

        print(
            f"[{idx}/{total}] "
            f"處理 {call_id}"
        )

        dialogue = load_dialogue(
            call_file
        )

        result = extract_one_call(
            client,
            prompt_template,
            examples,
            dialogue
        )

        results.append({
            "call_id": call_id,
            **result
        })

    # =====================
    # JSON
    # =====================

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    # =====================
    # CSV
    # =====================

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "call_id",
                "problem_description",
                "request_content",
                "status"
            ]
        )

        writer.writeheader()

        writer.writerows(results)

    success_count = sum(
        1
        for r in results
        if r["status"] == "success"
    )

    print()
    print("=" * 50)
    print(
        f"完成 "
        f"{success_count}/{total}"
    )
    print(
        f"JSON -> {OUTPUT_JSON}"
    )
    print(
        f"CSV  -> {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
