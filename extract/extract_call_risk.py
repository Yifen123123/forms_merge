import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.ollama_client import OllamaClient
from utils.json_validator import extract_json


MODEL_NAME = "qwen3:8b"

PROMPT_FILE = "prompts/extract_call_form.prompt"
EXAMPLE_FILE = "call_extraction_examples.json"
CALLS_DIR = "calls"

OUTPUT_JSON = "results/extracted_calls.json"
OUTPUT_CSV = "results/extracted_calls.csv"


def load_prompt() -> str:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()


def build_examples() -> str:
    with open(EXAMPLE_FILE, "r", encoding="utf-8") as f:
        examples_data = json.load(f)

    example_blocks = []

    for item in examples_data:
        problem = item.get("corrected_problem_description", "")
        request = item.get("corrected_request_content", "")

        block = f"""
【範例】

通話識別碼:
{item.get("call_id", "")}

正確輸出:

{{
  "cases": [
    {{
      "problem_description": "{problem}",
      "request_content": "{request}"
    }}
  ],
  "complaint_risk": false,
  "complaint_note": ""
}}
"""
        example_blocks.append(block)

    return "\n".join(example_blocks)


def load_dialogue(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in [
            "true",
            "1",
            "yes",
            "y",
            "是",
            "有"
        ]

    return False


def normalize_result(parsed: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(parsed, dict):
        return None

    cases = []

    if "cases" in parsed and isinstance(parsed["cases"], list):
        for case in parsed["cases"]:
            if not isinstance(case, dict):
                continue

            problem = case.get("problem_description", "")
            request = case.get("request_content", "")

            if not isinstance(problem, str):
                problem = ""

            if not isinstance(request, str):
                request = ""

            problem = problem.strip()
            request = request.strip()

            if problem or request:
                cases.append({
                    "problem_description": problem,
                    "request_content": request
                })

    elif (
        "problem_description" in parsed
        and "request_content" in parsed
    ):
        problem = parsed.get("problem_description", "")
        request = parsed.get("request_content", "")

        if not isinstance(problem, str):
            problem = ""

        if not isinstance(request, str):
            request = ""

        problem = problem.strip()
        request = request.strip()

        if problem or request:
            cases.append({
                "problem_description": problem,
                "request_content": request
            })

    complaint_risk = normalize_bool(
        parsed.get("complaint_risk", False)
    )

    complaint_note = parsed.get("complaint_note", "")

    if not isinstance(complaint_note, str):
        complaint_note = ""

    return {
        "cases": cases,
        "complaint_risk": complaint_risk,
        "complaint_note": complaint_note.strip()
    }


def extract_one_call(
    client: OllamaClient,
    prompt_template: str,
    examples: str,
    dialogue: str
) -> Dict[str, Any]:

    prompt = (
        prompt_template
        .replace("{examples}", examples)
        .replace("{dialogue}", dialogue)
    )

    raw_output = client.generate(prompt)

    parsed = extract_json(raw_output)

    if parsed is None:
        return {
            "cases": [],
            "complaint_risk": False,
            "complaint_note": "",
            "status": "json_error",
            "raw_output": raw_output
        }

    normalized = normalize_result(parsed)

    if normalized is None:
        return {
            "cases": [],
            "complaint_risk": False,
            "complaint_note": "",
            "status": "schema_error",
            "raw_output": raw_output
        }

    return {
        "cases": normalized["cases"],
        "complaint_risk": normalized["complaint_risk"],
        "complaint_note": normalized["complaint_note"],
        "status": "success",
        "raw_output": raw_output
    }


def main() -> None:
    Path("results").mkdir(exist_ok=True)

    client = OllamaClient(model=MODEL_NAME)

    prompt_template = load_prompt()
    examples = build_examples()

    call_files = sorted(Path(CALLS_DIR).glob("*.txt"))
    total = len(call_files)

    json_results: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    print(f"共找到 {total} 筆通話")

    for idx, call_file in enumerate(call_files, start=1):
        call_id = call_file.stem

        print("=" * 50)
        print(f"[{idx}/{total}] 處理 {call_id}")

        dialogue = load_dialogue(call_file)

        result = extract_one_call(
            client=client,
            prompt_template=prompt_template,
            examples=examples,
            dialogue=dialogue
        )

        status = result["status"]
        cases = result["cases"]
        complaint_risk = result["complaint_risk"]
        complaint_note = result["complaint_note"]

        print(f"status: {status}")
        print(f"case_count: {len(cases)}")
        print(f"complaint_risk: {complaint_risk}")

        json_results.append({
            "call_id": call_id,
            "status": status,
            "case_count": len(cases),
            "complaint_risk": complaint_risk,
            "complaint_note": complaint_note,
            "cases": cases,
            "raw_output": result.get("raw_output", "")
        })

        if status == "success" and cases:
            for case_idx, case in enumerate(cases, start=1):
                csv_rows.append({
                    "call_id": call_id,
                    "case_no": case_idx,
                    "problem_description": case["problem_description"],
                    "request_content": case["request_content"],
                    "complaint_risk": complaint_risk,
                    "complaint_note": complaint_note,
                    "status": status
                })
        else:
            csv_rows.append({
                "call_id": call_id,
                "case_no": "",
                "problem_description": "",
                "request_content": "",
                "complaint_risk": complaint_risk,
                "complaint_note": complaint_note,
                "status": status
            })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            json_results,
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "call_id",
                "case_no",
                "problem_description",
                "request_content",
                "complaint_risk",
                "complaint_note",
                "status"
            ]
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    success_count = sum(
        1 for r in json_results
        if r["status"] == "success"
    )

    total_cases = sum(
        r["case_count"]
        for r in json_results
    )

    complaint_count = sum(
        1 for r in json_results
        if r["complaint_risk"]
    )

    print()
    print("=" * 50)
    print(f"完成 {success_count}/{total}")
    print(f"共產生 {total_cases} 筆會辦單")
    print(f"客訴風險通話數：{complaint_count}")
    print(f"JSON -> {OUTPUT_JSON}")
    print(f"CSV  -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
