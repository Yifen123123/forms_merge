import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Any

from utils.ollama_client import call_ollama
from utils.json_validator import (
    parse_llm_json,
    validate_extraction_result,
    normalize_result,
)


DEFAULT_CALLS_DIR = "data/calls"
DEFAULT_EXAMPLES_PATH = "data/call_extraction_examples.json"
DEFAULT_PROMPT_PATH = "prompts/extract_call_form.prompt"
DEFAULT_OUTPUT_PATH = "data/outputs/extracted_call_form.jsonl"


def load_text(path: str | Path) -> str:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")

    return path.read_text(encoding="utf-8").strip()


def load_json(path: str | Path) -> Any:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到 JSON 檔案：{path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_few_shot_examples(
    examples_path: str | Path,
    calls_dir: str | Path,
    max_examples: int | None = None,
) -> str:
    """
    根據 call_extraction_examples.json 建立 few-shot examples。

    預期 examples JSON 每筆至少包含：
    - call_id
    - doc_type
    - category
    - unit
    - corrected_problem_description
    - corrected_request_content

    並且 data/calls/{call_id}.txt 存在。
    """

    examples = load_json(examples_path)

    if not isinstance(examples, list):
        raise ValueError("call_extraction_examples.json 最外層應該是 list")

    if max_examples is not None:
        examples = examples[:max_examples]

    calls_dir = Path(calls_dir)
    blocks = []

    for idx, item in enumerate(examples, start=1):
        call_id = str(item.get("call_id", "")).strip()

        if not call_id:
            raise ValueError(f"第 {idx} 筆 example 缺少 call_id")

        call_path = calls_dir / f"{call_id}.txt"

        if not call_path.exists():
            raise FileNotFoundError(
                f"找不到 example 對應的通話檔案：{call_path}"
            )

        dialogue = load_text(call_path)

        problem = str(
            item.get("corrected_problem_description", "")
        ).strip()

        request = str(
            item.get("corrected_request_content", "")
        ).strip()

        if not problem or not request:
            raise ValueError(
                f"第 {idx} 筆 example 缺少 corrected_problem_description "
                f"或 corrected_request_content"
            )

        block = f"""
Example {idx}

call_id：
{call_id}

會辦類型：
{item.get("doc_type", "未提供")}

會辦分類：
{item.get("category", "未提供")}

承辦單位：
{item.get("unit", "未提供")}

原通話內容：
{dialogue}

正確輸出：
{{
    "problem_description": "{problem}",
    "request_content": "{request}"
}}
""".strip()

        blocks.append(block)

    return "\n\n" + ("\n" + "=" * 80 + "\n").join(blocks) + "\n"


def build_prompt(
    prompt_template: str,
    examples_text: str,
    dialogue: str,
) -> str:
    return (
        prompt_template
        .replace("{examples}", examples_text)
        .replace("{dialogue}", dialogue)
    )


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_existing_call_ids(output_path: str | Path) -> set[str]:
    """
    若中途斷掉，可避免重複處理已成功輸出的 call_id。
    """

    output_path = Path(output_path)

    if not output_path.exists():
        return set()

    existing = set()

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
                call_id = item.get("call_id")

                if call_id:
                    existing.add(str(call_id))
            except json.JSONDecodeError:
                continue

    return existing


def extract_one_call(
    txt_path: Path,
    prompt_template: str,
    examples_text: str,
    model: str,
    base_url: str,
    temperature: float,
    timeout: int,
) -> dict:
    call_id = txt_path.stem
    dialogue = load_text(txt_path)

    prompt = build_prompt(
        prompt_template=prompt_template,
        examples_text=examples_text,
        dialogue=dialogue,
    )

    record = {
        "call_id": call_id,
        "source_file": str(txt_path),
        "model": model,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        raw_output = call_ollama(
            prompt=prompt,
            model=model,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
        )

        parsed = parse_llm_json(raw_output)
        validation = validate_extraction_result(parsed)

        if validation["is_valid"]:
            normalized = normalize_result(parsed)

            record.update(normalized)
            record["status"] = "success"
            record["errors"] = []
            record["raw_output"] = raw_output

        else:
            record["problem_description"] = ""
            record["request_content"] = ""
            record["status"] = "validation_failed"
            record["errors"] = validation["errors"]
            record["raw_output"] = raw_output

    except Exception as e:
        record["problem_description"] = ""
        record["request_content"] = ""
        record["status"] = "failed"
        record["errors"] = [str(e)]
        record["raw_output"] = ""

    return record


def run_batch(
    calls_dir: str | Path,
    examples_path: str | Path,
    prompt_path: str | Path,
    output_path: str | Path,
    model: str,
    base_url: str,
    temperature: float,
    timeout: int,
    max_examples: int | None,
    resume: bool,
) -> None:
    calls_dir = Path(calls_dir)
    output_path = Path(output_path)

    if not calls_dir.exists():
        raise FileNotFoundError(f"找不到 calls 資料夾：{calls_dir}")

    txt_files = sorted(calls_dir.glob("*.txt"))

    if not txt_files:
        raise ValueError(f"{calls_dir} 裡面沒有 .txt 檔案")

    prompt_template = load_text(prompt_path)

    examples_text = build_few_shot_examples(
        examples_path=examples_path,
        calls_dir=calls_dir,
        max_examples=max_examples,
    )

    existing_call_ids = get_existing_call_ids(output_path) if resume else set()

    print(f"共找到 {len(txt_files)} 筆通話檔案")
    print(f"Few-shot examples 已建立")
    print(f"輸出位置：{output_path}")

    for idx, txt_path in enumerate(txt_files, start=1):
        call_id = txt_path.stem

        if call_id in existing_call_ids:
            print(f"[{idx}/{len(txt_files)}] 略過已處理：{call_id}")
            continue

        print(f"[{idx}/{len(txt_files)}] 處理中：{call_id}")

        record = extract_one_call(
            txt_path=txt_path,
            prompt_template=prompt_template,
            examples_text=examples_text,
            model=model,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
        )

        append_jsonl(output_path, record)

        print(f"  status = {record['status']}")

    print("批次擷取完成")


def parse_args():
    parser = argparse.ArgumentParser(
        description="批次擷取客服通話紀錄中的問題描述與需求內容"
    )

    parser.add_argument(
        "--calls-dir",
        default=DEFAULT_CALLS_DIR,
        help="存放 56 筆 .txt 通話紀錄的資料夾"
    )

    parser.add_argument(
        "--examples-path",
        default=DEFAULT_EXAMPLES_PATH,
        help="需求方確認範例 JSON 檔案"
    )

    parser.add_argument(
        "--prompt-path",
        default=DEFAULT_PROMPT_PATH,
        help="Prompt template 檔案"
    )

    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="輸出 JSONL 檔案"
    )

    parser.add_argument(
        "--model",
        default="qwen2.5:7b",
        help="Ollama 模型名稱"
    )

    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama Server URL"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="模型 temperature，建議 0~0.2"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Ollama timeout 秒數"
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="最多使用幾筆 few-shot examples，預設全部使用"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="若 output 已存在，略過已處理 call_id"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_batch(
        calls_dir=args.calls_dir,
        examples_path=args.examples_path,
        prompt_path=args.prompt_path,
        output_path=args.output_path,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        timeout=args.timeout,
        max_examples=args.max_examples,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
