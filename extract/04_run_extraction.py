import argparse
import json
from pathlib import Path
from datetime import datetime

from build_examples import build_examples_text
from ollama_client import call_ollama
from json_validator import (
    parse_llm_json,
    validate_extraction_result,
    normalize_result,
)


def load_text(path: str | Path) -> str:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")

    return path.read_text(encoding="utf-8").strip()


def load_prompt_template(path: str | Path) -> str:
    return load_text(path)


def build_prompt(
    prompt_template: str,
    examples_text: str,
    dialogue: str,
    doc_type: str | None = None,
    category: str | None = None,
    unit: str | None = None,
) -> str:
    """
    將 prompt template 中的變數替換成實際內容。
    """

    prompt = prompt_template.replace("{examples}", examples_text)
    prompt = prompt.replace("{dialogue}", dialogue)

    prompt = prompt.replace("{doc_type}", doc_type or "未提供")
    prompt = prompt.replace("{category}", category or "未提供")
    prompt = prompt.replace("{unit}", unit or "未提供")

    return prompt


def get_call_id(txt_path: Path) -> str:
    return txt_path.stem


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_batch_extraction(
    calls_dir: str | Path,
    prompt_path: str | Path,
    examples_path: str | Path,
    output_path: str | Path,
    model: str,
    base_url: str,
    temperature: float,
    timeout: int,
    doc_type: str | None = None,
    category: str | None = None,
    unit: str | None = None,
):
    calls_dir = Path(calls_dir)

    if not calls_dir.exists():
        raise FileNotFoundError(f"找不到 calls 資料夾：{calls_dir}")

    txt_files = sorted(calls_dir.glob("*.txt"))

    if not txt_files:
        raise ValueError(f"{calls_dir} 裡面沒有 .txt 檔案")

    prompt_template = load_prompt_template(prompt_path)

    examples_text = build_examples_text(
        examples_path=examples_path,
        doc_type=doc_type,
    )

    results = []

    for idx, txt_path in enumerate(txt_files, start=1):
        call_id = get_call_id(txt_path)

        print(f"[{idx}/{len(txt_files)}] 處理中：{call_id}")

        dialogue = load_text(txt_path)

        prompt = build_prompt(
            prompt_template=prompt_template,
            examples_text=examples_text,
            dialogue=dialogue,
            doc_type=doc_type,
            category=category,
            unit=unit,
        )

        record = {
            "call_id": call_id,
            "source_file": str(txt_path),
            "doc_type": doc_type,
            "category": category,
            "unit": unit,
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

        results.append(record)

        write_jsonl(output_path, results)

    print(f"\n完成，結果已輸出至：{output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="批次擷取客服通話紀錄中的問題描述與需求內容"
    )

    parser.add_argument(
        "--calls-dir",
        default="data/calls",
        help="存放 .txt 通話紀錄的資料夾"
    )

    parser.add_argument(
        "--prompt-path",
        default="prompts/extract_call_form.prompt",
        help="Prompt template 路徑"
    )

    parser.add_argument(
        "--examples-path",
        default="data/call_extraction_gold_examples.json",
        help="需求方確認範例 JSON 路徑"
    )

    parser.add_argument(
        "--output-path",
        default="data/outputs/extracted.jsonl",
        help="輸出 JSONL 路徑"
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
        help="模型 temperature"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Ollama timeout 秒數"
    )

    parser.add_argument(
        "--doc-type",
        default=None,
        help="會辦類型，例如：行政會辦單、業務會辦單"
    )

    parser.add_argument(
        "--category",
        default=None,
        help="會辦分類"
    )

    parser.add_argument(
        "--unit",
        default=None,
        help="承辦單位"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_batch_extraction(
        calls_dir=args.calls_dir,
        prompt_path=args.prompt_path,
        examples_path=args.examples_path,
        output_path=args.output_path,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        timeout=args.timeout,
        doc_type=args.doc_type,
        category=args.category,
        unit=args.unit,
    )


if __name__ == "__main__":
    main()
