"""
通話記錄擷取系統
用途：讀取 calls/ 資料夾中的 .txt 通話記錄，利用 Ollama 模型擷取「問題描述」與「需求內容」
"""

import json
import os
import re
import time
from pathlib import Path

import requests

# ── 設定區 ──────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gpt-oss:20b"

CALLS_DIR = Path("calls")                              # 通話記錄資料夾
PROMPT_FILE = Path("extract_call_form.prompt")         # Prompt 模板
EXAMPLES_FILE = Path("call_extraction_examples.json")  # 6 筆 few-shot 範例
OUTPUT_FILE = Path("extraction_results.json")          # 輸出結果
LOG_FILE = Path("extraction_log.txt")                  # 錯誤紀錄
# ────────────────────────────────────────────────────────


def load_prompt_template(prompt_file: Path) -> str:
    """載入 prompt 模板"""
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


def load_examples(examples_file: Path) -> list[dict]:
    """載入 few-shot 範例"""
    with open(examples_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_few_shot_block(examples: list[dict]) -> str:
    """
    將 6 筆範例轉成 few-shot 文字區塊
    優先使用 corrected_* 欄位（人工修正後的答案）
    """
    blocks = []
    for ex in examples:
        call_id = ex.get("call_id", "unknown")
        problem = ex.get("corrected_problem_description") or ex.get("original_problem_description", "")
        request = ex.get("corrected_request_content") or ex.get("original_request_content", "")

        # 讀取對應的通話記錄內容（若存在）
        call_path = CALLS_DIR / f"{call_id}.txt"
        if call_path.exists():
            with open(call_path, "r", encoding="utf-8") as f:
                call_text = f.read().strip()
        else:
            call_text = "（通話記錄不存在）"

        blocks.append(
            f"### 範例\n"
            f"【通話記錄】\n{call_text}\n\n"
            f"【擷取結果】\n"
            f"問題描述：{problem}\n"
            f"需求內容：{request}"
        )

    return "\n\n---\n\n".join(blocks)


def build_prompt(template: str, few_shot_block: str, call_text: str) -> str:
    """
    將 prompt 模板、few-shot 範例、通話記錄組合成最終 prompt
    模板中請使用 {few_shot_examples} 與 {call_text} 作為佔位符
    """
    return (
        template
        .replace("{few_shot_examples}", few_shot_block)
        .replace("{call_text}", call_text)
    )


def call_ollama(prompt: str, retries: int = 3) -> str:
    """呼叫 Ollama API，失敗時最多重試 retries 次"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,   # 低溫，讓輸出穩定
            "num_predict": 512,
        }
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"  [!] 第 {attempt} 次請求失敗：{e}")
            if attempt < retries:
                time.sleep(3)

    return ""


def parse_model_output(raw_output: str) -> dict:
    """
    從模型輸出中解析出兩個欄位
    支援格式：
      問題描述：xxx
      需求內容：xxx
    """
    result = {
        "problem_description": "",
        "request_content": "",
        "raw_output": raw_output,
    }

    patterns = {
        "problem_description": r"問題描述[：:]\s*(.+?)(?=需求內容[：:]|$)",
        "request_content": r"需求內容[：:]\s*(.+?)(?=$)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, raw_output, re.DOTALL)
        if match:
            result[key] = match.group(1).strip()

    return result


def log_error(log_file: Path, call_id: str, message: str):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{call_id}] {message}\n")


def main():
    print("=" * 50)
    print("通話記錄擷取系統啟動")
    print("=" * 50)

    # 載入資源
    template = load_prompt_template(PROMPT_FILE)
    examples = load_examples(EXAMPLES_FILE)
    few_shot_block = build_few_shot_block(examples)

    # 取得範例中已有答案的 call_id，跑全部時跳過（可選）
    example_ids = {ex["call_id"] for ex in examples}

    # 掃描所有通話記錄
    call_files = sorted(CALLS_DIR.glob("*.txt"))
    print(f"找到 {len(call_files)} 筆通話記錄\n")

    # 讀取既有結果（支援中斷續跑）
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
        processed_ids = {r["call_id"] for r in results}
        print(f"已有 {len(processed_ids)} 筆處理完畢，跳過...\n")
    else:
        results = []
        processed_ids = set()

    # 逐筆處理
    for i, call_path in enumerate(call_files, 1):
        call_id = call_path.stem
        print(f"[{i:02d}/{len(call_files)}] 處理中：{call_id}")

        # 已處理過則跳過
        if call_id in processed_ids:
            print("  → 已處理，跳過")
            continue

        # 讀取通話記錄
        with open(call_path, "r", encoding="utf-8") as f:
            call_text = f.read().strip()

        if not call_text:
            log_error(LOG_FILE, call_id, "通話記錄為空")
            print("  → 空檔案，跳過")
            continue

        # 組 prompt 並呼叫模型
        prompt = build_prompt(template, few_shot_block, call_text)
        raw_output = call_ollama(prompt)

        if not raw_output:
            log_error(LOG_FILE, call_id, "模型無回應")
            print("  → 模型無回應，記錄錯誤")
            continue

        # 解析輸出
        parsed = parse_model_output(raw_output)
        is_example = call_id in example_ids

        record = {
            "call_id": call_id,
            "is_example": is_example,                              # 標記是否為 few-shot 範例
            "problem_description": parsed["problem_description"],
            "request_content": parsed["request_content"],
            "raw_output": parsed["raw_output"],                    # 保留原始輸出方便 debug
            "status": "extracted",
        }

        results.append(record)
        processed_ids.add(call_id)

        print(f"  ✓ 問題描述：{parsed['problem_description'][:40]}...")
        print(f"  ✓ 需求內容：{parsed['request_content'][:40]}...")

        # 每筆處理完就存檔（防止中途中斷遺失）
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(0.5)  # 避免 Ollama 過載

    print("\n" + "=" * 50)
    print(f"完成！共處理 {len(results)} 筆")
    print(f"結果輸出至：{OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    main()
