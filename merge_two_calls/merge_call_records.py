# merge_call_records.py

from pathlib import Path
import requests
import json


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:14b"  # 可改成你的模型，例如 gpt-oss:20b、qwen3:8b


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def read_txt(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"找不到檔案：{file_path}")

    return file_path.read_text(encoding="utf-8").strip()


def build_prompt(call_1: str, call_2: str) -> str:
    return f"""
你是一個客服通話紀錄資料生成器。

現在我會提供兩筆原始通話紀錄，這兩筆資料中分別包含不同客戶的不同需求。

你的任務是：
1. 將兩筆通話內容合併成「同一位客戶」與客服之間的一通完整電話。
2. 使用同一組假個資，個資可由你合理生成。
3. 對話中要先處理第一筆通話紀錄的需求。
4. 接著客戶要自然地補充第二筆通話紀錄的需求。
5. 兩個需求都要保留原本的重點，不可以遺漏重要資訊。
6. 不要寫成摘要，要寫成「客服與客戶的逐字稿對話」。
7. 語氣要自然，像真實客服通話。
8. 請使用繁體中文。
9. 請避免出現「根據第一筆資料」、「根據第二筆資料」這種說法。
10. 請不要輸出 JSON，只輸出完整通話紀錄文字。

假個資格式可包含：
- 客戶姓名
- 身分證字號
- 電話
- 出生年月日
- 保單號碼或案件編號，如果原文有提到可沿用；沒有則可合理生成

以下是第一筆通話紀錄：

【第一筆通話紀錄開始】
{call_1}
【第一筆通話紀錄結束】

以下是第二筆通話紀錄：

【第二筆通話紀錄開始】
{call_2}
【第二筆通話紀錄結束】

請生成合併後的一通完整客服通話紀錄。
"""


def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_ctx": 8192,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300,
    )

    response.raise_for_status()
    result = response.json()

    return result.get("response", "").strip()


def merge_two_call_records(file_1: str, file_2: str, output_file: str):
    call_1 = read_txt(DATA_DIR / file_1)
    call_2 = read_txt(DATA_DIR / file_2)

    prompt = build_prompt(call_1, call_2)

    merged_text = call_ollama(prompt)

    output_path = OUTPUT_DIR / output_file
    output_path.write_text(merged_text, encoding="utf-8")

    print(f"已完成合併通話紀錄：{output_path}")


def main():
    merge_two_call_records(
        file_1="data_001.txt",
        file_2="data_002.txt",
        output_file="merged_data_001_002.txt",
    )


if __name__ == "__main__":
    main()
