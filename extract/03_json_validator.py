import json
import re
from typing import Any


REQUIRED_KEYS = [
    "problem_description",
    "request_content"
]


def extract_json_text(text: str) -> str:
    """
    從 LLM 輸出中擷取 JSON 區塊。
    即使模型多輸出一些說明文字，也盡量抓出第一個 JSON object。
    """

    text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{[\s\S]*\}", text)

    if not match:
        raise ValueError("找不到 JSON 物件")

    return match.group(0)


def parse_llm_json(text: str) -> dict[str, Any]:
    """
    解析 LLM 回傳文字為 JSON dict。
    """

    json_text = extract_json_text(text)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失敗：{e}")

    if not isinstance(data, dict):
        raise ValueError("LLM 輸出不是 JSON object")

    return data


def validate_extraction_result(data: dict[str, Any]) -> dict[str, Any]:
    """
    驗證 LLM 擷取結果是否包含必要欄位。
    """

    errors = []

    for key in REQUIRED_KEYS:
        if key not in data:
            errors.append(f"缺少欄位：{key}")
        elif not isinstance(data[key], str):
            errors.append(f"欄位 {key} 應為字串")
        elif not data[key].strip():
            errors.append(f"欄位 {key} 不可為空")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    }


def normalize_result(data: dict[str, Any]) -> dict[str, str]:
    """
    只保留正式需要輸出的欄位。
    """

    return {
        "problem_description": str(data.get("problem_description", "")).strip(),
        "request_content": str(data.get("request_content", "")).strip()
    }
