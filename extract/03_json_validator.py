import json
import re


def extract_json(text: str):

    if not text:
        return None

    text = text.strip()

    # ```json ... ```
    match = re.search(
        r"```(?:json)?\s*(.*?)```",
        text,
        re.DOTALL
    )

    if match:
        text = match.group(1).strip()

    try:
        return json.loads(text)

    except Exception:
        return None


def validate_result(data):

    if not isinstance(data, dict):
        return False

    required_fields = [
        "problem_description",
        "request_content"
    ]

    return all(
        field in data
        for field in required_fields
    )
