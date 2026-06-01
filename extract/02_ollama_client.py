import requests


def call_ollama(
    prompt: str,
    model: str = "qwen2.5:14b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.1,
    timeout: int = 300,
) -> str:
    """
    呼叫 Ollama Server 並回傳模型輸出文字。
    """

    url = f"{base_url.rstrip('/')}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama 呼叫失敗：{e}")

    data = response.json()

    if "response" not in data:
        raise RuntimeError(f"Ollama 回傳格式異常：{data}")

    return data["response"].strip()
