from ollama import Client


class OllamaClient:

    def __init__(
        self,
        host="http://localhost:11434",
        model="qwen3:8b"
    ):
        self.client = Client(host=host)
        self.model = model

    def generate(self, prompt: str) -> str:

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=False
        )

        return response["response"]
