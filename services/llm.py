from transformers import pipeline

class LLMClient:
    def __init__(self, model_name: str = "gpt2"):
        self.pipeline = pipeline("text-generation", model=model_name)

    def query(self, prompt: str) -> str:
        response = self.pipeline(prompt, max_length=150, num_return_sequences=1)
        return response[0]["generated_text"]