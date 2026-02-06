from __future__ import annotations

from .ollama_text_model import OllamaTextModel


class Qwen3MediumModel(OllamaTextModel):
    model_id = "qwen3-medium"
    required_vram_mb = 150000
    spreadable = True
    supports_concurrent_requests = True

    def __init__(self) -> None:
        super().__init__(ollama_model="qwen3-medium")


class Qwen3LiteModel(OllamaTextModel):
    model_id = "qwen3-lite"
    required_vram_mb = 15000
    spreadable = True
    supports_concurrent_requests = True

    def __init__(self) -> None:
        super().__init__(ollama_model="qwen3-lite")


class GptOSS120BModel(OllamaTextModel):
    model_id = "gpt-oss-120b"
    required_vram_mb = 70000
    spreadable = True
    supports_concurrent_requests = True

    def __init__(self) -> None:
        super().__init__(ollama_model="gpt-oss:120b")


class GptOSS20BModel(OllamaTextModel):
    model_id = "gpt-oss-20b"
    required_vram_mb = 12000
    spreadable = True
    supports_concurrent_requests = True

    def __init__(self) -> None:
        super().__init__(ollama_model="gpt-oss:20b")
