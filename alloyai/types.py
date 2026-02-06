from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dependency
    from ollama import Image, Options
    from ollama._types import ChatResponse, EmbedResponse, JsonSchemaValue, Message
except Exception:  # pragma: no cover - optional dependency
    Image = Any
    Options = Any
    ChatResponse = Any
    EmbedResponse = Any
    JsonSchemaValue = Any
    Message = Any

__all__ = [
    "ChatResponse",
    "EmbedResponse",
    "Image",
    "JsonSchemaValue",
    "Message",
    "Options",
]
