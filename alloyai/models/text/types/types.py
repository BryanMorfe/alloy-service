from pydantic import BaseModel
from enum import Enum

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"


class Message(BaseModel):
    role: MessageRole
    content: str

class ChatResponse(BaseModel):
    messages: list[Message]
