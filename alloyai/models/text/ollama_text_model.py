from __future__ import annotations

import subprocess
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Union, overload, Literal

from ollama import Client, Image, Options
from ollama._types import Message, ChatResponse, JsonSchemaValue

from .. import EventPublisher, Model, Modality, ModelCapability

ToolLike = Union[Mapping[str, Any], Callable[..., Any]]
KeepAlive = Optional[Union[float, str, bool]]

_OLLAMA_KEEP_ALIVE_FOREVER = True


class OllamaTextModel(EventPublisher, Model):
    spreadable = False
    supports_concurrent_requests = False
    priority = 1
    required_vram_mb: int = 0
    capabilities = [
        ModelCapability(inputs={Modality.TEXT}, outputs={Modality.TEXT}, name="text-to-text"),
    ]

    def __init__(
        self,
        ollama_model: str,
        *,
        host: Optional[str] = None,
        keep_alive: KeepAlive = True,
        required_vram_mb: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.ollama_model = ollama_model
        if not getattr(self, "model_id", None):
            self.model_id = ollama_model
        self._client = Client(host=host) if host else Client()
        _ = keep_alive
        if required_vram_mb is not None:
            self.required_vram_mb = int(required_vram_mb)

    @overload
    def __call__(
        self,
        model="",
        messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
        *,
        tools: Optional[Sequence[ToolLike]] = None,
        stream: Literal[False] = False,
        think: Optional[Union[bool, Literal["low", "medium", "high"]]] = None,
        format: Optional[Union[Literal["", "json"], JsonSchemaValue]] = None,
        options: Optional[Union[Mapping[str, Any], Options]] = None,
        keep_alive: KeepAlive = None,
        request_id: Optional[str] = None,
    ) -> ChatResponse:
        ...

    @overload
    def __call__(
        self,
        model="",
        messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
        *,
        tools: Optional[Sequence[ToolLike]] = None,
        stream: Literal[True] = True,
        think: Optional[Union[bool, Literal["low", "medium", "high"]]] = None,
        format: Optional[Union[Literal["", "json"], JsonSchemaValue]] = None,
        options: Optional[Union[Mapping[str, Any], Options]] = None,
        keep_alive: KeepAlive = None,
        request_id: Optional[str] = None,
    ) -> Iterator[ChatResponse]:
        ...

    def __call__(
        self,
        model="",
        messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
        *,
        tools: Optional[Sequence[ToolLike]] = None,
        stream: bool = False,
        think: Optional[Union[bool, Literal["low", "medium", "high"]]] = None,
        format: Optional[Union[Literal["", "json"], JsonSchemaValue]] = None,
        options: Optional[Union[Mapping[str, Any], Options]] = None,
        keep_alive: KeepAlive = None,
        request_id: Optional[str] = None,
    ) -> Union[ChatResponse, Iterator[ChatResponse]]:
        _ = request_id
        _ = keep_alive
        response = self._client.chat(
            model=self.ollama_model,
            messages=messages,
            tools=tools,
            stream=stream,
            think=think,
            format=format,
            options=options,
            keep_alive=_OLLAMA_KEEP_ALIVE_FOREVER,
        )

        return response
    def load(self, gpu_id: Union[str, Mapping[str, str]]) -> None:
        _ = gpu_id
        self._client.chat(
            model=self.ollama_model,
            messages=[{"role": "user", "content": "ping"}],
            stream=False,
            options=Options(num_predict=1),
            keep_alive=_OLLAMA_KEEP_ALIVE_FOREVER,
        )

    def unload(self) -> None:
        subprocess.run(["ollama", "stop", self.ollama_model], check=True)
