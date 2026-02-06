from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Set, Union, runtime_checkable


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


Subscriber = queue.Queue["ModelEvent"]


@dataclass(frozen=True)
class ModelEvent:
    event: str
    payload: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@runtime_checkable
class Publisher(Protocol):
    def subscribe(self) -> Subscriber:
        """Subscribe to events from the model."""

    def unsubscribe(self, subscriber: Subscriber) -> None:
        """Stop receiving events for a subscriber."""

    def publish(self, event: ModelEvent) -> None:
        """Publish an event to all subscribers."""


class EventPublisher:
    def __init__(self) -> None:
        self._subscribers: set[Subscriber] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> Subscriber:
        subscriber: Subscriber = queue.Queue()
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: Subscriber) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    def publish(self, event: ModelEvent) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.put(event)


@dataclass(frozen=True)
class ModelCapability:
    inputs: Set[Modality]
    outputs: Set[Modality]
    name: Optional[str] = None


class CompositionMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


@runtime_checkable
class Model(Publisher, Protocol):
    """Base model interface; required_vram_mb is the peak VRAM needed when resident."""
    model_id: str
    required_vram_mb: Union[int, Mapping[str, int]]
    priority: int
    capabilities: Sequence[ModelCapability]
    spreadable: bool

    def __call__(self, **kwargs: Any) -> Any:
        """Run inference or generation for this model."""

    def load(self, gpu_id: Union[str, Mapping[str, str]]) -> None:
        """Load the model onto the given GPU(s)."""

    def unload(self) -> None:
        """Unload the model from its current device."""


@dataclass(frozen=True)
class CompositionPhase:
    models: Sequence[Model]
    name: Optional[str] = None


@runtime_checkable
class CompositeModel(Model, Protocol):
    composition: CompositionMode
    components: Sequence[Model]

    def plan(self, **kwargs: Any) -> Sequence[CompositionPhase]:
        """
        Return ordered phases of model usage; models listed in the same phase may
        run in parallel and should be considered resident together.
        """
