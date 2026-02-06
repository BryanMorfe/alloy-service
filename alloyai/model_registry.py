from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional

from .gpu_manager import AllocationResult, GPUManager, KeepAlive
from .models.model import Model

ModelFactory = Callable[[], Model]


class ModelRegistry:
    def __init__(self, gpu_manager: GPUManager) -> None:
        self._gpu_manager = gpu_manager
        self._models: Dict[str, Model] = {}
        self._factories: Dict[str, ModelFactory] = {}
        self._lock = threading.Lock()

    def register_model(self, model: Model, *, overwrite: bool = False) -> None:
        model_id = model.model_id
        with self._lock:
            if not overwrite and (model_id in self._models or model_id in self._factories):
                raise ValueError(f"model_id already registered: {model_id}")
            self._models[model_id] = model
            self._factories.pop(model_id, None)

    def register_factory(
        self,
        model_id: str,
        factory: ModelFactory,
        *,
        overwrite: bool = False,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must be non-empty")
        with self._lock:
            if not overwrite and (model_id in self._models or model_id in self._factories):
                raise ValueError(f"model_id already registered: {model_id}")
            self._factories[model_id] = factory
            self._models.pop(model_id, None)

    def unregister(self, model_id: str) -> bool:
        with self._lock:
            removed = False
            if model_id in self._models:
                del self._models[model_id]
                removed = True
            if model_id in self._factories:
                del self._factories[model_id]
                removed = True
            return removed

    def list_models(self) -> List[str]:
        with self._lock:
            return sorted(set(self._models) | set(self._factories))

    def get_model(self, model_id: str) -> Model:
        with self._lock:
            model = self._models.get(model_id)
            if model is not None:
                return model
            factory = self._factories.get(model_id)

        if factory is None:
            raise KeyError(f"model_id not registered: {model_id}")

        model = factory()
        if model.model_id != model_id:
            raise ValueError(
                f"factory returned model_id '{model.model_id}' for '{model_id}'"
            )

        with self._lock:
            existing = self._models.get(model_id)
            if existing is not None:
                return existing
            self._models[model_id] = model
        return model

    def request_model(
        self,
        model_id: str,
        *,
        priority: Optional[int] = None,
        keep_alive: KeepAlive = False,
    ) -> AllocationResult:
        model = self.get_model(model_id)
        return self._gpu_manager.request_allocation(
            model,
            priority=priority,
            keep_alive=keep_alive,
        )
