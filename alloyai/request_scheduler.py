from __future__ import annotations

import asyncio
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Dict, Optional

from .gpu_manager import AllocationResult, AllocationStatus, GPUManager, KeepAlive
from .model_registry import ModelRegistry
from .models.model import Model, ModelEvent


class AllocationError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


CompletedPayloadBuilder = Callable[[Any, AllocationResult, int], Dict[str, Any]]
EventPublisher = Callable[[ModelEvent], None]


class RequestScheduler:
    def __init__(
        self,
        registry: ModelRegistry,
        gpu_manager: GPUManager,
        *,
        concurrency_limit: int = 4,
    ) -> None:
        if concurrency_limit <= 0:
            raise ValueError("concurrency_limit must be positive")
        self._registry = registry
        self._gpu_manager = gpu_manager
        self._concurrency_limit = concurrency_limit
        self._map_lock = threading.Lock()
        self._locks: Dict[str, asyncio.Lock] = {}
        self._semaphores: Dict[str, tuple[int, asyncio.Semaphore]] = {}
        self._inflight: set[str] = set()
        self._inflight_lock = threading.Lock()

    def _get_model_lock(self, model_id: str) -> asyncio.Lock:
        with self._map_lock:
            lock = self._locks.get(model_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[model_id] = lock
            return lock

    def _get_model_semaphore(self, model: Model) -> asyncio.Semaphore:
        limit = self._concurrency_limit
        if not bool(getattr(model, "supports_concurrent_requests", False)):
            limit = 1
        with self._map_lock:
            entry = self._semaphores.get(model.model_id)
            if entry is not None and entry[0] == limit:
                return entry[1]
            semaphore = asyncio.Semaphore(limit)
            self._semaphores[model.model_id] = (limit, semaphore)
            return semaphore

    @asynccontextmanager
    async def _acquire_slot(self, semaphore: asyncio.Semaphore) -> Any:
        await semaphore.acquire()
        try:
            yield
        finally:
            semaphore.release()

    @asynccontextmanager
    async def _maybe_lock(self, lock: asyncio.Lock, enabled: bool) -> Any:
        if enabled:
            async with lock:
                yield
        else:
            yield

    @asynccontextmanager
    async def _track_request(self, request_id: str) -> Any:
        with self._inflight_lock:
            if request_id in self._inflight:
                raise AllocationError(409, f"Duplicate request_id: {request_id}")
            self._inflight.add(request_id)
        try:
            yield
        finally:
            with self._inflight_lock:
                self._inflight.discard(request_id)

    async def _wait_for_allocation(
        self,
        model_id: str,
        *,
        priority: Optional[int],
        keep_alive: KeepAlive,
        timeout_s: float,
        poll_s: float,
        request_id: Optional[str],
        on_event: Optional[EventPublisher],
    ) -> AllocationResult:
        deadline = time.monotonic() + timeout_s
        alloc = self._registry.request_model(
            model_id, priority=priority, keep_alive=keep_alive
        )
        last_status: Optional[AllocationStatus] = None

        while alloc.status in (AllocationStatus.QUEUED, AllocationStatus.ALREADY_QUEUED):
            if time.monotonic() >= deadline:
                raise AllocationError(503, "Allocation timed out")
            if (
                on_event is not None
                and request_id is not None
                and alloc.status != last_status
            ):
                on_event(
                    ModelEvent(
                        event="queued",
                        request_id=request_id,
                        payload={"status": alloc.status.value},
                    )
                )
                last_status = alloc.status
            await asyncio.sleep(poll_s)
            alloc = self._registry.request_model(
                model_id, priority=priority, keep_alive=keep_alive
            )

        if alloc.status == AllocationStatus.INSUFFICIENT_VRAM:
            raise AllocationError(400, alloc.reason or "Insufficient VRAM")

        return alloc

    async def run(
        self,
        model_id: str,
        payload: Dict[str, Any],
        *,
        priority: Optional[int] = None,
        keep_alive: KeepAlive = False,
        timeout_s: float = 300,
        poll_s: float = 0.5,
        request_id: Optional[str] = None,
        on_event: Optional[EventPublisher] = None,
        completed_payload_builder: Optional[CompletedPayloadBuilder] = None,
    ) -> tuple[AllocationResult, Any, int]:
        model = self._registry.get_model(model_id)
        if request_id is None:
            request_id = uuid.uuid4().hex

        if on_event is not None:
            on_event(
                ModelEvent(
                    event="received",
                    request_id=request_id,
                    payload={"model_id": model_id},
                )
            )

        async with self._track_request(request_id):
            alloc = await self._wait_for_allocation(
                model_id,
                priority=priority,
                keep_alive=keep_alive,
                timeout_s=timeout_s,
                poll_s=poll_s,
                request_id=request_id,
                on_event=on_event,
            )

            if on_event is not None:
                on_event(
                    ModelEvent(
                        event="allocated",
                        request_id=request_id,
                        payload={
                            "gpu_id": alloc.gpu_id,
                            "gpu_ids": alloc.gpu_ids,
                            "gpu_assignment": alloc.gpu_assignment,
                            "status": alloc.status.value,
                        },
                    )
                )

            model_lock = self._get_model_lock(model_id)
            model_semaphore = self._get_model_semaphore(model)
            use_lock = not bool(getattr(model, "supports_concurrent_requests", False))

            if on_event is not None and model_semaphore.locked():
                on_event(
                    ModelEvent(
                        event="queued",
                        request_id=request_id,
                        payload={"status": "concurrency_limit"},
                    )
                )

            async with self._acquire_slot(model_semaphore):
                async with self._maybe_lock(model_lock, use_lock):
                    self._gpu_manager.begin_request(model_id)
                    try:
                        started = time.monotonic()
                        model_payload = dict(payload)
                        model_payload["request_id"] = request_id
                        output = await asyncio.to_thread(model, **model_payload)
                        duration_ms = int((time.monotonic() - started) * 1000)
                    finally:
                        self._gpu_manager.end_request(model_id)

        if on_event is not None and completed_payload_builder is not None:
            on_event(
                ModelEvent(
                    event="completed",
                    request_id=request_id,
                    payload=completed_payload_builder(output, alloc, duration_ms),
                )
            )

        return alloc, output, duration_ms

    async def stream(
        self,
        model_id: str,
        payload: Dict[str, Any],
        *,
        priority: Optional[int] = None,
        keep_alive: KeepAlive = False,
        timeout_s: float = 300,
        poll_s: float = 0.5,
        completed_payload_builder: Optional[CompletedPayloadBuilder] = None,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[ModelEvent]:
        model = self._registry.get_model(model_id)
        if request_id is None:
            request_id = uuid.uuid4().hex
        subscriber = model.subscribe()

        async def run_inference() -> None:
            try:
                await self.run(
                    model_id,
                    payload,
                    priority=priority,
                    keep_alive=keep_alive,
                    timeout_s=timeout_s,
                    poll_s=poll_s,
                    request_id=request_id,
                    on_event=model.publish,
                    completed_payload_builder=completed_payload_builder,
                )
            except AllocationError as exc:
                model.publish(
                    ModelEvent(
                        event="error",
                        request_id=request_id,
                        payload={"message": str(exc)},
                    )
                )
            except Exception as exc:
                model.publish(
                    ModelEvent(
                        event="error",
                        request_id=request_id,
                        payload={"message": str(exc)},
                    )
                )
            finally:
                model.publish(ModelEvent(event="done", request_id=request_id))

        task = asyncio.create_task(run_inference())

        try:
            while True:
                event = await asyncio.to_thread(subscriber.get)
                if event.request_id and event.request_id != request_id:
                    continue
                yield event
                if event.event in ("done", "error", "completed"):
                    break
        finally:
            model.unsubscribe(subscriber)
            await task
