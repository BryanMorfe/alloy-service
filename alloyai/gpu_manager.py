from __future__ import annotations

import heapq
import itertools
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

from .models.model import Model

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore

_BYTES_PER_MB = 1024 * 1024


def _bytes_to_mb(value: int) -> int:
    return int(value // _BYTES_PER_MB)


def _normalize_gpu_id(gpu_id: object) -> str:
    return str(gpu_id)


def _resolve_nvml_index(gpu_id: str, nvml_index_map: Mapping[str, int]) -> Optional[int]:
    if gpu_id in nvml_index_map:
        return nvml_index_map[gpu_id]
    if gpu_id.isdigit():
        return int(gpu_id)
    if gpu_id.startswith("cuda:"):
        tail = gpu_id.split("cuda:", 1)[1]
        if tail.isdigit():
            return int(tail)
    return None


class ModelResolver(Protocol):
    def get_model(self, model_id: str) -> Model:
        ...


KeepAlive = Union[bool, int, float]


def _normalize_keep_alive(value: Optional[KeepAlive]) -> Optional[float]:
    if value is None or value is False:
        return None
    if value is True:
        return float("inf")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if value <= 0:
            return None
        return float(value)
    raise ValueError("keep_alive must be a bool or a number of seconds")


@dataclass(frozen=True)
class AllocationRequest:
    model: Model
    priority: int
    sequence: int
    requested_at: float
    keep_alive_s: Optional[float]


@dataclass
class AllocationRecord:
    model: Model
    vram_by_gpu_mb: Dict[str, int]
    gpu_assignment: Optional[Dict[str, str]]
    allocated_at: float
    last_used_at: float
    last_request_at: float
    active_requests: int
    keep_alive_s: Optional[float]


class AllocationStatus(str, Enum):
    ALLOCATED = "allocated"
    QUEUED = "queued"
    ALREADY_ALLOCATED = "already_allocated"
    ALREADY_QUEUED = "already_queued"
    INSUFFICIENT_VRAM = "insufficient_vram"


class SnapshotScope(str, Enum):
    MANAGER = "manager"
    SYSTEM = "system"


@dataclass(frozen=True)
class AllocationResult:
    status: AllocationStatus
    gpu_id: Optional[str] = None
    gpu_ids: Optional[List[str]] = None
    gpu_assignment: Optional[Dict[str, str]] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class GPUState:
    gpu_id: str
    total_vram_mb: int
    used_vram_mb: int
    free_vram_mb: int


@dataclass(frozen=True)
class GPUManagerSnapshot:
    gpus: List[GPUState]
    allocations: List[AllocationRecord]
    queue: List[AllocationRequest]
    usage_scope: SnapshotScope = SnapshotScope.MANAGER


class GPUManager:
    def __init__(
        self,
        gpu_vram_mb: Mapping[str, int],
        *,
        nvml_index_map: Optional[Mapping[str, int]] = None,
        enable_nvml: bool = True,
        physical_vram_mb: Optional[Mapping[str, int]] = None,
        use_system_vram: bool = True,
        keep_alive_poll_s: Optional[float] = 1.0,
    ) -> None:
        """
        gpu_vram_mb is a per-GPU VRAM cap in MB; if NVML is available, allocatable
        VRAM is min(cap, physical total).
        """
        if not gpu_vram_mb:
            raise ValueError("gpu_vram_mb must include at least one GPU")

        normalized_caps = {_normalize_gpu_id(k): int(v) for k, v in gpu_vram_mb.items()}
        invalid = [gpu_id for gpu_id, vram in normalized_caps.items() if vram <= 0]
        if invalid:
            raise ValueError(f"gpu_vram_mb has non-positive entries: {', '.join(invalid)}")

        self._nvml_enabled = enable_nvml
        self._nvml_index_map: Dict[str, int] = dict(nvml_index_map or {})
        self._pynvml: Optional[Any] = None
        self._use_system_vram = use_system_vram

        if physical_vram_mb is None and self._nvml_enabled:
            physical_vram_mb = self._read_physical_vram_mb(list(normalized_caps.keys()))

        normalized_physical = (
            {_normalize_gpu_id(k): int(v) for k, v in physical_vram_mb.items()}
            if physical_vram_mb
            else {}
        )

        self._physical_total_vram_mb: Dict[str, int] = normalized_physical
        self._total_vram_mb = self._apply_vram_caps(normalized_caps, normalized_physical)
        self._used_vram_mb = {gpu_id: 0 for gpu_id in self._total_vram_mb}
        self._allocations: Dict[str, AllocationRecord] = {}
        self._queue: List[Tuple[int, int, AllocationRequest]] = []
        self._queued_ids: set[str] = set()
        self._sequence = itertools.count()
        self._lock = threading.Lock()
        self._model_registry: Optional[ModelResolver] = None
        self._keep_alive_poll_s = keep_alive_poll_s if keep_alive_poll_s and keep_alive_poll_s > 0 else None
        self._keep_alive_stop = threading.Event()
        self._keep_alive_thread: Optional[threading.Thread] = None
        if self._keep_alive_poll_s is not None:
            self._keep_alive_thread = threading.Thread(
                target=self._keep_alive_loop,
                name="gpu-manager-keep-alive",
                daemon=True,
            )
            self._keep_alive_thread.start()

    @classmethod
    def from_nvml(
        cls,
        gpu_ids: Sequence[str],
        *,
        vram_cap_mb: Optional[Mapping[str, int]] = None,
        nvml_index_map: Optional[Mapping[str, int]] = None,
        use_system_vram: bool = True,
        keep_alive_poll_s: Optional[float] = 1.0,
    ) -> "GPUManager":
        """Build a manager from NVML totals with optional per-GPU caps."""
        normalized_ids = [_normalize_gpu_id(gpu_id) for gpu_id in gpu_ids]
        if not normalized_ids:
            raise ValueError("gpu_ids must include at least one GPU")

        nvml_index_map = dict(nvml_index_map or {})
        if pynvml is None:
            raise RuntimeError("pynvml is required to build from NVML")

        pynvml.nvmlInit()
        physical: Dict[str, int] = {}
        for gpu_id in normalized_ids:
            nvml_index = _resolve_nvml_index(gpu_id, nvml_index_map)
            if nvml_index is None:
                raise ValueError(
                    f"Unable to resolve NVML index for gpu_id '{gpu_id}'. "
                    "Provide nvml_index_map or use numeric GPU ids."
                )
            handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            physical[gpu_id] = _bytes_to_mb(info.total)

        caps: Dict[str, int] = {}
        for gpu_id in normalized_ids:
            cap = physical[gpu_id]
            if vram_cap_mb and gpu_id in vram_cap_mb:
                cap = int(vram_cap_mb[gpu_id])
            caps[gpu_id] = min(cap, physical[gpu_id])

        return cls(
            caps,
            physical_vram_mb=physical,
            nvml_index_map=nvml_index_map,
            enable_nvml=True,
            use_system_vram=use_system_vram,
            keep_alive_poll_s=keep_alive_poll_s,
        )

    def _keep_alive_loop(self) -> None:
        assert self._keep_alive_poll_s is not None
        while not self._keep_alive_stop.is_set():
            time.sleep(self._keep_alive_poll_s)
            try:
                self._expire_keep_alive()
            except Exception:
                continue

    def _expire_keep_alive(self) -> None:
        with self._lock:
            now = time.time()
            expired: List[AllocationRecord] = []
            for record in self._allocations.values():
                if record.active_requests > 0:
                    continue
                keep_alive_s = record.keep_alive_s
                if keep_alive_s is None or keep_alive_s == float("inf"):
                    continue
                if now - record.last_used_at >= keep_alive_s:
                    expired.append(record)

            if not expired:
                return

            for record in expired:
                self._deallocate_record(record)

            self._process_queue_locked()

    def request_allocation(
        self,
        model: Model,
        priority: Optional[int] = None,
        keep_alive: KeepAlive = False,
    ) -> AllocationResult:
        with self._lock:
            if model.model_id in self._allocations:
                record = self._allocations[model.model_id]
                record.keep_alive_s = _normalize_keep_alive(keep_alive)
                gpu_ids = sorted(record.vram_by_gpu_mb.keys())
                primary_gpu_id = gpu_ids[0] if gpu_ids else None
                if bool(getattr(model, "spreadable", False)) and len(gpu_ids) > 1:
                    primary_gpu_id = "auto"
                return AllocationResult(
                    AllocationStatus.ALREADY_ALLOCATED,
                    gpu_id=primary_gpu_id,
                    gpu_ids=gpu_ids,
                    gpu_assignment=record.gpu_assignment,
                )
            if model.model_id in self._queued_ids:
                self._process_queue_locked()
                record = self._allocations.get(model.model_id)
                if record is not None:
                    gpu_ids = sorted(record.vram_by_gpu_mb.keys())
                    primary_gpu_id = gpu_ids[0] if gpu_ids else None
                    if bool(getattr(model, "spreadable", False)) and len(gpu_ids) > 1:
                        primary_gpu_id = "auto"
                    return AllocationResult(
                        AllocationStatus.ALLOCATED,
                        gpu_id=primary_gpu_id,
                        gpu_ids=gpu_ids,
                        gpu_assignment=record.gpu_assignment,
                    )
                return AllocationResult(AllocationStatus.ALREADY_QUEUED)

            keep_alive_s = _normalize_keep_alive(keep_alive)
            system_free_by_gpu = self._system_free_vram_mb()
            allocation = self._try_allocate_locked(
                model, keep_alive_s, system_free_by_gpu
            )
            if allocation is not None:
                return allocation

            effective_priority = priority if priority is not None else model.priority
            sequence = next(self._sequence)
            request = AllocationRequest(
                model=model,
                priority=effective_priority,
                sequence=sequence,
                requested_at=time.time(),
                keep_alive_s=keep_alive_s,
            )
            heapq.heappush(self._queue, (-effective_priority, sequence, request))
            self._queued_ids.add(model.model_id)
            return AllocationResult(AllocationStatus.QUEUED)

    def attach_registry(self, registry: ModelResolver) -> None:
        self._model_registry = registry

    def request_model(
        self,
        model_id: str,
        *,
        priority: Optional[int] = None,
        keep_alive: KeepAlive = False,
    ) -> AllocationResult:
        if self._model_registry is None:
            raise RuntimeError("No model registry attached")
        model = self._model_registry.get_model(model_id)
        return self.request_allocation(model, priority=priority, keep_alive=keep_alive)

    def begin_request(self, model_id: str) -> None:
        with self._lock:
            record = self._allocations.get(model_id)
            if record is None:
                raise KeyError(f"model_id not allocated: {model_id}")
            now = time.time()
            record.active_requests += 1
            record.last_request_at = now
            record.last_used_at = now

    def end_request(self, model_id: str) -> None:
        with self._lock:
            record = self._allocations.get(model_id)
            if record is None:
                return
            if record.active_requests > 0:
                record.active_requests -= 1
            record.last_used_at = time.time()
            if record.active_requests == 0 and record.keep_alive_s is None:
                self._deallocate_record(record)
                self._process_queue_locked()

    def release_model(self, model_id: str) -> bool:
        with self._lock:
            record = self._allocations.get(model_id)
            if record is None:
                return False

            self._deallocate_record(record)
            self._process_queue_locked()
            return True

    def process_queue(self) -> List[AllocationResult]:
        with self._lock:
            return self._process_queue_locked()

    def status_snapshot(self, scope: SnapshotScope = SnapshotScope.MANAGER) -> GPUManagerSnapshot:
        with self._lock:
            if scope == SnapshotScope.MANAGER:
                gpus = [
                    GPUState(
                        gpu_id=gpu_id,
                        total_vram_mb=total,
                        used_vram_mb=self._used_vram_mb[gpu_id],
                        free_vram_mb=total - self._used_vram_mb[gpu_id],
                    )
                    for gpu_id, total in self._total_vram_mb.items()
                ]
            else:
                gpus = self._system_gpu_states()

            allocations = list(self._allocations.values())
            queue = [item[2] for item in sorted(self._queue)]
            return GPUManagerSnapshot(
                gpus=gpus,
                allocations=allocations,
                queue=queue,
                usage_scope=scope,
            )

    def _normalize_required_vram(self, required_vram_mb: Any) -> Union[int, Dict[str, int]]:
        if isinstance(required_vram_mb, Mapping):
            if not required_vram_mb:
                raise ValueError("required_vram_mb must be a non-empty mapping")
            normalized: Dict[str, int] = {}
            for part, value in required_vram_mb.items():
                if not isinstance(part, str) or not part:
                    raise ValueError("required_vram_mb keys must be non-empty strings")
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("required_vram_mb values must be numbers")
                vram_mb = int(value)
                if vram_mb <= 0:
                    raise ValueError("required_vram_mb values must be positive")
                normalized[part] = vram_mb
            return normalized

        if isinstance(required_vram_mb, bool) or not isinstance(required_vram_mb, (int, float)):
            raise ValueError("required_vram_mb must be an int or mapping of part->MB")
        vram_mb = int(required_vram_mb)
        if vram_mb <= 0:
            raise ValueError("required_vram_mb must be positive")
        return vram_mb

    def _active_requests_by_gpu(self) -> Dict[str, int]:
        active: Dict[str, int] = {gpu_id: 0 for gpu_id in self._total_vram_mb}
        for record in self._allocations.values():
            for gpu_id in record.vram_by_gpu_mb:
                active[gpu_id] = active.get(gpu_id, 0) + record.active_requests
        return active

    def _try_allocate_locked(
        self,
        model: Model,
        keep_alive_s: Optional[float],
        system_free_by_gpu: Optional[Dict[str, int]],
    ) -> Optional[AllocationResult]:
        try:
            required = self._normalize_required_vram(model.required_vram_mb)
        except ValueError as exc:
            return AllocationResult(AllocationStatus.INSUFFICIENT_VRAM, reason=str(exc))

        if isinstance(required, dict):
            return self._try_allocate_discrete(
                model, required, keep_alive_s, system_free_by_gpu
            )

        if bool(getattr(model, "spreadable", False)):
            return self._try_allocate_spreadable(
                model, required, keep_alive_s, system_free_by_gpu
            )

        return self._try_allocate_single(
            model, required, keep_alive_s, system_free_by_gpu
        )

    def _try_allocate_single(
        self,
        model: Model,
        required_vram_mb: int,
        keep_alive_s: Optional[float],
        system_free_by_gpu: Optional[Dict[str, int]],
    ) -> Optional[AllocationResult]:
        if required_vram_mb > max(self._total_vram_mb.values()):
            return AllocationResult(
                AllocationStatus.INSUFFICIENT_VRAM,
                reason="no GPU has enough total VRAM for this model",
            )

        gpu_id, needs_eviction = self._select_gpu_for_allocation(
            required_vram_mb, system_free_by_gpu
        )
        if gpu_id is None:
            return None

        if needs_eviction:
            self._evict_for_allocation(gpu_id, required_vram_mb, system_free_by_gpu)
        if self._available_vram_mb(gpu_id, system_free_by_gpu) < required_vram_mb:
            return None

        return self._allocate_model(
            model,
            vram_by_gpu_mb={gpu_id: required_vram_mb},
            gpu_assignment=None,
            keep_alive_s=keep_alive_s,
            load_arg=gpu_id,
            primary_gpu_id=gpu_id,
        )

    def _try_allocate_discrete(
        self,
        model: Model,
        required_by_part: Dict[str, int],
        keep_alive_s: Optional[float],
        system_free_by_gpu: Optional[Dict[str, int]],
    ) -> Optional[AllocationResult]:
        max_per_gpu = max(self._total_vram_mb.values())
        for part, vram_mb in required_by_part.items():
            if vram_mb > max_per_gpu:
                return AllocationResult(
                    AllocationStatus.INSUFFICIENT_VRAM,
                    reason=f"part '{part}' exceeds max GPU VRAM",
                )

        assignment = self._select_gpus_for_parts(
            required_by_part, system_free_by_gpu
        )
        if assignment is None:
            return None

        required_by_gpu: Dict[str, int] = {}
        for part, vram_mb in required_by_part.items():
            gpu_id = assignment[part]
            required_by_gpu[gpu_id] = required_by_gpu.get(gpu_id, 0) + vram_mb

        for gpu_id, vram_mb in required_by_gpu.items():
            if self._available_vram_mb(gpu_id, system_free_by_gpu) < vram_mb:
                self._evict_for_allocation(gpu_id, vram_mb, system_free_by_gpu)
            if self._available_vram_mb(gpu_id, system_free_by_gpu) < vram_mb:
                return None

        primary_gpu_id = (
            next(iter(required_by_gpu)) if len(required_by_gpu) == 1 else None
        )
        return self._allocate_model(
            model,
            vram_by_gpu_mb=required_by_gpu,
            gpu_assignment=assignment,
            keep_alive_s=keep_alive_s,
            load_arg=assignment,
            primary_gpu_id=primary_gpu_id,
        )

    def _try_allocate_spreadable(
        self,
        model: Model,
        required_vram_mb: int,
        keep_alive_s: Optional[float],
        system_free_by_gpu: Optional[Dict[str, int]],
    ) -> Optional[AllocationResult]:
        total_capacity = sum(self._total_vram_mb.values())
        if required_vram_mb > total_capacity:
            return AllocationResult(
                AllocationStatus.INSUFFICIENT_VRAM,
                reason="total VRAM across GPUs is insufficient",
            )

        available_by_gpu = {
            gpu_id: self._available_vram_mb(gpu_id, system_free_by_gpu)
            + self._evictable_vram_mb(gpu_id)
            for gpu_id in self._total_vram_mb
        }
        total_available = sum(available_by_gpu.values())
        if total_available < required_vram_mb:
            return None

        active_by_gpu = self._active_requests_by_gpu()
        order = sorted(
            available_by_gpu.items(),
            key=lambda item: (-item[1], active_by_gpu.get(item[0], 0)),
        )

        remaining = required_vram_mb
        required_by_gpu: Dict[str, int] = {}
        for gpu_id, available in order:
            if remaining <= 0:
                break
            take = min(available, remaining)
            if take <= 0:
                continue
            required_by_gpu[gpu_id] = take
            remaining -= take

        if remaining > 0:
            return None

        for gpu_id, vram_mb in required_by_gpu.items():
            if self._available_vram_mb(gpu_id, system_free_by_gpu) < vram_mb:
                self._evict_for_allocation(gpu_id, vram_mb, system_free_by_gpu)
            if self._available_vram_mb(gpu_id, system_free_by_gpu) < vram_mb:
                return None

        return self._allocate_model(
            model,
            vram_by_gpu_mb=required_by_gpu,
            gpu_assignment=None,
            keep_alive_s=keep_alive_s,
            load_arg="auto",
            primary_gpu_id="auto",
        )

    def _select_gpus_for_parts(
        self,
        required_by_part: Dict[str, int],
        system_free_by_gpu: Optional[Dict[str, int]],
    ) -> Optional[Dict[str, str]]:
        available_by_gpu: Dict[str, int] = {}
        for gpu_id in self._total_vram_mb:
            available_by_gpu[gpu_id] = (
                self._available_vram_mb(gpu_id, system_free_by_gpu)
                + self._evictable_vram_mb(gpu_id)
            )

        active_by_gpu = self._active_requests_by_gpu()
        remaining = dict(available_by_gpu)
        assignment: Dict[str, str] = {}

        for part, vram_mb in sorted(required_by_part.items(), key=lambda item: item[1], reverse=True):
            candidates = []
            for gpu_id, available in remaining.items():
                if available < vram_mb:
                    continue
                free_after = available - vram_mb
                candidates.append((active_by_gpu.get(gpu_id, 0), free_after, gpu_id))
            if not candidates:
                return None
            _, _, chosen_gpu = min(candidates)
            assignment[part] = chosen_gpu
            remaining[chosen_gpu] -= vram_mb

        return assignment

    def _allocate_model(
        self,
        model: Model,
        vram_by_gpu_mb: Dict[str, int],
        gpu_assignment: Optional[Dict[str, str]],
        keep_alive_s: Optional[float],
        load_arg: Union[str, Dict[str, str]],
        primary_gpu_id: Optional[str],
    ) -> AllocationResult:
        now = time.time()
        for gpu_id, vram_mb in vram_by_gpu_mb.items():
            self._used_vram_mb[gpu_id] += vram_mb

        record = AllocationRecord(
            model=model,
            vram_by_gpu_mb=vram_by_gpu_mb,
            gpu_assignment=gpu_assignment,
            allocated_at=now,
            last_used_at=now,
            last_request_at=now,
            active_requests=0,
            keep_alive_s=keep_alive_s,
        )
        self._allocations[model.model_id] = record

        try:
            model.load(load_arg)
        except Exception:
            for gpu_id, vram_mb in vram_by_gpu_mb.items():
                self._used_vram_mb[gpu_id] -= vram_mb
            self._allocations.pop(model.model_id, None)
            raise

        gpu_ids = sorted(vram_by_gpu_mb.keys())
        if primary_gpu_id is None:
            if isinstance(load_arg, str) and load_arg == "auto":
                primary_gpu_id = "auto"
            else:
                primary_gpu_id = gpu_ids[0] if gpu_ids else None

        return AllocationResult(
            AllocationStatus.ALLOCATED,
            gpu_id=primary_gpu_id,
            gpu_ids=gpu_ids,
            gpu_assignment=gpu_assignment,
        )

    def _process_queue_locked(self) -> List[AllocationResult]:
        results: List[AllocationResult] = []
        system_free_by_gpu = self._system_free_vram_mb()
        while self._queue:
            _, _, request = self._queue[0]
            allocation = self._try_allocate_locked(
                request.model, request.keep_alive_s, system_free_by_gpu
            )
            if allocation is None:
                break
            heapq.heappop(self._queue)
            self._queued_ids.discard(request.model.model_id)
            results.append(allocation)
        return results

    def _available_vram_mb(
        self,
        gpu_id: str,
        system_free_by_gpu: Optional[Mapping[str, int]] = None,
    ) -> int:
        total = self._total_vram_mb[gpu_id]
        manager_free = total - self._used_vram_mb[gpu_id]
        if system_free_by_gpu and gpu_id in system_free_by_gpu:
            system_free = system_free_by_gpu[gpu_id]
            return max(min(manager_free, system_free), 0)
        return max(manager_free, 0)

    def _evictable_vram_mb(self, gpu_id: str) -> int:
        evictable = 0
        for record in self._allocations.values():
            if record.active_requests > 0:
                continue
            vram_mb = record.vram_by_gpu_mb.get(gpu_id)
            if vram_mb:
                evictable += vram_mb
        return evictable

    def _select_gpu_for_allocation(
        self,
        required_vram_mb: int,
        system_free_by_gpu: Optional[Mapping[str, int]] = None,
    ) -> Tuple[Optional[str], bool]:
        best_gpu_id: Optional[str] = None
        best_free_after: Optional[int] = None
        best_needs_eviction = False
        best_active_requests: Optional[int] = None

        active_requests_by_gpu: Dict[str, int] = {gpu_id: 0 for gpu_id in self._total_vram_mb}
        for record in self._allocations.values():
            for gpu_id in record.vram_by_gpu_mb:
                active_requests_by_gpu[gpu_id] = (
                    active_requests_by_gpu.get(gpu_id, 0) + record.active_requests
                )

        for gpu_id in self._total_vram_mb:
            free = self._available_vram_mb(gpu_id, system_free_by_gpu)
            evictable = self._evictable_vram_mb(gpu_id)
            needs_eviction = False

            if free >= required_vram_mb:
                free_after = free - required_vram_mb
            elif free + evictable >= required_vram_mb:
                free_after = free + evictable - required_vram_mb
                needs_eviction = True
            else:
                continue

            active_requests = active_requests_by_gpu.get(gpu_id, 0)
            candidate_key = (active_requests, needs_eviction, free_after)
            best_key = (
                (best_active_requests, best_needs_eviction, best_free_after)
                if best_free_after is not None and best_active_requests is not None
                else None
            )
            if best_key is None or candidate_key < best_key:
                best_gpu_id = gpu_id
                best_needs_eviction = needs_eviction
                best_free_after = free_after
                best_active_requests = active_requests

        return best_gpu_id, best_needs_eviction

    def _apply_vram_caps(
        self,
        caps: Mapping[str, int],
        physical: Mapping[str, int],
    ) -> Dict[str, int]:
        allocatable: Dict[str, int] = {}
        for gpu_id, cap in caps.items():
            physical_total = physical.get(gpu_id)
            allocatable[gpu_id] = (
                min(cap, physical_total) if physical_total is not None else cap
            )
        return allocatable

    def _get_nvml(self) -> Optional[Any]:
        if not self._nvml_enabled or pynvml is None:
            return None
        if self._pynvml is not None:
            return self._pynvml
        try:
            pynvml.nvmlInit()
        except Exception:
            return None
        self._pynvml = pynvml
        return self._pynvml

    def _read_physical_vram_mb(self, gpu_ids: Sequence[str]) -> Dict[str, int]:
        nvml = self._get_nvml()
        if nvml is None:
            return {}
        physical: Dict[str, int] = {}
        for gpu_id in gpu_ids:
            nvml_index = _resolve_nvml_index(gpu_id, self._nvml_index_map)
            if nvml_index is None:
                continue
            handle = nvml.nvmlDeviceGetHandleByIndex(nvml_index)
            info = nvml.nvmlDeviceGetMemoryInfo(handle)
            physical[gpu_id] = _bytes_to_mb(info.total)
        return physical

    def _system_free_vram_mb(self) -> Optional[Dict[str, int]]:
        if not self._use_system_vram:
            return None
        nvml = self._get_nvml()
        if nvml is None:
            return None
        free_by_gpu: Dict[str, int] = {}
        for gpu_id in self._total_vram_mb:
            nvml_index = _resolve_nvml_index(gpu_id, self._nvml_index_map)
            if nvml_index is None:
                continue
            handle = nvml.nvmlDeviceGetHandleByIndex(nvml_index)
            info = nvml.nvmlDeviceGetMemoryInfo(handle)
            free_by_gpu[gpu_id] = _bytes_to_mb(info.free)
        return free_by_gpu

    def _system_gpu_states(self) -> List[GPUState]:
        nvml = self._get_nvml()
        if nvml is None:
            raise RuntimeError("pynvml is required for system-wide GPU usage snapshots")

        gpus: List[GPUState] = []
        for gpu_id in self._total_vram_mb:
            nvml_index = _resolve_nvml_index(gpu_id, self._nvml_index_map)
            if nvml_index is None:
                raise ValueError(
                    f"Unable to resolve NVML index for gpu_id '{gpu_id}'. "
                    "Provide nvml_index_map or use numeric GPU ids."
                )
            handle = nvml.nvmlDeviceGetHandleByIndex(nvml_index)
            info = nvml.nvmlDeviceGetMemoryInfo(handle)
            total = _bytes_to_mb(info.total)
            used = _bytes_to_mb(info.used)
            gpus.append(
                GPUState(
                    gpu_id=gpu_id,
                    total_vram_mb=total,
                    used_vram_mb=used,
                    free_vram_mb=max(total - used, 0),
                )
            )
        return gpus

    def _keep_alive_rank(self, record: AllocationRecord, now: float) -> int:
        if record.keep_alive_s is None:
            return 0
        if record.keep_alive_s == float("inf"):
            return 2
        if now - record.last_used_at >= record.keep_alive_s:
            return 1
        return 2

    def _eviction_candidates(
        self,
        gpu_id: str,
        now: float,
        *,
        ignore_keep_alive: bool,
    ) -> List[AllocationRecord]:
        candidates: List[Tuple[int, float, AllocationRecord]] = []
        for record in self._allocations.values():
            if gpu_id not in record.vram_by_gpu_mb:
                continue
            if record.active_requests > 0:
                continue
            if ignore_keep_alive:
                candidates.append((0, record.last_used_at, record))
            else:
                rank = self._keep_alive_rank(record, now)
                candidates.append((rank, record.last_used_at, record))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [record for _, _, record in candidates]

    def _evict_for_allocation(
        self,
        gpu_id: str,
        required_vram_mb: int,
        system_free_by_gpu: Optional[Dict[str, int]] = None,
    ) -> bool:
        now = time.time()
        for record in self._eviction_candidates(
            gpu_id, now, ignore_keep_alive=True
        ):
            if self._available_vram_mb(gpu_id, system_free_by_gpu) >= required_vram_mb:
                return True
            self._deallocate_record(record)
            if system_free_by_gpu is not None:
                for freed_gpu_id, freed_mb in record.vram_by_gpu_mb.items():
                    if freed_gpu_id in system_free_by_gpu:
                        system_free_by_gpu[freed_gpu_id] += freed_mb
        return self._available_vram_mb(gpu_id, system_free_by_gpu) >= required_vram_mb

    def _deallocate_record(self, record: AllocationRecord) -> None:
        record.model.unload()
        for gpu_id, vram_mb in record.vram_by_gpu_mb.items():
            self._used_vram_mb[gpu_id] -= vram_mb
        self._allocations.pop(record.model.model_id, None)
