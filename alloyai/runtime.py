from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import torch

from .gpu_manager import GPUManager
from .model_registry import ModelRegistry
from .models.image.qwen_image import QwenImageModel
from .models.image.qwen_image_edit import QwenImageEditModel
from .models.image.z_image import ZImageModel
from .models.image.z_image_turbo import ZImageTurboModel
from .models.image.flux2dev import Flux2DevModel
from .models.image.flux2dev_turbo import Flux2DevTurboModel
from .models.image.flux2klein import Flux2KleinModel
# from .models.image.glm_image import GLMImageModel
from .models.audio.qwen_tts import (
    QwenTTSBaseModel,
    QwenTTSCustomVoiceModel,
    QwenTTSVoiceDesignModel,
)
from .models.text.ollama_models import (
    GptOSS120BModel,
    GptOSS20BModel,
    Qwen3LiteModel,
    Qwen3MediumModel,
)
from .request_scheduler import RequestScheduler

_BYTES_PER_MB = 1024 * 1024


def _gpu_id_to_index(gpu_id: str) -> int:
    if gpu_id.isdigit():
        return int(gpu_id)
    if gpu_id.startswith("cuda:"):
        tail = gpu_id.split("cuda:", 1)[1]
        if tail.isdigit():
            return int(tail)
    raise ValueError(f"Unsupported gpu_id format: {gpu_id}")


def detect_gpu_ids() -> List[str]:
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]


def build_gpu_manager(
    gpu_ids: Optional[Sequence[str]] = None,
    *,
    vram_cap_mb: Optional[Mapping[str, int]] = None,
) -> GPUManager:
    gpu_ids = list(gpu_ids) if gpu_ids is not None else detect_gpu_ids()
    if not gpu_ids:
        raise RuntimeError("No CUDA devices detected")

    try:
        return GPUManager.from_nvml(
            gpu_ids,
            vram_cap_mb=vram_cap_mb,
            use_system_vram=True,
        )
    except RuntimeError:
        pass

    totals: Dict[str, int] = {}
    for gpu_id in gpu_ids:
        index = _gpu_id_to_index(gpu_id)
        props = torch.cuda.get_device_properties(index)
        totals[gpu_id] = int(props.total_memory // _BYTES_PER_MB)

    caps: Dict[str, int] = {}
    for gpu_id, total_mb in totals.items():
        cap = total_mb
        if vram_cap_mb and gpu_id in vram_cap_mb:
            cap = min(int(vram_cap_mb[gpu_id]), total_mb)
        caps[gpu_id] = cap

    return GPUManager(
        caps,
        physical_vram_mb=totals,
        enable_nvml=False,
        use_system_vram=False,
    )


def register_default_models(registry: ModelRegistry) -> None:
    registry.register_factory(QwenImageModel.model_id, QwenImageModel)
    registry.register_factory(QwenImageEditModel.model_id, QwenImageEditModel)
    registry.register_factory(ZImageModel.model_id, ZImageModel)
    registry.register_factory(ZImageTurboModel.model_id, ZImageTurboModel)
    registry.register_factory(Flux2DevModel.model_id, Flux2DevModel)
    registry.register_factory(Flux2KleinModel.model_id, Flux2KleinModel)
    registry.register_factory(Flux2DevTurboModel.model_id, Flux2DevTurboModel)
    registry.register_factory(QwenTTSVoiceDesignModel.model_id, QwenTTSVoiceDesignModel)
    registry.register_factory(QwenTTSCustomVoiceModel.model_id, QwenTTSCustomVoiceModel)
    registry.register_factory(QwenTTSBaseModel.model_id, QwenTTSBaseModel)
    registry.register_factory(Qwen3MediumModel.model_id, Qwen3MediumModel)
    registry.register_factory(Qwen3LiteModel.model_id, Qwen3LiteModel)
    registry.register_factory(GptOSS120BModel.model_id, GptOSS120BModel)
    registry.register_factory(GptOSS20BModel.model_id, GptOSS20BModel)
    # registry.register_factory(GLMImageModel.model_id, GLMImageModel)

@dataclass(frozen=True)
class AlloyRuntime:
    gpu_manager: GPUManager
    registry: ModelRegistry
    scheduler: RequestScheduler


def build_runtime(
    gpu_ids: Optional[Sequence[str]] = None,
    *,
    vram_cap_mb: Optional[Mapping[str, int]] = None,
) -> AlloyRuntime:
    gpu_manager = build_gpu_manager(gpu_ids, vram_cap_mb=vram_cap_mb)
    registry = ModelRegistry(gpu_manager)
    register_default_models(registry)
    gpu_manager.attach_registry(registry)
    scheduler = RequestScheduler(registry, gpu_manager)
    return AlloyRuntime(gpu_manager=gpu_manager, registry=registry, scheduler=scheduler)


_runtime_lock = threading.Lock()
_runtime: Optional[AlloyRuntime] = None


def get_runtime() -> AlloyRuntime:
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = build_runtime()
    return _runtime
