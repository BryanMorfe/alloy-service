from .model import (
    CompositeModel,
    CompositionMode,
    CompositionPhase,
    EventPublisher,
    Modality,
    Model,
    ModelCapability,
    ModelEvent,
    Publisher,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from .diffusion_model import DiffusionModel
    from .diffusion_pipeline_allocator import DiffusionPipelineAllocator

__all__ = [
    "CompositeModel",
    "CompositionMode",
    "CompositionPhase",
    "EventPublisher",
    "Modality",
    "Model",
    "ModelCapability",
    "ModelEvent",
    "Publisher",
]


def __getattr__(name: str):
    if name == "DiffusionModel":
        from .diffusion_model import DiffusionModel

        return DiffusionModel
    if name == "DiffusionPipelineAllocator":
        from .diffusion_pipeline_allocator import DiffusionPipelineAllocator

        return DiffusionPipelineAllocator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
