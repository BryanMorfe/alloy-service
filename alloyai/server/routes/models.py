from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter

from ...gpu_manager import SnapshotScope
from ...models.model import Modality
from ...runtime import get_runtime
from ..schemas import (
    AlloyModelResponse,
    AlloyModelsResponse,
    ModelAllocationStatus,
    ModelCapabilityResponse,
)

router = APIRouter()


@router.get("/models", response_model=AlloyModelsResponse)
def get_models() -> AlloyModelsResponse:
    runtime = get_runtime()
    registry = runtime.registry
    manager = runtime.gpu_manager

    snapshot = manager.status_snapshot(scope=SnapshotScope.MANAGER)
    allocation_by_model_id = {
        record.model.model_id: record for record in snapshot.allocations
    }
    queue_count_by_model_id: dict[str, int] = defaultdict(int)
    for request in snapshot.queue:
        queue_count_by_model_id[request.model.model_id] += 1

    grouped: dict[Modality, list[AlloyModelResponse]] = {
        Modality.IMAGE: [],
        Modality.AUDIO: [],
        Modality.VIDEO: [],
        Modality.TEXT: [],
    }

    for model_id in registry.list_models():
        model = registry.get_model(model_id)
        allocation = allocation_by_model_id.get(model_id)
        queue_count = queue_count_by_model_id.get(model_id, 0)
        active_requests = (allocation.active_requests if allocation else 0) + queue_count

        if allocation is not None:
            status = ModelAllocationStatus.ALLOCATED
        elif queue_count > 0:
            status = ModelAllocationStatus.QUEUE
        else:
            status = ModelAllocationStatus.DEALLOCATED

        model_response = AlloyModelResponse(
            model_id=model_id,
            active_requests=active_requests,
            is_supported=manager.supports_model(model),
            supports_concurrent_requests=bool(
                getattr(model, "supports_concurrent_requests", False)
            ),
            capabilities=[
                ModelCapabilityResponse(
                    inputs=set(capability.inputs),
                    outputs=set(capability.outputs),
                    name=capability.name,
                )
                for capability in model.capabilities
            ],
            allocation_status=status,
        )

        output_modalities = {
            modality
            for capability in model.capabilities
            for modality in capability.outputs
        }
        for modality in output_modalities:
            if modality in grouped:
                grouped[modality].append(model_response)

    for models in grouped.values():
        models.sort(key=lambda item: item.model_id)

    return AlloyModelsResponse(
        image=grouped[Modality.IMAGE],
        audio=grouped[Modality.AUDIO],
        video=grouped[Modality.VIDEO],
        text=grouped[Modality.TEXT],
    )
