from __future__ import annotations

from enum import Enum
from typing import List, Optional, Set

from pydantic import BaseModel, Field

from ..models.model import Modality


class ModelAllocationStatus(str, Enum):
    ALLOCATED = "allocated"
    QUEUE = "queue"
    DEALLOCATED = "deallocated"


class ModelCapabilityResponse(BaseModel):
    inputs: Set[Modality]
    outputs: Set[Modality]
    name: Optional[str] = None


class AlloyModelResponse(BaseModel):
    model_id: str
    active_requests: int
    is_supported: bool
    supports_concurrent_requests: bool = False
    capabilities: List[ModelCapabilityResponse]
    allocation_status: ModelAllocationStatus


class AlloyModelsResponse(BaseModel):
    image: List[AlloyModelResponse] = Field(default_factory=list)
    audio: List[AlloyModelResponse] = Field(default_factory=list)
    video: List[AlloyModelResponse] = Field(default_factory=list)
    text: List[AlloyModelResponse] = Field(default_factory=list)
