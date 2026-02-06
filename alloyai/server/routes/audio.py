from __future__ import annotations

import uuid
from typing import Any, Iterable, List

from fastapi import APIRouter, HTTPException, Request

from ...request_scheduler import AllocationError
from ...runtime import get_runtime

router = APIRouter()


def _normalize_outputs(outputs: Any) -> List[Any]:
    if outputs is None:
        return []
    if isinstance(outputs, list):
        items = outputs
    elif isinstance(outputs, tuple):
        items = list(outputs)
    else:
        items = [outputs]

    normalized: List[Any] = []
    for item in items:
        if hasattr(item, "tolist"):
            normalized.append(item.tolist())
        else:
            normalized.append(item)
    return normalized


def _normalize_audio_output(output: Any) -> dict[str, Any]:
    if isinstance(output, dict) and "outputs" in output and "sample_rate" in output:
        outputs = _normalize_outputs(output["outputs"])
        return {"outputs": outputs, "sample_rate": int(output["sample_rate"])}
    if isinstance(output, tuple) and len(output) == 2:
        outputs, sample_rate = output
        return {"outputs": _normalize_outputs(outputs), "sample_rate": int(sample_rate)}
    raise RuntimeError("Model did not return audio outputs")


@router.post("/audio")
async def generate_audio(request: Request) -> Any:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    payload = dict(payload)
    model_id = payload.pop("model_id", None)
    if not model_id:
        raise HTTPException(status_code=422, detail="model_id is required")
    if "text" not in payload:
        raise HTTPException(status_code=422, detail="text is required")

    stream = bool(payload.pop("stream", False))
    if stream:
        raise HTTPException(status_code=400, detail="Streaming not supported yet")

    keep_alive = payload.pop("keep_alive", False)
    priority = payload.pop("priority", None)
    timeout_s = float(payload.pop("allocation_timeout_s", 300))
    poll_s = float(payload.pop("allocation_poll_s", 0.5))
    request_id = payload.pop("request_id", None) or uuid.uuid4().hex

    runtime = get_runtime()
    scheduler = runtime.scheduler

    try:
        alloc, output, duration_ms = await scheduler.run(
            model_id,
            payload,
            priority=priority,
            keep_alive=keep_alive,
            timeout_s=timeout_s,
            poll_s=poll_s,
            request_id=request_id,
        )
    except AllocationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None

    audio = _normalize_audio_output(output)
    return {
        "model_id": model_id,
        "request_id": request_id,
        "gpu_id": alloc.gpu_id,
        "gpu_ids": alloc.gpu_ids,
        "gpu_assignment": alloc.gpu_assignment,
        "allocation_status": alloc.status.value,
        "duration_ms": duration_ms,
        **audio,
    }
