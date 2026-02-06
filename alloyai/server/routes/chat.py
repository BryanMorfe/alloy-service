from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ...request_scheduler import AllocationError
from ...runtime import get_runtime

router = APIRouter()


@router.post("/chat")
async def chat(request: Request) -> Any:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    payload = dict(payload)
    model_id = payload.get("model")
    if not model_id:
        raise HTTPException(status_code=422, detail="model is required")

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
        _, output, _ = await scheduler.run(
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

    return output
