from __future__ import annotations

import base64
import io
import json
import uuid
from typing import Any, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...request_scheduler import AllocationError
from ...runtime import get_runtime

router = APIRouter()


def _extract_images(output: Any) -> List[Any]:
    if output is None:
        return []
    if isinstance(output, dict) and "images" in output:
        return list(output["images"])
    if hasattr(output, "images"):
        return list(output.images)
    if isinstance(output, list):
        return list(output)
    if hasattr(output, "save"):
        return [output]
    return []


def _image_to_base64(image: Any) -> str:
    if not hasattr(image, "save"):
        raise ValueError("Image output does not support .save()")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _event_to_sse(event: Any) -> str:
    payload: dict[str, Any] = {}
    if isinstance(event.payload, dict):
        payload.update(event.payload)
    payload["request_id"] = event.request_id
    payload["timestamp"] = event.timestamp
    return f"event: {event.event}\ndata: {json.dumps(payload)}\n\n"


@router.post("/image")
async def generate_image(request: Request) -> Any:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    payload = dict(payload)
    model_id = payload.pop("model_id", None)
    if not model_id:
        raise HTTPException(status_code=422, detail="model_id is required")
    if "prompt" not in payload:
        raise HTTPException(status_code=422, detail="prompt is required")

    stream = bool(payload.pop("stream", False))
    keep_alive = payload.pop("keep_alive", False)
    priority = payload.pop("priority", None)
    timeout_s = float(payload.pop("allocation_timeout_s", 300))
    poll_s = float(payload.pop("allocation_poll_s", 0.5))
    request_id = payload.pop("request_id", None) or uuid.uuid4().hex

    runtime = get_runtime()
    scheduler = runtime.scheduler

    if not stream:
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

        images = _extract_images(output)
        if not images:
            raise HTTPException(status_code=500, detail="Model did not return images")

        encoded = [_image_to_base64(image) for image in images]
        return {
            "model_id": model_id,
            "images": encoded,
            "request_id": request_id,
            "gpu_id": alloc.gpu_id,
            "gpu_ids": alloc.gpu_ids,
            "gpu_assignment": alloc.gpu_assignment,
            "allocation_status": alloc.status.value,
            "duration_ms": duration_ms,
        }

    def _completed_payload_builder(output: Any, alloc: Any, duration_ms: int) -> dict[str, Any]:
        images = _extract_images(output)
        if not images:
            raise RuntimeError("Model did not return images")
        encoded = [_image_to_base64(image) for image in images]
        return {
            "gpu_id": alloc.gpu_id,
            "gpu_ids": alloc.gpu_ids,
            "gpu_assignment": alloc.gpu_assignment,
            "duration_ms": duration_ms,
            "images": encoded,
        }

    async def event_stream():
        async for event in scheduler.stream(
            model_id,
            payload,
            priority=priority,
            keep_alive=keep_alive,
            timeout_s=timeout_s,
            poll_s=poll_s,
            completed_payload_builder=_completed_payload_builder,
            request_id=request_id,
        ):
            yield _event_to_sse(event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
