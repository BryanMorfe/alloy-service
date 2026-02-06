from . import EventPublisher, ModelEvent
from typing import Any, Dict, Optional

def get_diffusion_callback(
        request_id: str,
        total_steps: Optional[int],
        existing_callback: Optional[Any],
        publisher: EventPublisher
) -> Any:
    def _callback_on_step_end(
        _pipe: Any,
        step_index: int,
        timestep: Any,
        callback_kwargs: Optional[Dict],
    ):
        payload: Dict[str, Any] = {"step": int(step_index)}
        if total_steps is not None:
            payload["total_steps"] = int(total_steps)
            payload["progress"] = (step_index + 1) / max(int(total_steps), 1)
        try:
            payload["timestep"] = int(timestep)
        except Exception:
            pass
        publisher.publish(
            ModelEvent(
                event="progress",
                request_id=request_id,
                payload=payload,
            )
        )
        if existing_callback is not None:
            result = existing_callback(_pipe, step_index, timestep, callback_kwargs)
            return callback_kwargs if result is None else result
        return callback_kwargs

    return _callback_on_step_end