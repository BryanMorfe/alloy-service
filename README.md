# Alloy

Alloy is a multi-modal model generation backend that provides generation endpoints for text, image, audio, and video.

It manages GPU memory, allocates and deallocates models based on requests, and supports streaming progress events.

Right now, the image and chat endpoints are implemented.

## Quick start

Run the server:

```bash
alloy serve --host 0.0.0.0 --port 8000
```

List models and show GPU status:

```bash
alloy list
alloy ps
```

## CUDA + PyTorch

CUDA is not pinned in `pyproject.toml`. Install the PyTorch build that matches your
driver and CUDA version (or CPU-only) using the official PyTorch install command.

Examples:

```bash
# CUDA (example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Then install Alloy with the desired extras, e.g.:

```bash
pip install -e .[all]
```

## Default models

- `qwen-image`
- `z-image-turbo`
- `flux2-dev`
- `flux2-dev-turbo`

## HTTP API

### GET /models

Returns grouped model metadata by output modality:

```json
{
  "image": [
    {
      "model_id": "qwen-image",
      "active_requests": 0,
      "is_supported": true,
      "capabilities": [
        {
          "inputs": ["text"],
          "outputs": ["image"],
          "name": "text-to-image"
        }
      ],
      "allocation_status": "deallocated"
    }
  ],
  "audio": [],
  "video": [],
  "text": []
}
```

`is_supported` is computed from model registration and total node GPU capacity (ignores current usage).

### POST /image

Request body (JSON):

```json
{
  "model_id": "qwen-image",
  "prompt": "A photo of a red fox in a forest",
  "negative_prompt": "blurry, low resolution",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 30,
  "true_cfg_scale": 4.0,
  "stream": false,
  "keep_alive": false,
  "priority": 1,
  "allocation_timeout_s": 300,
  "allocation_poll_s": 0.5
}
```

Notes:
- `prompt` can be a string or list of strings.
- Any additional fields are passed directly to the model.
- `keep_alive` can be `false`, `true`, or a number of seconds to keep the model resident after use.

Response (non-streaming):

```json
{
  "model_id": "qwen-image",
  "images": ["<base64-png>", "<base64-png>"],
  "gpu_id": "cuda:0",
  "allocation_status": "allocated",
  "duration_ms": 12345
}
```

### Streaming

Set `"stream": true` to receive Server-Sent Events (SSE). Progress events include step info when the model supports it.

```bash
curl -N -H "Content-Type: application/json" \
  -d '{"model_id":"qwen-image","prompt":"A red fox","stream":true}' \
  http://127.0.0.1:8000/image
```

Event types:
- `received`
- `queued`
- `allocated`
- `progress` (step, total_steps, progress, timestep)
- `completed` (includes `images`)
- `error`
- `done`

### POST /chat

Request body (JSON):

```json
{
  "model": "qwen3-medium",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false,
  "keep_alive": false,
  "priority": 1,
  "allocation_timeout_s": 300,
  "allocation_poll_s": 0.5
}
```

Response (non-streaming):

```json
{
  "model": "qwen3-medium",
  "created_at": "2026-01-11T16:18:40.43034166Z",
  "done": true,
  "message": {"role": "assistant", "content": "Hello!"}
}
```

## Python client

```python
from alloy.client import AlloyClient

client = AlloyClient("http://127.0.0.1:8000")

# Non-streaming
resp = client.image(model_id="qwen-image", prompt="A red fox")
image_bytes = resp["images"][0]

# Streaming
for event in client.image(model_id="qwen-image", prompt="A red fox", stream=True):
    print(event["event"], event.get("payload"))

# Chat (non-streaming)
resp = client.chat(
    model="qwen-medium",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp)
```
