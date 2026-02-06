from __future__ import annotations

import gc
import os
import tempfile
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import torch

from .. import EventPublisher, Model, Modality, ModelCapability


def _ensure_list(value: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _is_url(value: str) -> bool:
    parts = urllib.parse.urlparse(value)
    return parts.scheme in {"http", "https"}


def _download_audio(url: str) -> str:
    suffix = os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".wav"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.close()
    urllib.request.urlretrieve(url, handle.name)
    return handle.name


def _normalize_outputs(outputs: Iterable[Any]) -> List[Any]:
    normalized: List[Any] = []
    for item in outputs:
        if hasattr(item, "tolist"):
            normalized.append(item.tolist())
        else:
            normalized.append(item)
    return normalized


class QwenTTSPipeline(EventPublisher, Model):
    required_vram_mb = 6000
    priority = 1
    spreadable = False
    supports_concurrent_requests = False

    def __init__(
        self,
        repo_id: str,
        *,
        method_name: str,
        allow_speaker: bool = False,
        allow_ref_audio: bool = False,
        allow_ref_text: bool = False,
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self._method_name = method_name
        self._allow_speaker = allow_speaker
        self._allow_ref_audio = allow_ref_audio
        self._allow_ref_text = allow_ref_text
        self._model: Optional[Any] = None

    def load(self, gpu_id: Union[str, Mapping[str, str]]) -> None:
        if not isinstance(gpu_id, str) or gpu_id == "auto":
            raise ValueError("Qwen TTS models require a concrete gpu_id like 'cuda:0'")
        from qwen_tts import Qwen3TTSModel

        self._model = Qwen3TTSModel.from_pretrained(
            self.repo_id,
            device_map=gpu_id,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    def unload(self) -> None:
        del self._model
        self._model = None
        gc.collect()
        torch.cuda.empty_cache()

    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Model is not loaded")

        text = _ensure_list(kwargs.get("text"))
        if not text:
            raise ValueError("text is required")

        language = _ensure_list(kwargs.get("language"))
        speaker = _ensure_list(kwargs.get("speaker"))
        instruct = _ensure_list(kwargs.get("instruct"))
        ref_audio = _ensure_list(kwargs.get("ref_audio"))
        ref_text = _ensure_list(kwargs.get("ref_text"))

        if speaker and not self._allow_speaker:
            raise ValueError("speaker is only supported for the voice-design variant")
        if ref_audio and not self._allow_ref_audio:
            raise ValueError("ref_audio is only supported for the base variant")
        if ref_text and not self._allow_ref_text:
            raise ValueError("ref_text is only supported for the base variant")

        cleanup_paths: List[str] = []
        if ref_audio:
            resolved: List[str] = []
            for item in ref_audio:
                if isinstance(item, str) and _is_url(item):
                    path = _download_audio(item)
                    cleanup_paths.append(path)
                    resolved.append(path)
                else:
                    resolved.append(item)
            ref_audio = resolved

        payload: Dict[str, Any] = {
            "text": text,
        }
        if language is not None:
            payload["language"] = language
        if instruct is not None:
            payload["instruct"] = instruct
        if speaker is not None:
            payload["speaker"] = speaker
        if ref_audio is not None:
            payload["ref_audio"] = ref_audio
        if ref_text is not None:
            payload["ref_text"] = ref_text

        try:
            generator = getattr(self._model, self._method_name)
            outputs, sample_rate = generator(**payload)
        finally:
            for path in cleanup_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

        return {
            "outputs": _normalize_outputs(outputs),
            "sample_rate": int(sample_rate),
        }


class QwenTTSVoiceDesignModel(QwenTTSPipeline):
    model_id = "qwen3-tts-voice-design"
    capabilities = [
        ModelCapability(inputs={Modality.TEXT}, outputs={Modality.AUDIO}, name="text-to-audio"),
    ]

    def __init__(self) -> None:
        super().__init__(
            repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            method_name="generate_voice_design",
            allow_speaker=True,
        )


class QwenTTSCustomVoiceModel(QwenTTSPipeline):
    model_id = "qwen3-tts-custom-voice"
    capabilities = [
        ModelCapability(inputs={Modality.TEXT}, outputs={Modality.AUDIO}, name="text-to-audio"),
    ]

    def __init__(self) -> None:
        super().__init__(
            repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            method_name="generate_custom_voice",
        )


class QwenTTSBaseModel(QwenTTSPipeline):
    model_id = "qwen3-tts-base"
    capabilities = [
        ModelCapability(
            inputs={Modality.TEXT, Modality.AUDIO},
            outputs={Modality.AUDIO},
            name="text-audio-to-audio",
        ),
    ]

    def __init__(self) -> None:
        super().__init__(
            repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            method_name="generate_voice_clone",
            allow_ref_audio=True,
            allow_ref_text=True,
        )
