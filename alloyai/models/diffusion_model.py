import logging
import threading

from ..image_loader import ImageLoader
from . import EventPublisher, Model
from .diffusion_pipeline_allocator import DiffusionPipelineAllocator
from .get_diffusion_callback import get_diffusion_callback
import torch
import gc
from diffusers import DiffusionPipeline
from transformers import PreTrainedModel, ProcessorMixin
from typing import Optional, Type, Mapping

class DiffusionModel(EventPublisher, Model):
    spreadable = False

    def __init__(
            self,
            pretrained_model_id: str,
            torch_dtype: torch.dtype = torch.bfloat16,
            pipeline_class: Type[DiffusionPipeline] = DiffusionPipeline,
            text_encoder_type: Optional[Type[PreTrainedModel]] = None,
            tokenizer_type: Optional[Type[ProcessorMixin]] = None,
    ):
        super().__init__()
        self.allocator = DiffusionPipelineAllocator(
            model_id=pretrained_model_id,
            torch_dtype=torch_dtype,
            pipeline_class=pipeline_class,
            text_encoder_type=text_encoder_type,
            tokenizer_type=tokenizer_type,
        )
        self._active_requests = 0
        self._counter_lock = threading.Lock()

    def __call__(self, **kwargs):
        image_urls = kwargs.get("image", None)
        if image_urls is not None:
            # Load images from URLs / paths
            images = [ImageLoader.load(url) for url in image_urls]
            kwargs["image"] = images

        request_id = kwargs.pop("request_id", None)
        total_steps = kwargs.get("num_inference_steps")
        existing_callback = kwargs.get("callback_on_step_end")

        logging.getLogger(__name__).info(
            "Starting diffusion generation for model %s with rid %s",
            self.model_id,
            request_id,
        )

        if request_id:
            callback = get_diffusion_callback(
                request_id=request_id,
                total_steps=total_steps,
                existing_callback=existing_callback,
                publisher=self,
            )
            kwargs["callback_on_step_end"] = callback

        with self._counter_lock:
            self._active_requests += 1
            active_requests = self._active_requests

        request_pipe = None
        try:
            if active_requests > 1:
                request_pipe = self.allocator.request_pipeline()
                if request_pipe is None:
                    raise RuntimeError("Failed to create per-request pipeline")
                output = request_pipe(**kwargs)

                # clear up request_pipe to free memory
                del request_pipe
                gc.collect()
                torch.cuda.empty_cache()

                return output

            shared_pipe = self.allocator.get_pipeline()
            if shared_pipe is None:
                raise RuntimeError("Model pipeline is not loaded")
            return shared_pipe(**kwargs)
        finally:
            request_pipe = None
            with self._counter_lock:
                if self._active_requests > 0:
                    self._active_requests -= 1


    def load(self, gpu_id: str | Mapping[str, str]) -> None:
        if gpu_id == "auto":
            raise ValueError("DiffusionModel does not support auto device mapping")
        self.allocator.allocate_resources(gpu_id)


    def unload(self) -> None:
        self.allocator.release_resources()
