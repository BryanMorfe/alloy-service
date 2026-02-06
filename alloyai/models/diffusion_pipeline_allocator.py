import logging

from diffusers import DiffusionPipeline
from transformers import PreTrainedModel, ProcessorMixin
from typing import Optional, Type, Mapping, Union, Tuple
import inspect
import torch
import gc

class DiffusionPipelineAllocator:

    def __init__(
            self,
            model_id: str,
            torch_dtype: torch.dtype,
            pipeline_class: Type[DiffusionPipeline] = DiffusionPipeline,
            text_encoder_type: Optional[Type[PreTrainedModel]] = None,
            tokenizer_type: Optional[Type[ProcessorMixin]] = None,
    ) -> None:
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.pipeline_class = pipeline_class
        self.text_encoder_type = text_encoder_type
        self.tokenizer_type = tokenizer_type
        self.pipeline: Optional[DiffusionPipeline] = None
        self.text_encoder: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[ProcessorMixin] = None


    def allocate_resources(self, gpu_id: Union[str, Mapping[str, str]]) -> None:
        logging.getLogger(__name__).info(
            "Allocating DiffusionModel %s on device %s...",
            self.model_id,
            gpu_id,
        )
        pipeline_device, text_encoder_device = self._resolve_devices(gpu_id)

        if self.text_encoder_type is None:
            self.pipeline = self.pipeline_class.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype
            ).to(pipeline_device)
        else:
            self.pipeline = self.pipeline_class.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                text_encoder=None, # ensure text encoder is not loaded in the pipeline
            ).to(pipeline_device)

        if self.text_encoder_type is not None:
            if text_encoder_device is None:
                raise ValueError(
                    "text_encoder_device must be specified for loading text encoder."
                )
            self.text_encoder = self.text_encoder_type.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                dtype=self.torch_dtype,
                subfolder="text_encoder",
                
            ).to(text_encoder_device)

        if self.tokenizer_type is not None:
            self.tokenizer = self._load_tokenizer()


    def release_resources(self) -> None:
        del self.pipeline
        self.pipeline = None

        del self.text_encoder
        self.text_encoder = None

        del self.tokenizer
        self.tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()


    def get_pipeline(self) -> Optional[DiffusionPipeline]:
        return self.pipeline


    def request_pipeline(self) -> Optional[DiffusionPipeline]:
        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not loaded")

        try:
            scheduler = self.pipeline.scheduler.from_config(self.pipeline.scheduler.config)
        except Exception:
            return None

        try:
            request_pipe = self.pipeline.__class__.from_pipe(
                self.pipeline,
                scheduler=scheduler,
                torch_dtype=self.torch_dtype,
                vae=self.pipeline.vae,
                text_encoder=self.pipeline.text_encoder,
                tokenizer=self.pipeline.tokenizer,
            )
            logging.getLogger(__name__).debug("Created per-request pipeline.")
            return request_pipe
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Failed to create per-request pipeline: %s",
                e,
                exc_info=False,
            )
            return None
        

    def _resolve_devices(
        self, gpu_id: Union[str, Mapping[str, str]]
    ) -> Tuple[str, Optional[str]]:
        if isinstance(gpu_id, Mapping):
            pipeline_device = gpu_id.get("dit+vae")
            if pipeline_device is None:
                raise ValueError("gpu_id mapping must include 'dit+vae' for split allocation.")
            text_encoder_device = gpu_id.get("text_encoder")
            if (self.text_encoder_type or self.tokenizer_type) and text_encoder_device is None:
                raise ValueError(
                    "gpu_id mapping must include 'text_encoder' for split allocation."
                )
            return pipeline_device, text_encoder_device
        if isinstance(gpu_id, str):
            return gpu_id, None
        raise ValueError("gpu_id must be either a string or a mapping of device strings.")

    def _load_tokenizer(self) -> ProcessorMixin:
        if self.tokenizer_type is None:
            raise RuntimeError("tokenizer_type is not configured")

        base_kwargs = {"subfolder": "tokenizer"}
        attempts = [
            {},
            {"use_fast": False},
            {"trust_remote_code": True},
            {"use_fast": False, "trust_remote_code": True},
        ]
        last_exc: Optional[Exception] = None
        for extra in attempts:
            kwargs = dict(base_kwargs)
            kwargs.update(extra)
            try:
                return self.tokenizer_type.from_pretrained(
                    self.model_id,
                    **self._filter_from_pretrained_kwargs(kwargs),
                )
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError(f"Failed to load tokenizer for {self.model_id}") from last_exc

    def _filter_from_pretrained_kwargs(self, kwargs: Mapping[str, object]) -> dict[str, object]:
        try:
            signature = inspect.signature(self.tokenizer_type.from_pretrained)
        except Exception:
            return dict(kwargs)
        if any(param.kind == param.VAR_KEYWORD for param in signature.parameters.values()):
            return dict(kwargs)
        allowed = set(signature.parameters.keys())
        return {key: value for key, value in kwargs.items() if key in allowed}
