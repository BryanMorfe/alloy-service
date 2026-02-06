
from .. import Model
from typing import Union, Mapping
import gc
import torch

from transformers import pipeline

class PipelineTextModel(Model):
    spreadable = True
    
    def __init__(self, pretrained_model_id: str):
        self.repo_id = pretrained_model_id


    def __call__(self, **kwargs):
        output = self.pipeline(**kwargs)
        # TODO: Maybe convert to ChatMessage, but maybe we could do this on the client-side.
        return output


    def load(self, gpu_id: Union[str, Mapping[str, str]]):
        if gpu_id != "auto":
            raise RuntimeError("Pipeline text models only support \"auto\" for gpu_id")
        
        self.pipeline = pipeline("text-generation", model=self.repo_id, torch_dtype="auto", device_map="auto")

    
    def unload(self):
        del self.pipeline
        self.pipeline = None

        gc.collect()
        torch.cuda.empty_cache()
