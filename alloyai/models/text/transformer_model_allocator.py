from transformers import Qwen3Model
import gc
import torch

class TransformerModelAllocator:
    def __init__(self, pretrained_model_id: str):
        self.pretrained_model_id = pretrained_model_id
        self.pipeline = None

    def allocate_model(self, gpu_id: str) -> Qwen3Model:
        if gpu_id == "auto":
            self.pipeline = Qwen3Model.from_pretrained(self.pretrained_model_id, device_map="auto")
        else:
            self.pipeline = Qwen3Model.from_pretrained(self.pretrained_model_id, device_map={"": int(gpu_id)})

    def deallocate_model(self) -> None:
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
