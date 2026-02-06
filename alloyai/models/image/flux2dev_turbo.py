from .. import Modality, ModelCapability
from . import Flux2DevModel

# Pre-shifted custom sigmas for 8-step turbo inference
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

class Flux2DevTurboModel(Flux2DevModel):
    model_id = "flux2-dev-turbo"

    def __init__(self):
        super().__init__()
        

    def __call__(self, **kwargs):
        kwargs["num_inference_steps"] = 9 # 8 steps + 1 initial noise for turbo - fixed
        kwargs["sigmas"] = TURBO_SIGMAS

        return super().__call__(**kwargs)
    

    def load(self, gpu_id):
        super().load(gpu_id)
        # load turbo LoRA weights
        self.allocator.pipeline.load_lora_weights(
            "fal/FLUX.2-dev-Turbo",
            weight_name="flux.2-turbo-lora.safetensors"
        )
