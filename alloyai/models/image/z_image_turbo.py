from .. import Modality, ModelCapability
from ..diffusion_model import DiffusionModel

class ZImageTurboModel(DiffusionModel):
    model_id = "z-image-turbo"
    required_vram_mb = 30000
    capabilities = [ModelCapability(inputs={Modality.TEXT}, outputs={Modality.IMAGE}, name="text-to-image")]
    priority = 1
    supports_concurrent_requests = True

    def __init__(self):
        super().__init__(pretrained_model_id="Tongyi-MAI/Z-Image-Turbo")

    def __call__(self, **kwargs):
        kwargs["guidance_scale"] = 0.0 # should always be 0.0 for z-image-turbo
        kwargs.pop("image", None)  # z-image-turbo does not support image inputs
        return super().__call__(**kwargs)
