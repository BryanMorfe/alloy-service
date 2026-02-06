from .. import Modality, ModelCapability
from ..diffusion_model import DiffusionModel


class ZImageModel(DiffusionModel):
    model_id = "z-image"
    required_vram_mb = 30000
    capabilities = [ModelCapability(inputs={Modality.TEXT}, outputs={Modality.IMAGE}, name="text-to-image")]
    priority = 1
    supports_concurrent_requests = True

    def __init__(self):
        super().__init__(pretrained_model_id="Tongyi-MAI/Z-Image")

    def __call__(self, **kwargs):
        kwargs.pop("image", None)  # z-image does not support image inputs
        return super().__call__(**kwargs)
