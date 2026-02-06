from .. import Modality, ModelCapability
from ..diffusion_model import DiffusionModel

class Flux2KleinModel(DiffusionModel):
    model_id = "flux2-klein-9b"
    required_vram_mb = 50000
    capabilities = [
        ModelCapability(inputs={Modality.TEXT}, outputs={Modality.IMAGE}, name="text-to-image"),
        ModelCapability(inputs={Modality.TEXT, Modality.IMAGE}, outputs={Modality.IMAGE}, name="text-image-to-image")
    ]
    priority = 1
    supports_concurrent_requests = True

    def __init__(self):
        super().__init__(pretrained_model_id="black-forest-labs/FLUX.2-klein-9B")

    def __call__(self, **kwargs):
        return super().__call__(**kwargs)