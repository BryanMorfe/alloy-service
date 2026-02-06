from .. import Modality, ModelCapability
from ..diffusion_model import DiffusionModel

DEFAULT_GUIDANCE_SCALE = 1.5

class GLMImageModel(DiffusionModel):
    model_id = "glm-image"
    required_vram_mb = 48000
    capabilities = [
        ModelCapability(inputs={Modality.TEXT}, outputs={Modality.IMAGE}, name="text-to-image"),
        ModelCapability(inputs={Modality.TEXT, Modality.IMAGE}, outputs={Modality.IMAGE}, name="text-image-to-image"),
    ]
    priority = 1
    supports_concurrent_requests = True

    def __init__(self):
        super().__init__(pretrained_model_id="zai-org/GLM-Image")

    def __call__(self, **kwargs):
        if "guidance_scale" not in kwargs:
            kwargs["guidance_scale"] = DEFAULT_GUIDANCE_SCALE

        return super().__call__(**kwargs)
