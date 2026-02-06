from .. import Modality, ModelCapability
from ..diffusion_model import DiffusionModel

DEFAULT_TRUE_CFG_SCALE = 4.0

class QwenImageEditModel(DiffusionModel):
    model_id = "qwen-image-edit"
    required_vram_mb = 60000
    capabilities = [ModelCapability(inputs={Modality.TEXT, Modality.IMAGE}, outputs={Modality.IMAGE}, name="text-image-to-image")]
    priority = 1
    supports_concurrent_requests = True

    def __init__(self):
        super().__init__(pretrained_model_id="Qwen/Qwen-Image-Edit-2511")

    def __call__(self, **kwargs):
        if "image" not in kwargs:
            raise ValueError("QwenImageEditModel requires an 'image' input.")

        if "true_cfg_scale" not in kwargs:
            kwargs["true_cfg_scale"] = DEFAULT_TRUE_CFG_SCALE

        return super().__call__(**kwargs)
