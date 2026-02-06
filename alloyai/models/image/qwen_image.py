from .. import Modality, ModelCapability
from ..diffusion_model import DiffusionModel

DEFAULT_NEGATIVE_PROMPT = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
DEFAULT_TRUE_CFG_SCALE = 4.0

class QwenImageModel(DiffusionModel):
    model_id = "qwen-image"
    required_vram_mb = 60000
    capabilities = [ModelCapability(inputs={Modality.TEXT}, outputs={Modality.IMAGE}, name="text-to-image")]
    priority = 1
    supports_concurrent_requests = True

    def __init__(self):
        super().__init__(pretrained_model_id="Qwen/Qwen-Image-2512")

    def __call__(self, **kwargs):
        kwargs.pop("image", None)  # qwen-image does not support image inputs
        if "negative_prompt" not in kwargs:
            kwargs["negative_prompt"] = DEFAULT_NEGATIVE_PROMPT
        if "true_cfg_scale" not in kwargs:
            kwargs["true_cfg_scale"] = DEFAULT_TRUE_CFG_SCALE

        return super().__call__(**kwargs)
