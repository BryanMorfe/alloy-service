from .. import DiffusionModel, Modality, ModelCapability

class LTX2Model(DiffusionModel):
    model_id = "ltx2"
    capabilities = [
        ModelCapability(inputs={Modality.TEXT}, outputs={Modality.VIDEO}, name="text-to-video"),
        ModelCapability(inputs={Modality.IMAGE}, outputs={Modality.VIDEO}, name="image-to-video"),
        ModelCapability(inputs={Modality.AUDIO}, outputs={Modality.VIDEO}, name="audio-to-video"),
    ]
    required_vram_mb = 80000
    priority = 1
    supports_concurrent_requests = True

    def __init__(self):
        super().__init__(pretrained_model_id="Lightricks/LTX-2")

    def __call__(self, **kwargs):
        return super().__call__(**kwargs)