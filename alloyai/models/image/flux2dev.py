from .. import Modality, ModelCapability, DiffusionModel
from ...image_loader import ImageLoader
from typing import List
import torch
import gc
from diffusers import Flux2Pipeline
from transformers import Mistral3ForConditionalGeneration, PixtralProcessor

def format_text_input(
    prompts: List[str],
    system_message: str
):
    # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
    # when truncation is enabled. The processor counts [IMG] tokens and fails
    # if the count changes after truncation.
    cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]

    return [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        for prompt in cleaned_txt
    ]

class Flux2DevModel(DiffusionModel):
    model_id = "flux2-dev"
    required_vram_mb = {
        "text_encoder": 55000,
        "dit+vae": 65000
    }
    capabilities = [
        ModelCapability(inputs={Modality.TEXT}, outputs={Modality.IMAGE}, name="text-to-image"),
        ModelCapability(inputs={Modality.TEXT, Modality.IMAGE}, outputs={Modality.IMAGE}, name="text-image-to-image")
    ]
    priority = 1
    supports_concurrent_requests = False

    def __init__(self):
        super().__init__(
            pretrained_model_id="black-forest-labs/FLUX.2-dev",
            torch_dtype=torch.bfloat16,
            pipeline_class=Flux2Pipeline,
            text_encoder_type=Mistral3ForConditionalGeneration,
            tokenizer_type=PixtralProcessor,
        )
        

    def __call__(self, **kwargs):
        print("Generating prompt embeddings for Flux2-Dev...")
        prompt = kwargs.get("prompt", "")
        prompt_embeds = kwargs.get("prompt_embeds", None)

        if prompt_embeds is None:
            system_message = (
                "You are an AI that reasons about image descriptions. "
                "You give structured responses focusing on object relationships, "
                "object attribution and actions without speculation."
            )
            prompt_embeds = self._get_prompt_embeds(
                prompt=prompt,
                system_message=system_message,
            )

            # Move to CPU first because of a bug in CUDA with Blackwell arquitecture
            cpu_prompt_embeds = prompt_embeds.to("cpu", copy=True)

            # Clean up embeddings to free VRAM
            del prompt_embeds
            prompt_embeds = None
            gc.collect()
            torch.cuda.empty_cache()
            prompt_embeds = cpu_prompt_embeds.to(self.allocator.pipeline.device)

            kwargs["prompt_embeds"] = prompt_embeds
            kwargs.pop("prompt", None)
        else:
            kwargs.pop("prompt", None)
            prompt_embeds = kwargs["prompt_embeds"]

            # Move to CPU first because of a bug in CUDA with Blackwell arquitecture
            cpu_prompt_embeds = prompt_embeds.to("cpu", copy=True)

            # Move to model device
            del prompt_embeds
            prompt_embeds = None
            gc.collect()
            torch.cuda.empty_cache()
            prompt_embeds = cpu_prompt_embeds.to(self.allocator.pipeline.device)
            kwargs["prompt_embeds"] = prompt_embeds

        print("Embeddings generated!")

        return super().__call__(**kwargs)
    

    def _get_prompt_embeds(
        self,
        prompt: str | List[str],
        system_message: str,
        max_sequence_length: int = 512,
        hidden_states_layers: List[int] = (10, 20, 30),
    ):
        """
        Generates prompt embeddings using the text encoder.

        Requires that the text encoder and tokenizer are already loaded.
        """
        prompt = prompt if isinstance(prompt, list) else [prompt]
        messages_batch = format_text_input(prompts=prompt, system_message=system_message)
        inputs = self.allocator.tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        input_ids = inputs["input_ids"].to(self.allocator.text_encoder.device)
        attention_mask = inputs["attention_mask"].to(self.allocator.text_encoder.device)

        # Forward pass through the model
        output = self.allocator.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Only use outputs from intermediate layers and stack them
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=self.allocator.torch_dtype, device=self.allocator.text_encoder.device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
        return prompt_embeds
