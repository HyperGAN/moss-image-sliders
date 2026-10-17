"""Load pinned components through audited local classes, without remote Python execution."""

import torch
from diffusers import AutoencoderKLFlux2
from transformers import AutoModel, AutoTokenizer

from . import MODEL_ID, MODEL_REVISION
from .vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer


def load_pipeline(*, device="cuda:0", local_files_only=True, training=False, offload=False):
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Real MOSS runs require a working CUDA GPU; use smoke for CPU validation")
    if training and offload:
        raise ValueError("Training uses explicit component placement, not inference offload")
    common = dict(revision=MODEL_REVISION, local_files_only=local_files_only)
    transformer = MOSSImageTransformer.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16, **common)
    # Transformers 5.3 AutoTokenizer otherwise requires tokenizer/config.json,
    # absent from the official diffusion snapshot. Its model_index declares Qwen2Tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, subfolder="tokenizer", tokenizer_type="qwen2", **common)
    text_encoder = AutoModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", dtype=torch.bfloat16, **common).eval()
    vae = None if training else AutoencoderKLFlux2.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32, **common).eval()
    pipe = MOSSImagePipeline(transformer, MOSSImageScheduler(), text_encoder, tokenizer, vae)
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    for component in (transformer, text_encoder, vae):
        if component is not None:
            component.requires_grad_(False)
    if training:
        # Encoding happens before moving the DiT to GPU; no need to keep both resident.
        text_encoder.to(device)
    elif offload:
        pipe.enable_model_cpu_offload(gpu_id=int(device.split(":")[-1]))
    else:
        pipe.to(device)
    return pipe


def encode_training_prompts(pipe, prompts, device, max_length=256):
    # Pipeline device detection sees the CPU transformer during this staging phase.
    # Move only its device marker temporarily by using the encoder's tokenizer directly.
    result = []
    with torch.no_grad():
        for prompt in prompts:
            text = pipe.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=True)
            inputs = pipe.tokenizer(text, return_tensors="pt", truncation=True,
                                    max_length=max_length).to(device)
            hidden = pipe.text_encoder(**inputs, use_cache=False, return_dict=True).last_hidden_state
            result.append((hidden.detach().cpu(), inputs.attention_mask.bool().cpu()))
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    pipe.transformer.to(device)
    return [(embeds.to(device), mask.to(device)) for embeds, mask in result]
