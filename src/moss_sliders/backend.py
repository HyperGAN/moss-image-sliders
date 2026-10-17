"""Load pinned components through audited local classes, without remote Python execution."""

import hashlib

import torch
from diffusers import AutoencoderKLFlux2
from transformers import AutoModel, AutoTokenizer

from . import MODEL_ID, MODEL_REVISION
from .vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer


def load_transformer(*, device="cuda:0", local_files_only=True):
    """Load only the pinned DiT for an explicitly admitted cached campaign."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Real MOSS runs require a working CUDA GPU; use smoke for CPU validation")
    return MOSSImageTransformer.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
        revision=MODEL_REVISION, local_files_only=local_files_only,
    ).to(device).eval().requires_grad_(False)


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


def preflight_prompt(tokenizer, prompt, max_sequence_length=4096, *, purpose="Inference"):
    """Tokenize the complete native chat template and reject silent truncation."""
    if type(max_sequence_length) is not int or not 1 <= max_sequence_length <= 4096:
        raise ValueError("max_sequence_length must be an integer between 1 and 4096")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if tokenizer is None:
        raise ValueError("A native tokenizer is required to validate the full prompt")
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    token_count = inputs["input_ids"].shape[-1]
    if token_count > max_sequence_length:
        raise ValueError(f"{purpose} prompt has {token_count} tokens; budget is {max_sequence_length}; "
                         "shorten the prompt or explicitly raise max_sequence_length")
    identity = {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "formatted_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "input_ids_sha256": hashlib.sha256(
                    inputs["input_ids"].contiguous().numpy().tobytes()).hexdigest(),
                "mask_sha256": hashlib.sha256(
                    inputs["attention_mask"].contiguous().numpy().tobytes()).hexdigest(),
                "token_count": token_count, "max_sequence_length": max_sequence_length,
                "truncated": False}
    return inputs, identity


def encode_training_prompts(pipe, prompts, device, max_length=256):
    # Pipeline device detection sees the CPU transformer during this staging phase.
    # Move only its device marker temporarily by using the encoder's tokenizer directly.
    prepared, identity = [], []
    # Validate every full prompt before any encoder forward. A truncated positive
    # pole can otherwise silently become the neutral pole.
    for prompt in prompts:
        inputs, info = preflight_prompt(pipe.tokenizer, prompt, max_length, purpose="Training")
        prepared.append(inputs)
        identity.append(info)
    result = []
    with torch.no_grad():
        for inputs in prepared:
            inputs = inputs.to(device)
            hidden = pipe.text_encoder(**inputs, use_cache=False, return_dict=True).last_hidden_state
            result.append((hidden.detach().cpu(), inputs.attention_mask.bool().cpu()))
    pipe._moss_training_context_identity = identity
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    pipe.transformer.to(device)
    return [(embeds.to(device), mask.to(device)) for embeds, mask in result]
