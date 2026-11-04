"""Matched-seed ramps through the shared conditional-only CFG backend."""

import math
import torch

from particle_sliders.pg09_backends import (
    FlowSchedule, MossResponseAdapter, ResponseRow, moss_pipeline_decoder,
)

from . import MODEL_ID, MODEL_REVISION
from .backend import preflight_prompt


def render(pipe, slider, prompt, *, scale=0.0, seed=42, width=1024, height=1024,
           steps=50, guidance=3.0, skip_layer=False, output_type="pil", response_backend=None,
           max_sequence_length=4096):
    if width <= 0 or height <= 0 or width % 16 or height % 16:
        raise ValueError("Dimensions must be positive multiples of 16")
    if steps < 1 or not math.isfinite(guidance) or guidance < 0:
        raise ValueError("Steps must be positive and guidance finite/nonnegative")
    if skip_layer:
        raise ValueError("The shared GAN inference protocol requires negative layer skipping disabled")
    if output_type not in {"pil", "np", "pt", "latent"}:
        raise ValueError("Unsupported output type")
    if response_backend is not None:
        if (response_backend.transformer is not pipe.transformer
                or response_backend.slider is not slider):
            raise ValueError("Inference backend must own the same transformer and slider")
    elif any(getattr(block, "_pg09_scale_checkpoint", False) for block in pipe.transformer.layers):
        raise ValueError("Pass the training response_backend for a checkpointed transformer, "
                         "or load the portable adapter in a fresh inference process")
    # The native encoder truncates by default. Validate both complete formatted
    # branches before calling it, then pass the same explicit budget to encoding.
    for text in (prompt, ""):
        preflight_prompt(pipe.tokenizer, text, max_sequence_length)
    # Validate and restore scale even when encoding or decoder setup fails.
    with slider.at(scale), torch.inference_mode():
        device = pipe._execution_device
        prompt_context = pipe.encode_prompt(prompt, max_sequence_length=max_sequence_length)
        negative_context = pipe.encode_prompt("", max_sequence_length=max_sequence_length)
        scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
        scheduler.set_timesteps(steps, device=device)
        noise = torch.randn(
            (1, pipe.transformer.config.latent_channels, height // 16, width // 16),
            generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32,
        ).to(device)
        row = ResponseRow(prompt_context, prompt_context, negative_context, noise,
                          schedule=FlowSchedule.from_scheduler(scheduler), guidance=guidance)
        backend = response_backend or MossResponseAdapter(
            pipe.transformer, slider,
            model_identity={"model": MODEL_ID, "revision": MODEL_REVISION},
        )
        old_decoder = backend.decoder
        try:
            if output_type == "latent":
                result = backend._response(row, scale=scale)
                if not torch.isfinite(result).all():
                    raise FloatingPointError("Nonfinite MOSS inference latent")
            else:
                backend.decoder = moss_pipeline_decoder(pipe, output_type=output_type)
                result = backend.render(row, scale=scale)
            return result[0]
        finally:
            backend.decoder = old_decoder
            pipe.maybe_free_model_hooks()
