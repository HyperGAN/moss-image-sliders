"""Matched-seed slider ramps through the official pipeline."""

import math
import torch


def render(pipe, slider, prompt, *, scale=0.0, seed=42, width=1024, height=1024,
           steps=50, guidance=3.0, skip_layer=True):
    if width <= 0 or height <= 0 or width % 16 or height % 16:
        raise ValueError("Dimensions must be positive multiples of 16")
    if steps < 1 or not math.isfinite(guidance) or guidance < 0:
        raise ValueError("Steps must be positive and guidance finite/nonnegative")
    with slider.at(scale), torch.inference_mode():
        return pipe(prompt, width=width, height=height, num_inference_steps=steps,
                    guidance_scale=guidance, uncond_skip_layers=[22] if skip_layer else [],
                    generator=torch.Generator("cpu").manual_seed(seed)).images[0]
