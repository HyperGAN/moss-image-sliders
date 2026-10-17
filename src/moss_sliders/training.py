"""Contrastive velocity distillation on shared neutral denoising trajectories."""

import torch
from torch.nn import functional as F


def velocity(model, latent, timestep, context):
    embeds, mask = context
    return model(latent, timestep, embeds, mask).sample.float()


def train_step(model, slider, optimizer, latent, timestep, contexts, strength=1.0):
    """Fit neutral +/- concept direction; keep teachers frozen and on the same noise/time."""
    neutral, positive, negative = contexts
    with torch.no_grad(), slider.at(0):
        base = velocity(model, latent, timestep, neutral)
        delta = velocity(model, latent, timestep, positive) - velocity(model, latent, timestep, negative)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    # Backward while the scale context is active, including checkpoint recomputation.
    for sign in (1, -1):
        with slider.at(sign):
            prediction = velocity(model, latent, timestep, neutral)
            target = base + sign * strength * delta
            loss = F.mse_loss(prediction, target) / 2
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; refusing optimizer step")
            loss.backward()
            losses.append(float(loss.detach()))
    norm = torch.nn.utils.clip_grad_norm_(slider.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {"loss": sum(losses), "grad_norm": float(norm)}


@torch.no_grad()
def sample_training_state(pipe, slider, neutral, generator, resolution, rollout_steps, index):
    """Partial frozen neutral rollout, using official Euler/time conventions (t=1000*sigma)."""
    pipe.scheduler.set_timesteps(rollout_steps, device=pipe.transformer.device)
    times = pipe.scheduler.timesteps
    stop = index % len(times)
    latent = torch.randn((1, pipe.transformer.config.latent_channels,
                          resolution // 16, resolution // 16), generator=generator,
                         device="cpu", dtype=torch.float32).to(pipe.transformer.device)
    with slider.at(0):
        for time in times[:stop]:
            prediction = velocity(pipe.transformer, latent, time, neutral)
            latent = pipe.scheduler.step(prediction, time, latent, return_dict=False)[0]
    return latent.detach(), times[stop]


def checkpoint_blocks(transformer):
    """Checkpoint only main blocks; preserve official forward signatures and adapter hooks."""
    from torch.utils.checkpoint import checkpoint
    for block in transformer.layers:
        original = block.forward

        def forward(*args, _original=original, **kwargs):
            return checkpoint(_original, *args, use_reentrant=False, **kwargs)
        block.forward = forward
