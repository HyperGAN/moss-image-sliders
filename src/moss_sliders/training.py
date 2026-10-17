"""Frozen MOSS response and rollout helpers. Training lives in the shared GAN."""

import torch


def velocity(model, latent, timestep, context):
    embeds, mask = context
    return model(latent, timestep, embeds, mask).sample.float()


def train_step(model, slider, optimizer, latent, timestep, contexts, strength=1.0):
    """Reject the historical MSE API before mutating any state."""
    raise RuntimeError("Legacy MSE training is disabled; use moss_sliders.gan.build_run")


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


def checkpoint_blocks(transformer, network=None):
    """Capture each signed forward's scale for shared GAN batch recomputation."""
    if network is None:
        raise ValueError("GAN checkpointing requires the actual shared backend.network")
    from particle_sliders.pg09_backends import checkpoint_with_adapter_scale
    for block in transformer.layers:
        checkpoint_with_adapter_scale(block, network)
