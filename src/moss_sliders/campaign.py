"""Read-only frozen target preparation and diagnostics; GAN training is shared."""

import torch

from .training import velocity


def cached_backward(model, slider, sample, context, divisor=1):
    """Reject legacy MSE updates; target preparation/diagnostics remain read-only."""
    raise RuntimeError("Legacy MSE training is disabled; use moss_sliders.gan.build_cached_run")


@torch.no_grad()
def target_bundle(pipe, slider, neutral, positive, seed, resolution=512, steps=10):
    """Both frozen paths from identical noise, both prompt velocities at every position."""
    device = pipe.transformer.device
    noise = torch.randn(
        (1, pipe.transformer.config.latent_channels, resolution // 16, resolution // 16),
        generator=torch.Generator("cpu").manual_seed(seed),
    ).to(device)
    paths = {key: [] for key in ("latent", "base", "delta")}
    with slider.at(0):
        for positive_path in (False, True):
            pipe.scheduler.set_timesteps(steps, device=device)
            latent = noise.clone()
            values = {key: [] for key in paths}
            for time in pipe.scheduler.timesteps:
                base = velocity(pipe.transformer, latent, time, neutral)
                edited = velocity(pipe.transformer, latent, time, positive)
                for key, value in dict(latent=latent, base=base, delta=edited - base).items():
                    if not value.isfinite().all():
                        raise RuntimeError("Non-finite frozen target")
                    values[key].append(value.detach().cpu())
                latent = pipe.scheduler.step(
                    edited if positive_path else base, time, latent, return_dict=False
                )[0]
            for key in paths:
                paths[key].append(torch.stack(values[key]))
    return {
        **{key: torch.stack(value) for key, value in paths.items()},
        "time": pipe.scheduler.timesteps.detach().cpu(),
    }


@torch.no_grad()
def edit_diagnostics(model, slider, fixtures):
    """Fixed held-out states, separately reporting both signs; no fitted normalization."""
    fixtures = list(fixtures)
    if not fixtures:
        raise ValueError("No fixed diagnostic states")
    result = {}
    for sign in (1, -1):
        error = teacher_energy = student_energy = dot = count = 0.0
        for sample, context in fixtures:
            with slider.at(sign):
                student = (
                    velocity(model, sample["latent"], sample["time"], context) - sample["base"]
                )
            teacher = sign * sample["delta"]
            error += float((student - teacher).square().sum())
            teacher_energy += float(teacher.square().sum())
            student_energy += float(student.square().sum())
            dot += float((student * teacher).sum())
            count += teacher.numel()
        result[str(sign)] = dict(
            relative_edit_mse=error / max(teacher_energy, 1e-30),
            teacher_rms=(teacher_energy / count) ** 0.5,
            student_rms=(student_energy / count) ** 0.5,
            edit_rms_ratio=(student_energy / max(teacher_energy, 1e-30)) ** 0.5,
            edit_cosine=dot / max((student_energy * teacher_energy) ** 0.5, 1e-30),
        )
    return result
