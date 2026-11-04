import copy

import pytest
import torch

from moss_sliders.campaign import (
    cached_backward,
    edit_diagnostics,
    target_bundle,
)
from moss_sliders.lora import Slider
from moss_sliders.smoke import tiny_model
from moss_sliders.training import velocity
from moss_sliders.vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler


def setup():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = tiny_model()
    slider = Slider(model, rank=2)
    optimizer = torch.optim.AdamW(slider.parameters(), lr=0.003)
    context = (torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool))
    positive = (torch.randn(1, 3, 16), context[1])
    latent = torch.randn(1, 4, 2, 2)
    time = torch.tensor(500.0)
    with torch.no_grad():
        base = velocity(model, latent, time, context)
        delta = velocity(model, latent, time, positive) - base
    sample = dict(latent=latent, time=time, base=base, delta=delta)
    return model, slider, optimizer, context, positive, sample



def test_cached_mse_entrypoint_rejects_before_update():
    model, slider, optimizer, context, positive, sample = setup()
    before = copy.deepcopy(slider.state_dict())
    with pytest.raises(RuntimeError, match="Legacy MSE training is disabled"):
        cached_backward(model, slider, sample, context, divisor=8)
    assert all(torch.equal(value, slider.state_dict()[key]) for key, value in before.items())
    assert all(parameter.grad is None for parameter in slider.parameters())
    probe = edit_diagnostics(model, slider, [(sample, context)])
    assert probe["1"]["teacher_rms"] > 0


def test_target_bundle_uses_both_paths_same_noise_and_all_times():
    model, slider, _, context, positive, _ = setup()
    pipe = MOSSImagePipeline(model, MOSSImageScheduler())
    first = target_bundle(pipe, slider, context, positive, 11, resolution=32, steps=4)
    repeat = target_bundle(pipe, slider, context, positive, 11, resolution=32, steps=4)
    assert all(torch.equal(value, repeat[key]) for key, value in first.items())
    assert first["latent"].shape == (2, 4, 1, 4, 2, 2)
    assert torch.equal(first["latent"][0, 0], first["latent"][1, 0])
    assert not torch.equal(first["latent"][0, -1], first["latent"][1, -1])
    assert len(set(first["time"].tolist())) == 4
    assert all(not value.requires_grad and value.isfinite().all() for value in first.values())
