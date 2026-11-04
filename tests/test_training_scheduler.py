"""Pinned SFT schedules and isolation of a caller's in-progress Euler state."""
from copy import deepcopy

import pytest
import torch

from moss_sliders.lora import Slider
from moss_sliders.smoke import tiny_model
from moss_sliders.training import sample_training_state, velocity
from moss_sliders.vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler


def test_sft_rollout_and_render_grids_are_native_and_distinct():
    four, fifty = MOSSImageScheduler(), MOSSImageScheduler()
    four.set_timesteps(4)
    fifty.set_timesteps(50)
    torch.testing.assert_close(four.timesteps, torch.tensor([1000., 910., 717., 24.]),
                               rtol=0, atol=0.0001)
    assert len(fifty.timesteps) == 50
    assert fifty.timesteps[0] == four.timesteps[0] == 1000
    assert fifty.timesteps[-1] == four.timesteps[-1] == 24
    assert fifty.config.shift == four.config.shift == 5
    assert fifty.config.training_shift == four.config.training_shift == 1


@pytest.mark.parametrize("fail", [False, True])
def test_partial_rollout_leaves_borrowed_scheduler_and_scale_unchanged(fail, monkeypatch):
    from test_gan import equal
    torch.set_num_threads(1)
    model = tiny_model()
    slider = Slider(model, rank=2)
    scheduler = MOSSImageScheduler()
    scheduler.set_timesteps(50)
    scheduler.step(torch.zeros(1, 4, 2, 2), scheduler.timesteps[0],
                   torch.zeros(1, 4, 2, 2), return_dict=False)
    before = deepcopy(scheduler.__dict__)
    pipe = MOSSImagePipeline(model, scheduler)
    context = (torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool))
    if fail:
        def failure(*args, **kwargs):
            raise RuntimeError("rollout failure")
        monkeypatch.setattr(model, "forward", failure)
    with slider.at(.35):
        if fail:
            with pytest.raises(RuntimeError, match="rollout failure"):
                sample_training_state(pipe, slider, context, torch.Generator().manual_seed(3),
                                      32, 4, 3)
        else:
            actual, time = sample_training_state(pipe, slider, context,
                                                 torch.Generator().manual_seed(3), 32, 4, 3)
            # Reproduce the pre-fix numerical trajectory in a separate scheduler.
            reference = MOSSImageScheduler.from_config(scheduler.config)
            reference.set_timesteps(4)
            expected = torch.randn(1, 4, 2, 2, generator=torch.Generator().manual_seed(3))
            with slider.at(0), torch.no_grad():
                for t in reference.timesteps[:3]:
                    expected = reference.step(velocity(model, expected, t, context), t,
                                              expected, return_dict=False)[0]
            assert torch.equal(expected, actual) and time == reference.timesteps[3]
        assert slider.scale == .35
    assert pipe.scheduler is scheduler
    equal(before, scheduler.__dict__)
