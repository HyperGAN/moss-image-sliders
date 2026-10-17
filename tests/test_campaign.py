import copy
import json

import pytest
import torch

from moss_sliders.campaign import (
    StateSampler,
    cached_backward,
    edit_diagnostics,
    target_bundle,
    latest_training_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from moss_sliders.lora import Slider
from moss_sliders.smoke import tiny_model
from moss_sliders.training import train_step, velocity
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


def test_cartesian_sampling_covers_every_row_time_and_restores():
    sampler = StateSampler(24 * 16 * 2 * 10, 7)
    first = sampler.take(sampler.size)
    assert sorted(first) == list(range(sampler.size))
    for row in range(24):
        assert {i % 10 for i in first if i // (16 * 2 * 10) == row} == set(range(10))
    saved = copy.deepcopy(sampler.state_dict())
    expected = sampler.take(100)
    restored = StateSampler(sampler.size, 999)
    restored.load_state_dict(saved)
    assert restored.take(100) == expected
    with pytest.raises(ValueError, match="dataset changed"):
        StateSampler(4, 0).load_state_dict(saved)


def test_cached_effective_batch_preserves_original_signed_objective():
    model, slider, optimizer, context, positive, sample = setup()
    expected = train_step(
        model, slider, optimizer, sample["latent"], sample["time"], [context, positive, context]
    )
    expected_state = copy.deepcopy(slider.state_dict())
    model, slider, optimizer, context, positive, sample = setup()
    optimizer.zero_grad(set_to_none=True)
    observed = [cached_backward(model, slider, sample, context, divisor=8) for _ in range(8)]
    torch.nn.utils.clip_grad_norm_(slider.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    assert observed[0] == pytest.approx(expected["loss"])
    for key, value in expected_state.items():
        torch.testing.assert_close(value, slider.state_dict()[key], atol=1e-7, rtol=1e-5)
    probe = edit_diagnostics(model, slider, [(sample, context)])
    assert probe["1"]["teacher_rms"] > 0
    assert probe["1"]["relative_edit_mse"] < 1
    assert edit_diagnostics(model, slider, iter([(sample, context)]))["-1"]["teacher_rms"] > 0


def test_optimizer_rng_resume_matches_uninterrupted_training(tmp_path):
    model, slider, optimizer, context, _, sample = setup()
    sampler = StateSampler(20, 9)
    identity = dict(model="tiny", recipe="fixed")

    def update():
        sampler.take(8)
        modified = {**sample, "delta": sample["delta"] * (0.5 + torch.rand(()))}
        optimizer.zero_grad(set_to_none=True)
        cached_backward(model, slider, modified, context)
        optimizer.step()

    update()
    directory = save_training_checkpoint(tmp_path, 1, slider, optimizer, sampler, identity)
    update()
    expected = copy.deepcopy(slider.state_dict())
    expected_next = sampler.take(8)
    expected_random = torch.rand(4)
    assert restore_training_checkpoint(directory, slider, optimizer, sampler, identity) == 1
    update()
    assert all(torch.equal(expected[k], v) for k, v in slider.state_dict().items())
    assert sampler.take(8) == expected_next
    assert torch.equal(torch.rand(4), expected_random)
    with pytest.raises(ValueError, match="identity"):
        restore_training_checkpoint(directory, slider, optimizer, sampler, {"other": True})
    with pytest.raises(FileExistsError):
        save_training_checkpoint(tmp_path, 1, slider, optimizer, sampler, identity)
    assert json.loads((tmp_path / "LATEST.json").read_text())["step"] == 1
    (tmp_path / "LATEST.json").unlink()
    (tmp_path / ".pending-interrupted").mkdir()
    assert latest_training_checkpoint(tmp_path, identity) == directory
    with (directory / "training.pt").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        restore_training_checkpoint(directory, slider, optimizer, sampler, identity)


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
