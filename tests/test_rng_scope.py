"""CPU regressions for caller-owned RNG initialization; no CUDA runtime calls."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from moss_sliders import gan
from moss_sliders.smoke import tiny_model
from moss_sliders.vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler


@pytest.fixture(autouse=True)
def isolated_cpu_rng_and_forbidden_global_cuda_seed(monkeypatch):
    torch.set_num_threads(1)
    ambient = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Initialization must not seed unowned CUDA generators")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    monkeypatch.setattr(torch.cuda, "manual_seed", forbidden)
    try:
        yield
    finally:
        torch.set_rng_state(ambient)


def fixture():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(91)
        model = tiny_model()
        original = deepcopy(model.state_dict())
        contexts = [(torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool))
                    for _ in range(3)]
    config = dict(objective="pg09_gan_only", pg_version="0.9.1", rank=2, seed=17,
                  resolution=32, rollout_steps=2, strength=1., batch_size=2,
                  steps=8, lr=.001, critic_width=8)

    def new_pipe():
        with torch.random.fork_rng(devices=[]):
            model = tiny_model()
            model.load_state_dict(original)
        return MOSSImagePipeline(model, MOSSImageScheduler())

    return new_pipe, contexts, config


def test_actual_tiny_build_never_seeds_unowned_cuda_and_restores_cpu_rng(monkeypatch):
    new_pipe, contexts, config = fixture()
    pipe = new_pipe()

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU initialization must not enter the CUDA runtime")

    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(torch.cuda, "get_rng_state", forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state", forbidden)
    before = torch.get_rng_state().clone()
    run = gan.build_run(pipe, contexts, config)
    assert torch.equal(before, torch.get_rng_state())
    assert type(run.game).__module__ == "particle_sliders.pg09"
    assert len(run.game.rows) == 4
    assert any(p.count_nonzero() for p in run.slider.parameters())


def test_same_seed_reproduces_actual_adapter_critic_and_frozen_rows():
    new_pipe, contexts, config = fixture()
    first = gan.build_run(new_pipe(), contexts, config)
    torch.random.default_generator.manual_seed(873)
    second = gan.build_run(new_pipe(), contexts, config)
    for module_a, module_b in ((first.slider, second.slider),
                               (first.game.critic, second.game.critic)):
        assert module_a.state_dict().keys() == module_b.state_dict().keys()
        for name, value in module_a.state_dict().items():
            assert torch.equal(value, module_b.state_dict()[name])
    assert torch.equal(first.game.conditions, second.game.conditions)
    for row_a, row_b in zip(first.game.rows, second.game.rows):
        for value_a, value_b in ((row_a.target, row_b.target),
                                 (row_a.neutral, row_b.neutral),
                                 (row_a.row.latent, row_b.row.latent),
                                 (row_a.row.timestep, row_b.row.timestep)):
            assert torch.equal(value_a, value_b)
        assert row_a.row.scale == row_b.row.scale


@pytest.mark.parametrize("device,expected_index", [("cuda:0", 0), ("cuda:3", 3), ("cuda", 2)])
def test_mock_cuda_seeds_only_the_fork_owned_generator(monkeypatch, device, expected_index):
    # Stop immediately after RNG setup. All CUDA-facing objects below are mocks;
    # no CUDA tensor, runtime initialization, or real CUDA generator is used.
    seeds = [[] for _ in range(4)]
    generators = [SimpleNamespace(manual_seed=values.append) for values in seeds]
    monkeypatch.setattr(torch.cuda, "default_generators", generators)
    current_calls = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_calls.append(True) or 2)
    forks = []

    @contextmanager
    def mock_fork_rng(*, devices):
        forks.append(list(devices))
        before = torch.get_rng_state().clone()
        try:
            yield
        finally:
            torch.set_rng_state(before)

    monkeypatch.setattr(torch.random, "fork_rng", mock_fork_rng)
    observed = []

    class StopAfterSeed(Exception):
        pass

    def capture_seed(*args, **kwargs):
        observed.append(torch.get_rng_state().clone())
        raise StopAfterSeed

    monkeypatch.setattr(gan, "Slider", capture_seed)
    before = torch.get_rng_state().clone()
    config = dict(objective="pg09_gan_only", pg_version="0.9.1", seed=317, rank=2)
    pipe = SimpleNamespace(transformer=SimpleNamespace(device=torch.device(device)))
    with pytest.raises(StopAfterSeed):
        gan._build_run(pipe, (), config, checkpointing=False, model_identity=None, owner=[])
    assert forks == [[expected_index]]
    assert seeds == [[317] if index == expected_index else [] for index in range(4)]
    assert len(current_calls) == (1 if device == "cuda" else 0)
    expected_cpu = torch.Generator("cpu").manual_seed(317).get_state()
    assert len(observed) == 1 and torch.equal(observed[0], expected_cpu)
    assert torch.equal(before, torch.get_rng_state())


def test_late_failed_build_restores_rng_hooks_flags_and_allows_retry(monkeypatch):
    new_pipe, contexts, config = fixture()
    pipe = new_pipe()
    model = pipe.transformer
    flags = [p.requires_grad for p in model.parameters()]
    forwards = [block.forward for block in model.layers]
    before = torch.get_rng_state().clone()
    original_game = gan.ConditionalPG09

    def failure(*args, **kwargs):
        assert all(getattr(block, "_pg09_scale_checkpoint", False) for block in model.layers)
        raise RuntimeError("late game construction failure")

    monkeypatch.setattr(gan, "ConditionalPG09", failure)
    with pytest.raises(RuntimeError, match="late game construction failure"):
        gan.build_run(pipe, contexts, config)
    assert torch.equal(before, torch.get_rng_state())
    assert [p.requires_grad for p in model.parameters()] == flags
    assert [block.forward for block in model.layers] == forwards
    assert all(not getattr(block, "_pg09_scale_checkpoint", False) for block in model.layers)
    assert all(not getattr(module, "_moss_slider_attached", False) for module in model.modules())
    monkeypatch.setattr(gan, "ConditionalPG09", original_game)
    retried = gan.build_run(pipe, contexts, config)
    assert len(retried.game.rows) == 4
    assert torch.equal(before, torch.get_rng_state())
