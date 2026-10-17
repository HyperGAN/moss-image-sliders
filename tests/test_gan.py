"""CPU execution regressions against the installed shared PG0.9 implementation."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from moss_sliders.cli import main, validate_config
from moss_sliders.gan import (
    build_cached_run, build_run, restore_checkpoint, save_checkpoint, train_to_budget,
)
from moss_sliders.lora import Slider
from moss_sliders.smoke import tiny_model
from moss_sliders.vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler


@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(1)
    torch.manual_seed(17)


def fixture():
    model = tiny_model()
    original = deepcopy(model.state_dict())
    contexts = [(torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool)) for _ in range(3)]
    config = dict(objective="pg09_gan_only", pg_version="0.9.1", rank=2, seed=17,
                  resolution=32, rollout_steps=2, strength=1., batch_size=2,
                  steps=8, lr=.001, critic_width=8)

    def make(**changes):
        model = tiny_model()
        model.load_state_dict(original)
        return build_run(MOSSImagePipeline(model, MOSSImageScheduler()), contexts,
                         dict(config, **changes))
    return make, original, config


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def test_shared_game_executes_gan_only_balanced_signed_rows_and_frozen_base(monkeypatch):
    make, original, _ = fixture()
    run = make()
    assert type(run.game).__module__ == "particle_sliders.pg09"
    assert run.game.recipe.name == "bispf_optimized"
    assert run.game.prior is None
    def forbidden(*args, **kwargs):
        raise AssertionError("No MSE/reconstruction loss may execute during GAN training")
    monkeypatch.setattr(torch.nn.functional, "mse_loss", forbidden)
    d_ids, g_ids = [], []
    before = deepcopy(run.slider.state_dict())
    for _ in range(2):
        metric = run.step()
        assert metric["g_loss"] == metric["g_adv"]
        d_ids += metric["d_rows"]
        g_ids += metric["g_rows"]
    assert sorted(d_ids) == sorted(g_ids) == list(range(4))
    assert any(not torch.equal(t, run.slider.state_dict()[k]) for k, t in before.items())
    for name, value in run.backend.transformer.state_dict().items():
        assert torch.equal(value, original[name])
    plus, minus = run.game.rows[:2]
    torch.testing.assert_close(plus.target + minus.target, 2 * plus.neutral)
    assert torch.equal(plus.row.latent, minus.row.latent)
    assert float(plus.row.timestep) == float(minus.row.timestep)
    assert run.game.conditions[:, 17].tolist() == [1., -1., 1., -1.]
    assert all(p.grad is None for p in run.backend.transformer.parameters())
    assert not torch.cuda.is_initialized()


def test_full_disk_resume_matches_losses_all_optimizers_streams_and_samples(tmp_path):
    make, _, _ = fixture()
    first = make()
    first.step()
    first.step()
    saved = save_checkpoint(first, tmp_path / "checkpoints")
    expected = first.step()
    expected_sample = first.game.evaluate(0, seed=93)
    second = make()
    assert restore_checkpoint(second, saved) == 2
    assert second.step() == expected
    equal(first.game.state_dict(), second.game.state_dict())
    assert torch.equal(expected_sample, second.game.evaluate(0, seed=93))
    with pytest.raises(FileExistsError):
        save_checkpoint(make_after_restore(make, saved), tmp_path / "checkpoints")
    assert not torch.cuda.is_initialized()


def make_after_restore(make, saved):
    run = make()
    restore_checkpoint(run, saved)
    return run


def test_resume_checks_checksum_recipe_and_rejects_legacy_before_mutation(tmp_path):
    make, _, _ = fixture()
    run = make()
    run.step()
    path = save_checkpoint(run, tmp_path)
    other = make(steps=9)
    before = deepcopy(other.game.state_dict())
    with pytest.raises(ValueError, match="mismatch"):
        restore_checkpoint(other, path)
    equal(before, other.game.state_dict())
    meta_path = path / "checkpoint.json"
    meta = json.loads(meta_path.read_text())
    meta["schema"] = "legacy-mse"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="full PG09 GAN"):
        restore_checkpoint(run, path)
    meta["schema"] = "moss-pg09-checkpoint-v1"
    meta_path.write_text(json.dumps(meta))
    with (path / "training.pt").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        restore_checkpoint(run, path)


def test_inference_export_cannot_be_laundered_into_a_full_training_checkpoint(tmp_path):
    make, _, _ = fixture()
    source = make()
    source.step()
    inference = make()
    inference.game.load_export_state_dict(source.game.export_state_dict())
    with pytest.raises(RuntimeError, match="inference export"):
        save_checkpoint(inference, tmp_path)
    assert list(tmp_path.glob("step-*")) == []
    assert list(tmp_path.glob(".pending-*")) == []


def test_partial_run_resume_keeps_original_budget_and_exports_actual_leaves(tmp_path):
    make, _, recipe = fixture()
    run = make()
    run.step()
    checkpoint = save_checkpoint(run, tmp_path / "first")
    other = make()
    train_to_budget(other, tmp_path / "resumed", recipe, resume=checkpoint, checkpoint_every=3)
    assert other.game.completed_steps == recipe["steps"]
    assert (tmp_path / "resumed/checkpoints/step-000008/training.pt").is_file()
    manifest = json.loads((tmp_path / "resumed/adapter/slider.json").read_text())
    assert manifest["objective"] == "pg09_gan_only"
    assert manifest["core_config"]["recipe"]["total_steps"] == recipe["steps"]
    assert manifest["completed_steps"] == recipe["steps"]
    assert manifest["training"]["initial_completed_steps"] == 1


@pytest.mark.parametrize("change", [{"batch_size": 1}, {"strength": 0.}, {"steps": True},
                                   {"objective": "mse"}, {"lr": float("nan")}])
def test_invalid_build_leaves_model_reusable(change):
    _, _, config = fixture()
    model = tiny_model()
    pipe = MOSSImagePipeline(model, MOSSImageScheduler())
    flags = [p.requires_grad for p in model.parameters()]
    context = (torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError):
        build_run(pipe, [context] * 3, dict(config, **change))
    assert [p.requires_grad for p in model.parameters()] == flags
    assert all(not getattr(b, "_pg09_scale_checkpoint", False) for b in model.layers)
    Slider(model).detach()


def test_cached_invalid_budget_rejects_before_reading_cache_or_attaching_hooks():
    model = tiny_model()
    flags = [p.requires_grad for p in model.parameters()]
    for settings in ({"batch_size": 1}, {"total_steps": 3841}, {"total_steps": True}):
        with pytest.raises(ValueError):
            build_cached_run(model, "nonexistent-cache", **settings)
    assert [p.requires_grad for p in model.parameters()] == flags
    Slider(model).detach()


def test_legacy_dispatchers_import_without_gpu_queries_or_process_launches(monkeypatch):
    import subprocess
    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy tools must never query a GPU or dispatch processes")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(torch.cuda, "mem_get_info", forbidden)
    for name in ("monitor_campaign", "cuda_campaign_smoke"):
        spec = importlib.util.spec_from_file_location(name, Path("tools") / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with pytest.raises(SystemExit, match="disabled"):
            module.main()


def test_cli_rejects_legacy_recipe_and_missing_resume_before_model_load(tmp_path):
    config = json.loads(Path("examples/warm-light.json").read_text())
    validate_config(config)
    config["objective"] = "mse"
    with pytest.raises(ValueError, match="GAN-only"):
        validate_config(config)
    with pytest.raises(FileExistsError):
        main(["train-campaign", "--cache", "missing", "--output", str(tmp_path)])
    with pytest.raises(FileNotFoundError, match="complete full GAN"):
        main(["train", "--config", "examples/warm-light.json", "--resume", "missing",
              "--output", str(tmp_path / "new-output")])


def test_campaign_cache_is_checked_before_any_model_load(tmp_path, monkeypatch):
    from moss_sliders import backend
    def forbidden(**kwargs):
        raise AssertionError("An invalid cache must fail before native model loading")
    monkeypatch.setattr(backend, "load_transformer", forbidden)
    with pytest.raises(FileNotFoundError):
        main(["train-campaign", "--cache", str(tmp_path / "missing-cache"),
              "--output", str(tmp_path / "new-output")])
