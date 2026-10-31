"""Actual installed HNDL graphs and native tiny MOSS; no mocks or weights."""
from copy import deepcopy
from importlib.metadata import version
import json

import pytest
import torch

pytestmark = pytest.mark.skipif(version("particle-sliders-core") != "0.12.0",
                                reason="requires the separate HyperGAN/HNDL environment")


def make_fixture():
    from moss_sliders.native_hndl import build_hndl_run
    from moss_sliders.smoke import tiny_model
    from moss_sliders.vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler
    torch.set_num_threads(1)
    torch.manual_seed(131)
    original = deepcopy(tiny_model().state_dict())
    contexts = [(torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool)) for _ in range(3)]
    config = dict(objective="pg09_gan_only", pg_version="0.9.4", rank=2, seed=17,
                  resolution=32, rollout_steps=2, strength=1., batch_size=2,
                  steps=4, lr=.001, critic_width=8)

    def make(**changes):
        model = tiny_model()
        model.load_state_dict(original)
        return build_hndl_run(MOSSImagePipeline(model, MOSSImageScheduler()), contexts,
                              dict(config, **changes))
    return make, original


def test_real_graph_updates_own_parameters_and_preserves_native_field_and_base():
    from test_gan import equal
    make, original = make_fixture()
    run = make()
    before = deepcopy(run.slider.state_dict())
    critic_before = deepcopy(run.game.critic.state_dict())
    assert type(run.game).__module__ == "particle_sliders.pg09"
    assert type(run.game.critic.native.graph).__module__.startswith("hndl.")
    assert all(type(r._graph).__module__.startswith("hndl.") for r in run.slider.residuals)
    assert run.game.response_shape == (1, 4, 2, 2)
    assert run.game.prior is None
    assert run.game.export_state_dict()["config"]["generator_arm"] == "adapter_gan"
    assert run.game.critic.identity["hypergan_commit"] == "5f368a3770cf5bdae0448995c9f76e29af8049f7"
    g_ids = {id(p) for g in run.game.opt_g.param_groups for p in g["params"]}
    d_ids = {id(p) for g in run.game.opt_d.param_groups for p in g["params"]}
    assert g_ids == {id(p) for p in run.slider.parameters()}
    assert d_ids == {id(p) for p in run.game.critic.parameters()}
    assert not g_ids & d_ids
    metric = run.step()
    assert metric["g_loss"] == metric["g_adv"]
    assert any(not torch.equal(v, run.slider.state_dict()[k]) for k, v in before.items())
    assert any(not torch.equal(v, run.game.critic.state_dict()[k]) for k, v in critic_before.items())
    equal(original, run.backend.transformer.state_dict())
    assert all(p.grad is None for p in run.backend.transformer.parameters())
    assert not torch.cuda.is_initialized()


def test_hndl_attention_lora_matched_weight_forward_and_vjp():
    from moss_sliders.lora import LowRankResidual
    from moss_sliders.native_hndl import HNDLLowRankResidual
    torch.manual_seed(17)
    base = torch.nn.Linear(8, 12)
    original = LowRankResidual(base, 2, 2)
    actual = HNDLLowRankResidual(base, 2, 2)
    with torch.no_grad():
        original.up.weight.normal_()
    actual.load_state_dict(original.state_dict())
    value = torch.randn(2, 5, 8, requires_grad=True)
    expected = original(value)
    observed = actual(value)
    assert torch.equal(observed, expected)
    cotangent = torch.randn_like(expected)
    first = torch.autograd.grad(expected, (value, *original.parameters()), cotangent)
    second = torch.autograd.grad(observed, (value, *actual.parameters()), cotangent)
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_unregistered_graph_keeps_registered_state_and_conversion_contracts():
    from moss_sliders.native_hndl import HNDLLowRankResidual
    residual = HNDLLowRankResidual(torch.nn.Linear(8, 12), 2, 2)
    with torch.no_grad():
        residual.up.weight.normal_()
    state = deepcopy(residual.state_dict())
    assert set(state) == {"down.weight", "up.weight"}
    assert len(list(residual.parameters())) == 2
    assert {id(p) for p in residual.parameters()} == {id(p) for p in residual._graph.parameters()}
    assert list(residual._graph.buffers()) == []
    assert residual.down is residual._graph["down"] and residual.up is residual._graph["up"]
    value = torch.randn(2, 5, 8)
    expected = residual(value)
    cloned = deepcopy(residual).double().eval()
    assert cloned.down is cloned._graph["down"] and cloned.up is cloned._graph["up"]
    assert not cloned._graph.training
    assert all(p.dtype == torch.float64 for p in cloned.parameters())
    assert cloned(value).dtype == torch.float64
    torch.testing.assert_close(cloned(value).float(), expected)
    # Meta is a state-relocation check only; this is not CUDA qualification.
    cloned.to_empty(device="meta")
    assert cloned._graph._runtime_device == torch.device("meta")
    cloned.to_empty(device="cpu").float().train()
    cloned.load_state_dict(state)
    assert cloned._graph._runtime_device == torch.device("cpu")
    assert cloned._graph.training
    assert torch.equal(cloned(value), expected)


@pytest.mark.parametrize("identity_key,wrong", [("pg_version", "0.9.1"),
                                               ("pg_source_profile", "release"),
                                               ("penalty_input_domain", "continuous")])
def test_exact_hndl_checkpoint_restore_and_rejection_without_mutation(tmp_path, identity_key, wrong):
    from moss_sliders.gan import restore_checkpoint, save_checkpoint
    from test_gan import equal
    make, _ = make_fixture()
    run = make()
    run.step()
    saved = save_checkpoint(run, tmp_path)
    expected = run.step()
    receiver = make()
    assert restore_checkpoint(receiver, saved) == 1
    assert receiver.step() == expected
    equal(run.game.state_dict(), receiver.game.state_dict())
    packet = receiver.game.state_dict()
    assert packet["config"]["pg_version"] == "0.9.4"
    assert packet["config"]["pg_source_profile"] == "release-0.9.4"
    assert packet["config"]["backend"]["model_identity"]["network_runtime"]["core_commit"] == (
        "a794e3b4a8813b012f2c4faa3da82b78d4a90214")
    corrupt = deepcopy(packet)
    corrupt["config"][identity_key] = wrong
    with pytest.raises(ValueError):
        receiver.game.load_state_dict(corrupt)
    equal(packet, receiver.game.state_dict())


@pytest.mark.parametrize("field,wrong", [("pg_version", "0.9.1"),
                                        ("pg_source_profile", "release"),
                                        ("penalty_input_domain", "continuous"),
                                        ("pg_source_profile", None)])
def test_checkpoint_json_cohort_refuses_before_tensor_load(tmp_path, monkeypatch, field, wrong):
    from moss_sliders.gan import restore_checkpoint, save_checkpoint
    from test_gan import equal
    make, _ = make_fixture()
    receiver = make()
    before = receiver.game.state_dict()
    saved = save_checkpoint(receiver, tmp_path)
    metadata_path = saved / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["schema"] == "moss-pg09-checkpoint-v2"
    assert (metadata["pg_version"], metadata["pg_source_profile"],
            metadata["penalty_input_domain"]) == ("0.9.4", "release-0.9.4", "auto")
    metadata[field] = wrong
    metadata_path.write_text(json.dumps(metadata))

    def forbidden(*args, **kwargs):
        raise AssertionError("incompatible checkpoint header reached tensor load")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="cohort header"):
        restore_checkpoint(receiver, saved)
    equal(before, receiver.game.state_dict())


def test_legacy_header_refuses_before_tensor_load_in_preview(tmp_path, monkeypatch):
    from moss_sliders.gan import restore_checkpoint, save_checkpoint
    make, _ = make_fixture()
    receiver = make()
    saved = save_checkpoint(receiver, tmp_path)
    metadata_path = saved / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["schema"] = "moss-pg09-checkpoint-v1"
    for key in ("pg_version", "pg_source_profile", "penalty_input_domain"):
        del metadata[key]
    metadata_path.write_text(json.dumps(metadata))

    def forbidden(*args, **kwargs):
        raise AssertionError("unidentified legacy checkpoint reached tensor load")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="explicit PG/profile/domain header"):
        restore_checkpoint(receiver, saved)


def test_new_export_records_actual_cohort_and_zero_is_adapter_free(tmp_path):
    from moss_sliders.artifacts import load_slider, save_gan_slider
    from moss_sliders.training import velocity
    from test_gan import equal
    make, original = make_fixture()
    run = make()
    run.step()
    row = run.game.rows[0].row
    with run.slider.at(0), torch.no_grad():
        attached = velocity(run.backend.transformer, row.latent, row.timestep, row.neutral)
    path = tmp_path / "export"
    save_gan_slider(run.slider, path, {}, core_config=run.game.export_state_dict()["config"],
                    completed_steps=run.game.completed_steps)
    manifest = json.loads((path / "slider.json").read_text())
    assert manifest["pg_version"] == "0.9.4"
    assert manifest["core_config"]["pg_source_profile"] == "release-0.9.4"
    identity = manifest["core_config"]["backend"]["model_identity"]["network_runtime"]
    assert identity["profile"] == "hypergan-hndl-v2"
    state = deepcopy(run.slider.state_dict())
    run.slider.detach()
    with torch.no_grad():
        detached = velocity(run.backend.transformer, row.latent, row.timestep, row.neutral)
    assert torch.equal(attached, detached)
    equal(original, run.backend.transformer.state_dict())
    restored = load_slider(run.backend.transformer, path)
    equal(state, restored.state_dict())
    restored.detach()


def test_hndl_rejects_false_or_changed_geometry():
    from moss_sliders.native_hndl import MossHNDLFieldCritic
    with pytest.raises(ValueError):
        MossHNDLFieldCritic((1, 48000), 21, 8)
    with pytest.raises(ValueError):
        MossHNDLFieldCritic((1, 4, 2, 2, 2), 21, 8)
    critic = MossHNDLFieldCritic((1, 4, 1, 2, 2), 21, 8)
    response = torch.randn(2, 4, 1, 2, 2, requires_grad=True)
    score = critic(response, torch.randn(2, 21))
    assert score.shape == (2,)
    score.sum().backward()
    assert response.grad is not None and response.grad.isfinite().all()
    with pytest.raises(ValueError, match="geometry"):
        critic(torch.randn(2, 4, 2, 3), torch.randn(2, 21))


def test_field_critic_matched_weights_and_penalty_second_derivatives():
    from particle_sliders.field_critic import FieldCritic
    from moss_sliders.native_hndl import MossHNDLFieldCritic
    original = FieldCritic(4, 21, 8)
    actual = MossHNDLFieldCritic((1, 4, 3, 3), 21, 8)
    pairs = [(original.field[0], actual.native.graph["field0"]),
             (original.field[2], actual.native.graph["field2"]),
             (original.condition, actual.native.graph["condition"]),
             (original.score[0], actual.native.graph["score0"]),
             (original.score[2], actual.native.graph["score2"])]
    for source, target in pairs:
        target.load_state_dict(source.state_dict())
    x = torch.randn(2, 4, 3, 3, requires_grad=True)
    condition = torch.randn(2, 21)
    expected, observed = original(x, condition), actual(x, condition)
    torch.testing.assert_close(expected, observed)
    first = torch.autograd.grad(expected.sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(observed.sum(), x, create_graph=True)[0]
    torch.testing.assert_close(first, second)
    expected_parameters = [p for source, _ in pairs for p in source.parameters()]
    actual_parameters = [p for _, target in pairs for p in target.parameters()]
    left = torch.autograd.grad(first.square().sum(), expected_parameters, allow_unused=True)
    right = torch.autograd.grad(second.square().sum(), actual_parameters, allow_unused=True)
    for a, b in zip(left, right):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b)


def test_builder_and_core_source_tamper_refuse_before_native_attachment(tmp_path, monkeypatch):
    import hndl
    import particle_sliders
    from moss_sliders.native_hndl import build_hndl_run
    with monkeypatch.context() as patch:
        patch.setattr(hndl, "__file__", str(tmp_path / "hndl" / "__init__.py"))
        with pytest.raises(ValueError, match="HNDL source closure"):
            build_hndl_run(object(), (), {})
    with monkeypatch.context() as patch:
        patch.setattr(particle_sliders, "__file__", str(tmp_path / "core" / "__init__.py"))
        with pytest.raises(ValueError, match="core source closure"):
            build_hndl_run(object(), (), {})


@pytest.mark.parametrize("name", ["pg09.py", "fibo_critic.hndl", "pg094_release.json", "extra.py"])
def test_entire_core_code_graph_data_closure_rejects_changes(tmp_path, monkeypatch, name):
    from pathlib import Path
    import shutil
    import particle_sliders
    from moss_sliders.native_hndl import build_hndl_run
    root = Path(particle_sliders.__file__).parent
    copied = tmp_path / "particle_sliders"
    shutil.copytree(root, copied, ignore=shutil.ignore_patterns("__pycache__"))
    path = copied / name
    path.write_bytes((path.read_bytes() if path.exists() else b"") + b"\n ")
    monkeypatch.setattr(particle_sliders, "__file__", str(copied / "__init__.py"))
    with pytest.raises(ValueError, match="core source closure"):
        build_hndl_run(object(), (), {})


def test_failed_modern_build_rolls_back_and_same_pipeline_can_retry(monkeypatch):
    from moss_sliders import gan
    from moss_sliders.native_hndl import build_hndl_run
    from moss_sliders.smoke import tiny_model
    from moss_sliders.vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler
    torch.set_num_threads(1)
    model = tiny_model()
    pipe = MOSSImagePipeline(model, MOSSImageScheduler())
    contexts = [(torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool)) for _ in range(3)]
    config = dict(objective="pg09_gan_only", pg_version="0.9.4", rank=2, seed=17,
                  resolution=32, rollout_steps=2, strength=1., batch_size=2,
                  steps=4, lr=.001, critic_width=8)
    flags = [p.requires_grad for p in model.parameters()]
    forwards = [block.forward for block in model.layers]
    rng = torch.get_rng_state().clone()
    def failure(*args, **kwargs):
        raise RuntimeError("late construction failure")
    with monkeypatch.context() as patch:
        patch.setattr(gan, "ConditionalPG09", failure)
        with pytest.raises(RuntimeError, match="late construction failure"):
            build_hndl_run(pipe, contexts, config)
    assert [p.requires_grad for p in model.parameters()] == flags
    assert [block.forward for block in model.layers] == forwards
    assert torch.equal(rng, torch.get_rng_state())
    assert not any(getattr(m, "_moss_slider_attached", False) for m in model.modules())
    run = build_hndl_run(pipe, contexts, config)
    metric = run.step()
    assert run.game.completed_steps == 1 and torch.isfinite(torch.tensor(metric["g_loss"]))
