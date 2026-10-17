"""CPU regressions for the real shared response path and portable GAN weights."""

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import BatchEncoding

from particle_sliders.pg09_backends import MossResponseAdapter, checkpoint_with_adapter_scale
from moss_sliders import MODEL_ID, MODEL_REVISION
from moss_sliders.artifacts import load_slider, save_gan_slider, save_slider
from moss_sliders import backend
from moss_sliders.inference import render
from moss_sliders.lora import Slider
from moss_sliders.smoke import tiny_model
from moss_sliders.vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler


@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(1)
    torch.manual_seed(71)


def render_fixture():
    model = tiny_model()
    slider = Slider(model, rank=2)
    with torch.no_grad():
        for residual in slider.residuals:
            residual.up.weight.fill_(.03)
    contexts = {
        "hello": (torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool)),
        "": (torch.randn(1, 2, 16), torch.ones(1, 2, dtype=torch.bool)),
    }
    pipe = MOSSImagePipeline(model, MOSSImageScheduler())
    pipe.encode_prompt = lambda prompt: contexts[prompt]
    return model, slider, contexts, pipe


def test_render_executes_shared_cfg_with_frozen_negative_and_fresh_native_schedule(monkeypatch):
    model, slider, contexts, pipe = render_fixture()
    observed, shared_calls = [], []
    native = model.forward
    response = MossResponseAdapter._response

    def forward(latent, time, embeds, mask, **kwargs):
        observed.append((embeds is contexts[""][0], slider.scale, kwargs))
        return native(latent, time, embeds, mask, **kwargs)

    def shared(self, row, **kwargs):
        shared_calls.append(row)
        return response(self, row, **kwargs)

    monkeypatch.setattr(model, "forward", forward)
    monkeypatch.setattr(MossResponseAdapter, "_response", shared)
    original_config = deepcopy(dict(pipe.scheduler.config))
    result = render(pipe, slider, "hello", scale=-1, seed=7, steps=3,
                    width=48, height=32, output_type="latent")
    assert result.shape == (4, 2, 3) and torch.isfinite(result).all()
    assert len(shared_calls) == 1
    assert all(scale == (0 if negative else -1) for negative, scale, _ in observed)
    assert {negative for negative, _, _ in observed} == {False, True}
    assert all("skip_layers" not in kwargs for _, _, kwargs in observed)
    assert dict(pipe.scheduler.config) == original_config
    assert not hasattr(pipe.scheduler, "_step_index") or pipe.scheduler._step_index is None
    assert slider.scale == 0
    assert torch.equal(result, render(pipe, slider, "hello", scale=-1, seed=7,
                                     steps=3, width=48, height=32, output_type="latent"))
    assert not torch.cuda.is_initialized()


def test_render_zero_matches_adapter_free_native_pipeline_and_signs_use_loaded_weights():
    model, slider, contexts, pipe = render_fixture()
    kwargs = dict(seed=42, steps=3, width=32, height=32, output_type="latent")
    zero = render(pipe, slider, "hello", scale=0, **kwargs)
    plus = render(pipe, slider, "hello", scale=1, **kwargs)
    minus = render(pipe, slider, "hello", scale=-1, **kwargs)
    slider.detach()
    native = pipe(prompt_embeds=contexts["hello"][0], prompt_attention_mask=contexts["hello"][1],
                  negative_prompt_embeds=contexts[""][0], negative_prompt_attention_mask=contexts[""][1],
                  width=32, height=32, num_inference_steps=3, guidance_scale=3,
                  uncond_skip_layers=[], output_type="latent",
                  generator=torch.Generator("cpu").manual_seed(42)).images[0]
    assert torch.equal(zero, native)
    assert not torch.equal(plus, zero) and not torch.equal(minus, zero)
    assert not torch.equal(plus, minus)


def test_render_rejects_legacy_skipping_and_restores_scale_on_encoder_error():
    _, slider, _, pipe = render_fixture()
    with pytest.raises(ValueError, match="skipping disabled"):
        render(pipe, slider, "hello", skip_layer=True)
    pipe.encode_prompt = Mock(side_effect=RuntimeError("encoder error"))
    with pytest.raises(RuntimeError, match="encoder error"):
        render(pipe, slider, "hello", scale=-1)
    assert slider.scale == 0


def test_same_process_checkpointed_render_requires_owning_backend():
    model, slider, _, pipe = render_fixture()
    owning = MossResponseAdapter(model, slider)
    for block in model.layers:
        checkpoint_with_adapter_scale(block, owning.network)
    with pytest.raises(ValueError, match="training response_backend"):
        render(pipe, slider, "hello", width=32, height=32, output_type="latent")
    result = render(pipe, slider, "hello", scale=-1, steps=2, width=32, height=32,
                    output_type="latent", response_backend=owning)
    assert result.isfinite().all() and slider.scale == 0
    wrong = SimpleNamespace(transformer=object(), slider=slider)
    with pytest.raises(ValueError, match="same transformer and slider"):
        render(pipe, slider, "hello", response_backend=wrong)


def gan_artifact_fixture():
    from moss_sliders.gan import build_run
    model = tiny_model()
    contexts = [(torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool)) for _ in range(3)]
    config = dict(objective="pg09_gan_only", pg_version="0.9.1", rank=2, lr=.001,
                  seed=71, resolution=32, rollout_steps=2, strength=1.,
                  batch_size=2, steps=2, critic_width=8)
    run = build_run(MOSSImagePipeline(model, MOSSImageScheduler()), contexts, config,
                    checkpointing=False)
    run.step()
    return model, run


def test_portable_gan_roundtrip_uses_actual_shared_signature_and_weights(tmp_path):
    model, run = gan_artifact_fixture()
    config = run.game.export_state_dict()["config"]
    pipe = MOSSImagePipeline(model, MOSSImageScheduler())
    neutral = run.game.rows[0].row.neutral
    negative = run.game.rows[0].row.negative
    pipe.encode_prompt = lambda prompt: neutral if prompt else negative
    render_kwargs = dict(scale=1, steps=2, width=32, height=32, output_type="latent")
    expected_image = render(pipe, run.slider, "neutral", **render_kwargs)
    save_gan_slider(run.slider, tmp_path / "gan", {"source": "CPU regression"},
                    core_config=config, completed_steps=run.game.completed_steps)
    expected = {k: v.clone() for k, v in run.slider.state_dict().items()}
    run.slider.detach()
    restored = load_slider(model, tmp_path / "gan")
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in expected.items())
    assert all(p.dtype == torch.float32 for p in restored.parameters())
    metadata = json.loads((tmp_path / "gan/slider.json").read_text())
    assert metadata["objective"] == "pg09_gan_only"
    assert metadata["base_model"] == MODEL_ID and metadata["revision"] == MODEL_REVISION
    assert metadata["core_config"]["generator_arm"] == "adapter_gan"
    assert torch.equal(expected_image, render(pipe, restored, "neutral", **render_kwargs))
    restored.detach()


@pytest.mark.parametrize("failure", ["checksum", "objective", "dtype", "shape", "config"])
def test_invalid_gan_artifacts_fail_before_attachment(tmp_path, failure):
    model, run = gan_artifact_fixture()
    folder = tmp_path / "gan"
    save_gan_slider(run.slider, folder, {}, core_config=run.game.export_state_dict()["config"],
                    completed_steps=run.game.completed_steps)
    run.slider.detach()
    manifest = json.loads((folder / "slider.json").read_text())
    weights = folder / "adapter.safetensors"
    if failure == "checksum":
        with weights.open("ab") as stream:
            stream.write(b"altered")
    elif failure == "objective":
        manifest["objective"] = "mse"
    elif failure == "config":
        manifest["core_config"]["generator_arm"] = "legacy_mse"
    else:
        state = load_file(str(weights))
        key = next(iter(state))
        state[key] = state[key].half() if failure == "dtype" else state[key][:-1]
        save_file(state, str(weights))
        manifest["adapter_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    (folder / "slider.json").write_text(json.dumps(manifest))
    model.requires_grad_(True)
    before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    with pytest.raises((ValueError, RuntimeError)):
        load_slider(model, folder)
    assert before == {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    assert all(not getattr(module, "_moss_slider_attached", False) for module in model.modules())
    Slider(model, rank=2).detach()


def test_gan_save_refuses_legacy_recipe_before_creating_output(tmp_path):
    model = tiny_model()
    slider = Slider(model, rank=2)
    with pytest.raises(ValueError, match="shared ParticleGAN 0.9"):
        save_gan_slider(slider, tmp_path / "not-gan", {"objective": "mse"},
                        core_config={"objective": "mse"}, completed_steps=1)
    assert not (tmp_path / "not-gan").exists()


def test_legacy_v1_remains_readable_but_shared_worker_exports_need_verified_conversion(tmp_path):
    model = tiny_model()
    slider = Slider(model, rank=2)
    save_slider(slider, tmp_path / "legacy")
    slider.detach()
    load_slider(model, tmp_path / "legacy").detach()
    raw = tmp_path / "worker.safetensors"
    save_file({"weights.0": torch.ones(1)}, str(raw))
    with pytest.raises(ValueError, match="MossCampaignRun.load_adapter"):
        load_slider(model, raw)


def tokenizer_fixture(lengths):
    tokenizer = Mock()
    tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: messages[0]["content"]
    tokenizer.side_effect = lambda text, **kwargs: BatchEncoding(
        dict(input_ids=torch.ones(1, lengths[text], dtype=torch.long),
             attention_mask=torch.ones(1, lengths[text], dtype=torch.long)))
    pipe = SimpleNamespace(tokenizer=tokenizer, text_encoder=Mock(), transformer=Mock())
    pipe.text_encoder.side_effect = lambda **kwargs: SimpleNamespace(
        last_hidden_state=torch.ones(1, kwargs["input_ids"].shape[-1], 16))
    return pipe


def test_training_prompt_budget_rejects_all_before_any_forward():
    pipe = tokenizer_fixture({"short": 2, "long pole": 5})
    with pytest.raises(ValueError, match="5 tokens; budget is 4"):
        backend.encode_training_prompts(pipe, ["short", "long pole"], "cpu", max_length=4)
    pipe.text_encoder.assert_not_called()
    pipe.transformer.to.assert_not_called()
    assert all(call.kwargs["truncation"] is False for call in pipe.tokenizer.call_args_list)


def test_training_prompt_identity_records_full_tokenization(monkeypatch):
    pipe = tokenizer_fixture({"short": 2, "long pole": 4})
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    contexts = backend.encode_training_prompts(pipe, ["short", "long pole"], "cpu", max_length=4)
    assert [x.shape[1] for x, _ in contexts] == [2, 4]
    assert all(mask.dtype == torch.bool for _, mask in contexts)
    identity = pipe._moss_training_context_identity
    assert [entry["token_count"] for entry in identity] == [2, 4]
    assert all(not entry["truncated"] for entry in identity)
    assert identity[0]["prompt_sha256"] == hashlib.sha256(b"short").hexdigest()


def test_cached_transformer_loader_pins_revision_and_freezes_without_other_components(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cls = Mock()
    monkeypatch.setattr(backend, "MOSSImageTransformer", cls)
    backend.load_transformer(device="cuda:1")
    assert cls.from_pretrained.call_args.kwargs == dict(
        subfolder="transformer", torch_dtype=torch.bfloat16, revision=MODEL_REVISION,
        local_files_only=True)
    cls.from_pretrained.return_value.to.assert_called_once_with("cuda:1")
    cls.from_pretrained.return_value.to.return_value.eval.return_value.requires_grad_.assert_called_once_with(False)
