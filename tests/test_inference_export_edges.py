"""CPU public-API checks for prompt bounds and transactional GAN export."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from moss_sliders import artifacts
from moss_sliders.artifacts import load_slider, save_gan_slider
from moss_sliders.gan import build_run
from moss_sliders.inference import render
from moss_sliders.lora import Slider
from moss_sliders.smoke import tiny_model
from moss_sliders.vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler


@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(1)


def inference_fixture():
    model = tiny_model()
    slider = Slider(model, rank=2)
    pipe = MOSSImagePipeline(model, MOSSImageScheduler())
    pipe.tokenizer = Mock()
    pipe.tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: (
        "USER " + messages[0]["content"] + " ASSISTANT")
    pipe.tokenizer.side_effect = lambda text, **kwargs: dict(
        input_ids=torch.ones(1, len(text.split()), dtype=torch.long),
        attention_mask=torch.ones(1, len(text.split()), dtype=torch.long))
    # The native transformer still executes on small contexts. The full-token
    # preflight is independent of the encoder fixture's deliberately tiny output.
    pipe.encode_prompt = Mock(return_value=(torch.ones(1, 3, 16),
                                           torch.ones(1, 3, dtype=torch.bool)))
    return pipe, slider


def test_inference_rejects_4097_full_template_tokens_before_any_encoder_call():
    pipe, slider = inference_fixture()
    with pytest.raises(ValueError, match="4097 tokens; budget is 4096"):
        render(pipe, slider, "word " * 4095, width=32, height=32, output_type="latent")
    pipe.encode_prompt.assert_not_called()
    assert pipe.tokenizer.call_args.kwargs == dict(return_tensors="pt", truncation=False)
    assert pipe.tokenizer.apply_chat_template.call_args.kwargs == dict(
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    assert slider.scale == 0
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("budget", [4, 4096])
def test_inference_exact_boundary_passes_explicit_budget_without_truncation(budget):
    pipe, slider = inference_fixture()
    prompt = "word " * (budget - 2)
    result = render(pipe, slider, prompt, width=32, height=32, steps=2,
                    output_type="latent", max_sequence_length=budget)
    assert result.shape == (4, 2, 2) and result.isfinite().all()
    assert pipe.encode_prompt.call_args_list == [
        ((prompt,), {"max_sequence_length": budget}),
        (("",), {"max_sequence_length": budget}),
    ]
    assert all(call.kwargs == dict(return_tensors="pt", truncation=False)
               for call in pipe.tokenizer.call_args_list)
    assert len(pipe.tokenizer.call_args_list) == 2


@pytest.mark.parametrize("budget", [0, 4097, True, 4.5])
def test_inference_invalid_budget_is_rejected_before_encoding(budget):
    pipe, slider = inference_fixture()
    with pytest.raises(ValueError, match="max_sequence_length must be an integer"):
        render(pipe, slider, "prompt", max_sequence_length=budget)
    pipe.encode_prompt.assert_not_called()


def test_inference_checks_empty_negative_budget_before_encoding_either_branch():
    pipe, slider = inference_fixture()
    with pytest.raises(ValueError, match="2 tokens; budget is 1"):
        render(pipe, slider, "", max_sequence_length=1)
    pipe.encode_prompt.assert_not_called()


@pytest.fixture
def gan_export():
    model = tiny_model()
    contexts = [(torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool)) for _ in range(3)]
    config = dict(objective="pg09_gan_only", pg_version="0.9.1", rank=2, lr=.001,
                  seed=71, resolution=32, rollout_steps=2, strength=1.,
                  batch_size=2, steps=2, critic_width=8)
    run = build_run(MOSSImagePipeline(model, MOSSImageScheduler()), contexts, config,
                    checkpointing=False)
    return model, run.slider, run.game.export_state_dict()["config"]


def save_fixture(export, folder, training=None):
    _, slider, config = export
    save_gan_slider(slider, folder, training or {}, core_config=config, completed_steps=0)


def assert_valid_retry(export, folder):
    save_fixture(export, folder, {"valid": True})
    assert {path.name for path in folder.iterdir()} == {"adapter.safetensors", "slider.json"}
    model, slider, _ = export
    expected = {key: value.clone() for key, value in slider.state_dict().items()}
    slider.detach()
    loaded = load_slider(model, folder)
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in expected.items())
    loaded.detach()


def test_nan_provenance_fails_before_any_output_and_valid_retry_succeeds(tmp_path, gan_export):
    folder = tmp_path / "new-parent" / "adapter"
    with pytest.raises(ValueError, match="Out of range float values"):
        save_fixture(gan_export, folder, {"nested": {"diagnostic": float("nan")}})
    assert not folder.parent.exists()
    assert_valid_retry(gan_export, folder)
    assert not list(folder.parent.glob(".*.pending-*"))


def test_staged_weight_write_failure_cleans_partial_files_and_allows_retry(
        tmp_path, gan_export, monkeypatch):
    folder = tmp_path / "adapter"
    native_save = artifacts.save_file

    def fail_after_partial_write(state, path, **kwargs):
        Path(path).write_bytes(b"incomplete weights")
        raise OSError("simulated safetensors failure")

    monkeypatch.setattr(artifacts, "save_file", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated safetensors failure"):
        save_fixture(gan_export, folder)
    assert not folder.exists() and list(tmp_path.iterdir()) == []
    monkeypatch.setattr(artifacts, "save_file", native_save)
    assert_valid_retry(gan_export, folder)


def test_manifest_publication_failure_rolls_back_only_reserved_output_and_allows_retry(
        tmp_path, gan_export, monkeypatch):
    folder = tmp_path / "adapter"
    native_replace = Path.replace

    def fail_manifest(source, target):
        if source.name == "slider.json":
            assert (folder / "adapter.safetensors").is_file()
            assert not (folder / "slider.json").exists()
            with pytest.raises(ValueError, match="portable MOSS"):
                load_slider(gan_export[0], folder)
            raise OSError("simulated manifest publication failure")
        return native_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_manifest)
    with pytest.raises(OSError, match="simulated manifest publication failure"):
        save_fixture(gan_export, folder)
    assert not folder.exists() and list(tmp_path.iterdir()) == []
    monkeypatch.setattr(Path, "replace", native_replace)
    assert_valid_retry(gan_export, folder)


@pytest.mark.parametrize("nonempty", [False, True])
def test_existing_export_directory_is_never_clobbered(tmp_path, gan_export, nonempty):
    folder = tmp_path / "adapter"
    folder.mkdir()
    if nonempty:
        (folder / "existing").write_bytes(b"keep me")
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    with pytest.raises(FileExistsError):
        save_fixture(gan_export, folder)
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before
    assert list(tmp_path.iterdir()) == [folder]


def test_export_destination_created_during_staging_is_preserved(tmp_path, gan_export, monkeypatch):
    folder = tmp_path / "adapter"
    native_save = artifacts.save_file

    def competing_directory(state, path, **kwargs):
        native_save(state, path, **kwargs)
        folder.mkdir()
        (folder / "other-writer").write_bytes(b"preserve competing writer")

    monkeypatch.setattr(artifacts, "save_file", competing_directory)
    with pytest.raises(FileExistsError):
        save_fixture(gan_export, folder)
    assert (folder / "other-writer").read_bytes() == b"preserve competing writer"
    assert list(tmp_path.iterdir()) == [folder]
