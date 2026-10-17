from unittest.mock import Mock

import pytest
import torch

from moss_sliders import MODEL_REVISION
from moss_sliders import backend
from moss_sliders import inference
from moss_sliders.gan import build_run
from moss_sliders.inference import render
from moss_sliders.lora import Slider
from moss_sliders.smoke import tiny_model
from moss_sliders.ui import build_ui
from moss_sliders.vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler


def test_official_pipeline_cfg_and_repeatable_generator():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = tiny_model()
    slider = Slider(model, rank=2)
    with torch.no_grad():
        for residual in slider.residuals:
            residual.up.weight.fill_(0.03)
    pipe = MOSSImagePipeline(model, MOSSImageScheduler())
    kwargs = dict(prompt_embeds=torch.randn(1, 3, 16),
                  negative_prompt_embeds=torch.randn(1, 2, 16),
                  height=32, width=32, num_inference_steps=2,
                  guidance_scale=3, uncond_skip_layers=[1], output_type='latent')
    def generate(scale):
        with slider.at(scale):
            return pipe(**kwargs, generator=torch.Generator('cpu').manual_seed(42)).images
    first = generate(1)
    assert torch.equal(first, generate(1))
    zero = generate(0)
    assert not torch.equal(first, zero)
    slider.detach()
    assert torch.equal(zero, pipe(**kwargs, generator=torch.Generator('cpu').manual_seed(42)).images)


def test_loader_pins_all_components_and_avoids_remote_code(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    calls = {}
    for name in ['MOSSImageTransformer', 'AutoTokenizer', 'AutoModel', 'AutoencoderKLFlux2']:
        mock = Mock()
        calls[name] = mock.from_pretrained
        monkeypatch.setattr(backend, name, mock)
    pipe = backend.load_pipeline(local_files_only=True, training=True)
    assert pipe.vae is None
    calls['AutoencoderKLFlux2'].assert_not_called()
    for name in ['MOSSImageTransformer', 'AutoTokenizer', 'AutoModel']:
        kwargs = calls[name].call_args.kwargs
        assert kwargs['revision'] == MODEL_REVISION
        assert kwargs['local_files_only'] is True
        assert not kwargs.get('trust_remote_code')
    assert calls['AutoTokenizer'].call_args.kwargs['tokenizer_type'] == 'qwen2'


def test_cpu_real_model_load_refused_before_download(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA'):
        backend.load_pipeline(local_files_only=False)


def test_render_controls_forwarded_and_scale_reset(monkeypatch):
    model = tiny_model()
    slider = Slider(model)
    pipe = MOSSImagePipeline(model, MOSSImageScheduler())
    pipe.tokenizer = Mock()
    pipe.tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: messages[0]['content']
    pipe.tokenizer.return_value = dict(input_ids=torch.ones(1, 3, dtype=torch.long),
                                      attention_mask=torch.ones(1, 3, dtype=torch.long))
    pipe.encode_prompt = Mock(return_value=(torch.randn(1, 3, 16),
                                           torch.ones(1, 3, dtype=torch.bool)))
    captured = []
    shared_render = inference.MossResponseAdapter.render

    def record(adapter, row, *, scale):
        captured.append((row, scale))
        return shared_render(adapter, row, scale=scale)

    monkeypatch.setattr(inference.MossResponseAdapter, 'render', record)
    decoder = Mock(return_value=lambda latent: ['image'])
    monkeypatch.setattr(inference, 'moss_pipeline_decoder', decoder)
    assert render(pipe, slider, 'hello', scale=-0.5, seed=7, steps=5,
                  width=32, height=32) == 'image'
    row, scale = captured[0]
    assert scale == -.5 and row.guidance == 3
    assert len(row.schedule.timesteps) == 5
    assert torch.equal(row.latent, torch.randn(row.latent.shape,
                       generator=torch.Generator('cpu').manual_seed(7)))
    assert pipe.encode_prompt.call_args_list == [(("hello",), {'max_sequence_length': 4096}),
                                                (("",), {'max_sequence_length': 4096})]
    decoder.assert_called_once_with(pipe, output_type='pil')
    assert slider.scale == 0
    with pytest.raises(ValueError):
        render(pipe, slider, 'hello', width=513)


def test_gradio_controls_build_offline():
    slider = Slider(tiny_model())
    demo = build_ui(Mock(), slider)
    config = demo.get_config_file()
    labels = {component['props'].get('label') for component in config['components']}
    assert {'Concept strength', 'Seed', 'Width', 'Height', 'Sampling steps', 'CFG'} <= labels


def test_bfloat16_base_with_float32_adapter_training():
    torch.set_num_threads(1)
    model = tiny_model().to(dtype=torch.bfloat16)
    frozen = {k: v.clone() for k, v in model.state_dict().items()}
    contexts = [(torch.randn(1, 3, 16).bfloat16(), torch.ones(1, 3, dtype=torch.bool))
                for _ in range(3)]
    config = dict(objective='pg09_gan_only', pg_version='0.9.1', rank=2, lr=.001,
                  seed=71, resolution=32, rollout_steps=2, strength=1.,
                  batch_size=2, steps=2, critic_width=8)
    run = build_run(MOSSImagePipeline(model, MOSSImageScheduler()), contexts, config)
    before = {k: v.clone() for k, v in run.slider.state_dict().items()}
    result = run.step()
    assert all(p.dtype == torch.float32 for p in run.slider.parameters())
    assert result['g_loss'] == result['g_adv']
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in run.slider.parameters())
    assert any(not torch.equal(before[k], v) for k, v in run.slider.state_dict().items())
    assert all(torch.equal(frozen[k], v) for k, v in model.state_dict().items())
