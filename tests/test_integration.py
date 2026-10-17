from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from moss_sliders import MODEL_REVISION
from moss_sliders import backend
from moss_sliders.inference import render
from moss_sliders.lora import Slider
from moss_sliders.smoke import tiny_model
from moss_sliders.training import train_step
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


def test_cpu_real_model_load_refused_before_download(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA'):
        backend.load_pipeline(local_files_only=False)


def test_render_controls_forwarded_and_scale_reset():
    slider = Slider(tiny_model())
    pipe = Mock(return_value=SimpleNamespace(images=['image']))
    assert render(pipe, slider, 'hello', scale=-0.5, seed=7, steps=5) == 'image'
    kwargs = pipe.call_args.kwargs
    assert kwargs['generator'].initial_seed() == 7
    assert kwargs['uncond_skip_layers'] == [22]
    assert kwargs['num_inference_steps'] == 5
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
    slider = Slider(model, rank=2)
    assert all(p.dtype == torch.float32 for p in slider.parameters())
    contexts = [(torch.randn(1, 3, 16).bfloat16(), torch.ones(1, 3, dtype=torch.bool))
                for _ in range(3)]
    result = train_step(model, slider, torch.optim.AdamW(slider.parameters(), lr=0.001),
                        torch.randn(1, 4, 2, 2), torch.tensor(500.), contexts)
    assert result['grad_norm'] > 0
