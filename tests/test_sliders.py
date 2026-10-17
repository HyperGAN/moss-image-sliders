import json
from pathlib import Path
import hashlib

import pytest
import torch

from moss_sliders import MODEL_REVISION
from moss_sliders.artifacts import load_slider, save_slider
from moss_sliders.cli import validate_config, main
from moss_sliders.lora import Slider
from moss_sliders.smoke import tiny_model, run_smoke
from moss_sliders.training import checkpoint_blocks, train_step, sample_training_state
from moss_sliders.vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler


@pytest.fixture(autouse=True)
def seed():
    torch.set_num_threads(1)
    torch.manual_seed(12)


def inputs():
    x = torch.randn(1, 4, 2, 2)
    contexts = [(torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool)) for _ in range(3)]
    return x, torch.tensor(500.), contexts


def test_scale_zero_exact_and_signed_residual():
    model = tiny_model()
    base = model.layers[0].proj['q']
    x = torch.randn(1, 3, 32)
    original = base(x).detach()
    slider = Slider(model, rank=2)
    assert len(slider.targets) == 8
    with torch.no_grad():
        for r in slider.residuals:
            r.up.weight.fill_(0.1)
    assert torch.equal(base(x), original)
    with slider.at(1):
        plus = base(x)
    with slider.at(-1):
        minus = base(x)
    assert torch.allclose(plus - original, original - minus, atol=1e-6)
    assert not torch.equal(plus, original)
    slider.detach()
    assert torch.equal(base(x), original)


def test_scale_restored_after_exception_and_invalid_scale():
    slider = Slider(tiny_model())
    with pytest.raises(RuntimeError), slider.at(-1):
        raise RuntimeError("abort render")
    assert slider.scale == 0
    for bad in [float('nan'), float('inf'), 4]:
        with pytest.raises(ValueError), slider.at(bad):
            pass


def test_legacy_mse_and_unsafe_checkpoint_wrappers_are_disabled():
    model = tiny_model()
    x, t, contexts = inputs()
    slider = Slider(model, rank=2)
    before = {k: v.clone() for k, v in slider.state_dict().items()}
    with pytest.raises(RuntimeError, match="Legacy MSE training is disabled"):
        train_step(model, slider, None, x, t, contexts)
    assert all(torch.equal(v, slider.state_dict()[k]) for k, v in before.items())
    with pytest.raises(ValueError, match="actual shared backend.network"):
        checkpoint_blocks(model)


def test_checkpoint_roundtrip_strict_and_no_overwrite(tmp_path):
    model = tiny_model()
    slider = Slider(model, rank=2)
    with torch.no_grad():
        slider.residuals[0].up.weight.fill_(0.02)
    save_slider(slider, tmp_path / 'adapter')
    with pytest.raises(FileExistsError):
        save_slider(slider, tmp_path / 'adapter')
    expected = {k: v.clone() for k, v in slider.state_dict().items()}
    slider.detach()
    restored = load_slider(model, tmp_path / 'adapter')
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in expected.items())
    restored.detach()
    manifest_path = tmp_path / 'adapter' / 'slider.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['revision'] = 'wrong-model'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mismatch"):
        load_slider(model, tmp_path / 'adapter')


def test_invalid_weights_detach_hooks(tmp_path):
    from safetensors.torch import save_file
    model = tiny_model()
    slider = Slider(model, rank=2)
    save_slider(slider, tmp_path / 'adapter')
    slider.detach()
    save_file({'bad': torch.ones(1)}, str(tmp_path / 'adapter' / 'adapter.safetensors'))
    with pytest.raises(RuntimeError):
        load_slider(model, tmp_path / 'adapter')
    Slider(model, rank=2)  # Failed load has left the base reusable.


def test_duplicate_and_target_mismatch_rejected():
    model = tiny_model()
    with pytest.raises(ValueError):
        Slider(model, targets=['not-a-moss-target'])
    Slider(model)
    with pytest.raises(ValueError, match="already attached"):
        Slider(model)


def test_partial_rollout_and_scheduler_reset():
    model = tiny_model()
    slider = Slider(model, rank=2)
    pipe = MOSSImagePipeline(model, MOSSImageScheduler())
    _, _, contexts = inputs()
    for i in [0, 1, 2, 3, 0]:
        x, t = sample_training_state(pipe, slider, contexts[0],
                                     torch.Generator('cpu').manual_seed(42), 32, 4, i)
        assert x.shape == (1, 4, 2, 2)
        assert x.dtype == torch.float32 and torch.isfinite(x).all()
        assert 0 < t <= 1000
    assert slider.scale == 0


def test_real_tiny_architecture_end_to_end():
    receipt = run_smoke(4)
    assert receipt['checkpoint_roundtrip'] and receipt['frozen_base_unchanged']
    assert len(receipt['losses']) == 4


def test_examples_and_config_guards():
    for path in Path('examples').glob('*.json'):
        config = validate_config(json.loads(path.read_text()))
        for key, value in [('steps', 0), ('rank', 0), ('resolution', 513),
                           ('lr', float('nan')), ('neutral', '')]:
            invalid = {**config, key: value}
            with pytest.raises(ValueError):
                validate_config(invalid)


def test_official_source_checksum():
    import moss_sliders.vendor.pipeline_moss_image as vendor
    assert hashlib.sha256(Path(vendor.__file__).read_bytes()).hexdigest() == (
        '1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31')
    assert len(MODEL_REVISION) == 40


def test_cli_smoke_bounds_and_collision(tmp_path):
    with pytest.raises(ValueError):
        main(['smoke', '--steps', '100'])
    with pytest.raises(FileExistsError):
        main(['train', '--config', 'missing.json', '--output', str(tmp_path)])
