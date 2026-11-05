"""Current adapter GAN optimizer rates and exact replay through public config."""
from copy import deepcopy

import pytest
import torch

from moss_sliders.cli import parser, validate_config
from moss_sliders.execution import build_training, validate_training
from moss_sliders.gan import restore_checkpoint, save_checkpoint
from moss_sliders.learning_rates import recipe_overrides


def config(**changes):
    return dict(neutral='neutral scene', positive='warm scene', negative='cool scene',
                objective='pg09_gan_only', pg_version='0.9.5', preset='bispf_optimized',
                rank=2, seed=17, resolution=32, rollout_steps=2, strength=1.,
                batch_size=2, max_sequence_length=32, steps=4, lr=1e-4, **changes)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def test_actual_default_and_independent_critic_rates():
    for multiplier in (None, 100.):
        value = config(**({} if multiplier is None else {'d_lr_mult': multiplier}))
        run = build_training(validate_training(dict(value, dummy=True), fixture=True), fixture=True)
        try:
            assert run.game.recipe.lr == 1e-4
            assert run.game.recipe.d_lr_mult == (1.5 if multiplier is None else multiplier)
            assert run.game.prior is None
            assert run.game.base_lrs[0] == [1e-4]
            assert run.game.base_lrs[1]
            assert all(rate == 1e-4 * (1.5 if multiplier is None else multiplier)
                       for rate in run.game.base_lrs[1])
            metric = run.step()
            assert metric['step'] == 1
            assert run.game.opt_g.param_groups[0]['lr'] == 1e-4
            assert all(group['lr'] == 1e-4 * (1.5 if multiplier is None else multiplier)
                       for group in run.game.opt_d.param_groups)
        finally:
            run.slider.detach()


def test_adjusted_rates_resume_exactly_and_changed_multiplier_refuses(tmp_path):
    value = config(d_lr_mult=100.)
    first = build_training(value, fixture=True)
    try:
        first.step()
        saved = save_checkpoint(first, tmp_path)
        expected = first.step()
        second = build_training(value, fixture=True)
        try:
            restore_checkpoint(second, saved)
            assert second.step() == expected
            equal(first.game.state_dict(), second.game.state_dict())
        finally:
            second.slider.detach()
        wrong = build_training(dict(value, d_lr_mult=1.5), fixture=True)
        try:
            before = deepcopy(wrong.game.state_dict())
            with pytest.raises(ValueError, match='mismatch'):
                restore_checkpoint(wrong, saved)
            equal(before, wrong.game.state_dict())
        finally:
            wrong.slider.detach()
    finally:
        first.slider.detach()


@pytest.mark.parametrize('value', [0, -1, True, False, None, '100', float('inf'), float('nan')])
def test_critic_multiplier_refuses_invalid_before_native_load(value):
    with pytest.raises(ValueError, match='finite positive'):
        validate_config(config(d_lr_mult=value), fixture=True)


def test_adapter_rejects_bank_rate_and_cli_exposes_same_names():
    with pytest.raises(ValueError, match='particle bank'):
        recipe_overrides(config(prior_lr_mult=2.5))
    with pytest.raises(ValueError, match='Expected config keys'):
        validate_config(config(prior_lr_mult=2.5), fixture=True)
    args = parser().parse_args(['train', '--config', 'config.json', '--output', 'new-run',
                               '--lr', '0.00001', '--d-lr-mult', '100'])
    assert args.lr == 1e-5 and args.d_lr_mult == 100.
    assert recipe_overrides(config()) == {'lr': 1e-4}
