"""CPU-only reproduction from the immutable pre-audit source, not current code."""
import hashlib
import json
import subprocess
import types

import torch

from moss_sliders.lora import Slider
from moss_sliders.smoke import tiny_model

BASE = "0aa3867965aad20f3ee653dcf14e49a1fbca06d9"


def original(name):
    source = subprocess.check_output(["git", "show", f"{BASE}:src/moss_sliders/{name}.py"])
    module = types.ModuleType("legacy_" + name)
    exec(compile(source, "legacy_" + name + ".py", "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    old_training, training_sha = original("training")
    old_inference, inference_sha = original("inference")
    model = tiny_model()
    slider = Slider(model, rank=2)
    with torch.no_grad():
        for residual in slider.residuals:
            residual.up.weight.fill_(.03)
    latent = torch.randn(1, 4, 2, 2)
    context = (torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool))
    old_training.checkpoint_blocks(model)
    values = []
    for sign in (1, -1):
        with slider.at(sign):
            values.append(model(latent, torch.tensor(500.), *context).sample)
    try:
        sum(value.square().mean() for value in values).backward()
    except torch.utils.checkpoint.CheckpointError as exc:
        checkpoint = dict(confirmed=True, error=type(exc).__name__, message=str(exc))
    else:
        raise AssertionError("The immutable legacy wrapper unexpectedly passed delayed backward")

    scales = []
    class Probe:
        def __call__(self, *args, **kwargs):
            # The outer slider scope is the scope seen by both native CFG forwards.
            scales.extend([slider.scale, slider.scale])
            self.skip = kwargs["uncond_skip_layers"]
            return types.SimpleNamespace(images=["scope probe"])
    probe = Probe()
    old_inference.render(probe, slider, "neutral", scale=1.)
    assert scales == [1., 1.] and probe.skip == [22] and slider.scale == 0
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(
        immutable_source=BASE, training_sha256=training_sha, inference_sha256=inference_sha,
        delayed_signed_backward=checkpoint,
        legacy_cfg_scope=dict(conditional_adapter_scale=scales[0], negative_adapter_scale=scales[1],
                              negative_skip=probe.skip, required_negative_scale=0.,
                              required_negative_skip=[]),
        cuda_initialized=False, pretrained_weights_loaded=False,
    ), indent=2))


if __name__ == "__main__":
    main()
