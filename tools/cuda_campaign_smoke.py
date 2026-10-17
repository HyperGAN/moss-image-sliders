"""Bounded real CUDA/BF16/checkpoint/optimizer-resume check with tiny random MOSS."""

import copy
import json
import tempfile

import torch

from moss_sliders.campaign import (
    StateSampler,
    cached_backward,
    restore_training_checkpoint,
    save_training_checkpoint,
    target_bundle,
)
from moss_sliders.lora import Slider
from moss_sliders.smoke import tiny_model
from moss_sliders.training import checkpoint_blocks
from moss_sliders.vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler

torch.set_num_threads(2)
torch.manual_seed(31)
model = tiny_model().to(device="cuda:0", dtype=torch.bfloat16)
slider = Slider(model, rank=2)
assert all(p.dtype == torch.float32 for p in slider.parameters())
contexts = [
    (
        torch.randn(1, 3, 16, device="cuda:0", dtype=torch.bfloat16),
        torch.ones(1, 3, device="cuda:0", dtype=torch.bool),
    )
    for _ in range(2)
]
pipe = MOSSImagePipeline(model, MOSSImageScheduler())
bundle = target_bundle(pipe, slider, *contexts, seed=7, resolution=32, steps=4)
sample = {k: bundle[k][0, 2].to("cuda:0") for k in ("latent", "base", "delta")}
sample["time"] = bundle["time"][2].to("cuda:0")
checkpoint_blocks(model)
optimizer = torch.optim.AdamW(slider.parameters(), lr=0.003)
sampler = StateSampler(32, 7)


def update():
    sampler.take(8)
    optimizer.zero_grad(set_to_none=True)
    for _ in range(8):
        cached_backward(model, slider, sample, contexts[0], divisor=8)
    norm = torch.nn.utils.clip_grad_norm_(slider.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


update()
with tempfile.TemporaryDirectory() as directory:
    saved = save_training_checkpoint(directory, 1, slider, optimizer, sampler, "cuda-smoke")
    update()
    expected = copy.deepcopy(slider.state_dict())
    restore_training_checkpoint(saved, slider, optimizer, sampler, "cuda-smoke")
    update()
    assert all(torch.equal(v, slider.state_dict()[k]) for k, v in expected.items())
assert all(p.grad is None for p in model.parameters())
print(
    json.dumps(
        dict(
            passed=True,
            device=torch.cuda.get_device_name(),
            bf16=True,
            checkpointing=True,
            effective_batch=8,
            exact_optimizer_resume=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
    )
)
