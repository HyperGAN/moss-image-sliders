"""CPU receipt using the real vendored MOSS architecture at tiny dimensions."""

import tempfile
import torch

from .artifacts import load_slider, save_slider
from .lora import Slider
from .training import train_step
from .vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer


def tiny_model():
    return MOSSImageTransformer(dim=32, ffn_dim=48, num_heads=4, num_layers=2,
                                text_dim=16, freq_dim=16, latent_channels=4).eval()


def run_smoke(steps=4):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = tiny_model()
    latent = torch.randn(1, 4, 2, 2)
    contexts = [(torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool)) for _ in range(3)]
    original = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    slider = Slider(model, rank=2)
    optimizer = torch.optim.AdamW(slider.parameters(), lr=0.01)
    receipts = [train_step(model, slider, optimizer, latent, torch.tensor(500.), contexts)
                for _ in range(steps)]
    with tempfile.TemporaryDirectory() as directory:
        path = directory + "/slider"
        save_slider(slider, path, {"smoke": True})
        slider.detach()
        restored = load_slider(model, path)
        pipe = MOSSImagePipeline(model, MOSSImageScheduler())
        outputs = []
        for scale in [-1, 0, 1]:
            with restored.at(scale):
                result = pipe(prompt_embeds=contexts[0][0], prompt_attention_mask=contexts[0][1],
                              height=32, width=32, num_inference_steps=2, guidance_scale=1,
                              uncond_skip_layers=[], output_type="latent",
                              generator=torch.Generator("cpu").manual_seed(42)).images
                outputs.append(result)
        assert all(torch.isfinite(value).all() for value in outputs)
        assert not torch.equal(outputs[0], outputs[2])
        assert all(torch.equal(original[key], tensor) for key, tensor in model.state_dict().items())
    return {"backend": "official MOSSImageTransformer (tiny random weights)", "device": "cpu",
            "model_weights_downloaded": False, "optimizer_steps": steps,
            "losses": [value["loss"] for value in receipts],
            "ramp_scales": [-1, 0, 1], "checkpoint_roundtrip": True,
            "frozen_base_unchanged": True, "image_quality_validated": False}
