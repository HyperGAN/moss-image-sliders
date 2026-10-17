"""CPU receipt using the real vendored MOSS architecture at tiny dimensions."""

import tempfile
import torch

from .artifacts import load_slider
from .vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer


def tiny_model():
    return MOSSImageTransformer(dim=32, ffn_dim=48, num_heads=4, num_layers=2,
                                text_dim=16, freq_dim=16, latent_channels=4).eval()


def run_smoke(steps=4):
    from .gan import build_run, restore_checkpoint, save_checkpoint
    from .artifacts import save_gan_slider
    from particle_sliders.pg09_backends import FlowSchedule, ResponseRow
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = tiny_model()
    latent = torch.randn(1, 4, 2, 2)
    contexts = [(torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool)) for _ in range(3)]
    original = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    pipe = MOSSImagePipeline(model, MOSSImageScheduler())
    config = dict(objective="pg09_gan_only", pg_version="0.9.1", rank=2, seed=42,
                  resolution=32, rollout_steps=2, strength=1., batch_size=2,
                  steps=steps, lr=0.001, critic_width=8)
    run = build_run(pipe, contexts, config)
    receipts = [run.step() for _ in range(steps - 1)]
    with tempfile.TemporaryDirectory() as directory:
        path = directory + "/slider"
        checkpoint = save_checkpoint(run, directory + "/checkpoints")
        expected = run.step()
        receipts.append(expected)
        restore_checkpoint(run, checkpoint)
        assert run.step() == expected
        save_gan_slider(run.slider, path, {"smoke": True},
                        core_config=run.game.export_state_dict()["config"],
                        completed_steps=run.game.completed_steps)
        run.slider.detach()
        # Fresh unwrapped architecture, as used by CLI inference/reload.
        model = tiny_model()
        model.load_state_dict(original)
        restored = load_slider(model, path)
        from particle_sliders.pg09_backends import MossResponseAdapter
        backend = MossResponseAdapter(model, restored)
        outputs = []
        for scale in [-1, 0, 1]:
            row = ResponseRow(contexts[0], contexts[0], contexts[2], latent,
                              schedule=FlowSchedule.linear(2, time_scale=1000), guidance=3.)
            with torch.no_grad():
                outputs.append(backend._response(row, scale=scale))
        assert all(torch.isfinite(value).all() for value in outputs)
        assert not torch.equal(outputs[0], outputs[2])
        assert all(torch.equal(original[key], tensor) for key, tensor in model.state_dict().items())
    return {"backend": "official MOSSImageTransformer (tiny random weights)", "device": "cpu",
            "model_weights_downloaded": False, "optimizer_steps": steps,
            "objective": "pg09_gan_only", "pg_version": "0.9.1",
            "game": type(run.game).__module__ + "." + type(run.game).__name__,
            "losses": [value["g_loss"] for value in receipts],
            "ramp_scales": [-1, 0, 1], "checkpoint_roundtrip": True,
            "exact_full_game_resume": True,
            "replayed_validation_updates": 1,
            "frozen_base_unchanged": True, "image_quality_validated": False}
