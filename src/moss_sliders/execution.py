"""Product-owned factory/worker for the shared isolated training dispatcher.

The shared core owns every GAN update. This module owns model admission and
configuration, and labels random CPU architecture fixtures explicitly.
"""
import argparse
import json
from pathlib import Path


def profile():
    return dict(schema="particle-slider-profile-v1", name="moss",
                product="moss-image-sliders", config_kind="native", status="source_only",
                reason="CPU software conformance only; pretrained CUDA/BF16/offload and media quality remain unqualified",
                command=["{python}", "-m", "moss_sliders.execution"], runtime={},
                target_steps_path="steps", default_checkpoint_every=32,
                capabilities=dict(check=True, resume=True, periodic_checkpoint=True, device=True))


def fixture_profile():
    value = profile()
    value.update(name="moss-cpu-fixture", status="fixture",
                 reason="Tiny random native MOSS architecture; no pretrained model or image quality qualification",
                 command=["{python}", "-m", "moss_sliders.execution", "--fixture"],
                 runtime=dict(device="cpu"))
    return value


def validate_training(value, *, fixture=False):
    from .cli import validate_config
    value = dict(value)
    if fixture and value.pop("dummy", None) is not True:
        raise ValueError("CPU fixture requires explicit dummy=true")
    config = validate_config(value, fixture=fixture)
    if fixture and (config["steps"] > 20 or config["resolution"] > 64 or
                    config["rank"] > 4 or config["rollout_steps"] > 4 or
                    config["batch_size"] > 4):
        raise ValueError("CPU fixture exceeds its bounded tiny-architecture budget")
    return config


def build_training(config, *, fixture=False, device=None):
    """Build native product components; never substitute a fixture for a model."""
    from .native_hndl import build_hndl_run
    if fixture:
        if device not in (None, "cpu"):
            raise ValueError("CPU fixture requires device=cpu")
        import torch
        from .smoke import tiny_model
        from .vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config["seed"])
            pipe = MOSSImagePipeline(tiny_model(), MOSSImageScheduler())
            contexts = [(torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool))
                        for _ in range(3)]
    else:
        from .backend import encode_training_prompts, load_pipeline
        device = device or "cuda:0"
        pipe = load_pipeline(device=device, training=True, local_files_only=True)
        contexts = encode_training_prompts(
            pipe, [config[key] for key in ("neutral", "positive", "negative")],
            device, config["max_sequence_length"])
    return build_hndl_run(pipe, contexts, config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    config = validate_training(json.loads(args.config.read_text()), fixture=args.fixture)
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be positive")
    if args.fixture and args.device not in (None, "cpu"):
        raise ValueError("CPU fixture requires device=cpu")
    if not args.fixture and args.device and not args.device.startswith("cuda"):
        raise ValueError("Real MOSS runs require a CUDA device")
    if args.resume and not all((args.resume / name).is_file()
                               for name in ("checkpoint.json", "training.pt")):
        raise FileNotFoundError("A complete full GAN checkpoint is required for resume")
    from .native_hndl import verify_runtime
    identity = verify_runtime()
    qualification = "fixture" if args.fixture else "source_only"
    if args.check:
        print(json.dumps(dict(status="checked", qualification=qualification,
                              model_loaded=False, runtime=identity)))
        return
    from .gan import train_to_budget
    if args.output.exists():
        raise FileExistsError(args.output)
    run = build_training(config, fixture=args.fixture, device=args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        training = args.output / "training"
        train_to_budget(run, training, dict(config, dummy=args.fixture), resume=args.resume,
                        checkpoint_every=args.checkpoint_every)
        checkpoint = training / "checkpoints" / f"step-{run.game.completed_steps:06d}"
        if not checkpoint.exists() and args.resume is not None:
            checkpoint = args.resume
        result = dict(status="completed", qualification=qualification,
                      completed_steps=run.game.completed_steps, dummy=args.fixture,
                      checkpoint=str(checkpoint.resolve()),
                      adapter=str((training / "adapter").resolve()))
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        run.slider.detach()


if __name__ == "__main__":
    main()
