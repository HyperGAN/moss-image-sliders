"""Explicit small smoke tests and opt-in real model execution."""

import argparse
import json
import math
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(description="MOSS-Image signed LoRA concept sliders")
    commands = root.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="CPU, tiny real architecture, no downloads")
    smoke.add_argument("--steps", type=int, default=4)
    smoke.add_argument("--receipt", type=Path)
    commands.add_parser("doctor", help="Report CUDA and NVMe availability; makes no changes")
    for name in ("train", "infer", "ui"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--device", default="cuda:0")
        cmd.add_argument("--download", action="store_true", help="Allow pinned model downloads (~28 GB)")
        if name == "train":
            cmd.add_argument("--config", type=Path, required=True)
            cmd.add_argument("--output", type=Path, required=True)
        else:
            cmd.add_argument("--adapter", type=Path, required=True)
            cmd.add_argument("--offload", action="store_true")
            if name == "infer":
                cmd.add_argument("--prompt", required=True)
                cmd.add_argument("--scales", type=float, nargs="+", default=[-1, 0, 1])
                cmd.add_argument("--seed", type=int, default=42)
                cmd.add_argument("--width", type=int, default=1024)
                cmd.add_argument("--height", type=int, default=1024)
                cmd.add_argument("--steps", type=int, default=50)
                cmd.add_argument("--guidance", type=float, default=3)
                cmd.add_argument("--output", type=Path, required=True)
    return root


def validate_config(config):
    required = {"neutral", "positive", "negative", "steps", "rank", "lr", "resolution",
                "rollout_steps", "strength", "seed", "max_sequence_length"}
    if set(config) != required:
        raise ValueError(f"Expected config keys: {sorted(required)}")
    for key in ("neutral", "positive", "negative"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be nonempty text")
    for key, lower, upper in [("steps", 1, 10000), ("rank", 1, 128),
                              ("rollout_steps", 2, 50), ("resolution", 256, 2048),
                              ("max_sequence_length", 1, 4096), ("seed", 0, 2**32 - 1)]:
        if type(config[key]) is not int or not lower <= config[key] <= upper:
            raise ValueError(f"{key} must be an integer in [{lower}, {upper}]")
    if config["resolution"] % 16:
        raise ValueError("resolution must be a multiple of 16")
    for key, upper in [("strength", 10), ("lr", 0.1)]:
        if not math.isfinite(config[key]) or not 0 < config[key] <= upper:
            raise ValueError(f"{key} must be positive and <= {upper}")
    return config


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "doctor":
        import os
        import shutil
        import torch
        report = {"cuda_available": torch.cuda.is_available(), "devices": []}
        if report["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                report["devices"].append({"name": torch.cuda.get_device_name(index),
                                          "free_total_bytes": torch.cuda.mem_get_info(index)})
        if Path("/mnt/nvme").exists():
            report["nvme"] = {"free_bytes": shutil.disk_usage("/mnt/nvme").free,
                              "read_only": bool(os.statvfs("/mnt/nvme").f_flag & os.ST_RDONLY)}
        print(json.dumps(report, indent=2))
        return
    if args.command == "smoke":
        if not 1 <= args.steps <= 20:
            raise ValueError("Smoke tests are bounded to 1..20 steps")
        from .smoke import run_smoke
        receipt = json.dumps(run_smoke(args.steps), indent=2)
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(receipt + "\n")
        print(receipt)
        return
    # Fail on output collisions and invalid settings before loading large components.
    if hasattr(args, "output") and args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.command == "train":
        config = validate_config(json.loads(args.config.read_text()))
    elif args.command == "infer":
        if any(not math.isfinite(s) or abs(s) > 3 for s in args.scales):
            raise ValueError("Scales must be finite and in [-3, 3]")
        if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
            raise ValueError("Dimensions must be positive multiples of 16")
        if args.steps < 1 or not math.isfinite(args.guidance) or args.guidance < 0:
            raise ValueError("Invalid steps or guidance")
    import torch
    from .backend import load_pipeline
    pipe = load_pipeline(device=args.device, local_files_only=not args.download,
                         training=args.command == "train", offload=getattr(args, "offload", False))
    if args.command == "train":
        from .artifacts import save_slider
        from .backend import encode_training_prompts
        from .lora import Slider
        from .training import checkpoint_blocks, sample_training_state, train_step
        torch.manual_seed(config["seed"])
        contexts = encode_training_prompts(
            pipe, [config[key] for key in ("neutral", "positive", "negative")], args.device,
            config["max_sequence_length"])
        slider = Slider(pipe.transformer, rank=config["rank"])
        checkpoint_blocks(pipe.transformer)
        optimizer = torch.optim.AdamW(slider.parameters(), lr=config["lr"])
        generator = torch.Generator("cpu").manual_seed(config["seed"])
        metrics = []
        for step in range(config["steps"]):
            latent, time = sample_training_state(pipe, slider, contexts[0], generator,
                                                 config["resolution"], config["rollout_steps"], step)
            result = train_step(pipe.transformer, slider, optimizer, latent, time, contexts,
                                config["strength"])
            metrics.append({"step": step + 1, **result})
            print(json.dumps(metrics[-1]), flush=True)
        save_slider(slider, args.output, config)
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    else:
        from .artifacts import load_slider
        slider = load_slider(pipe.transformer, args.adapter)
        slider.to(args.device)
        if args.command == "ui":
            from .ui import launch
            launch(pipe, slider)
        else:
            from .inference import render
            args.output.mkdir(parents=True, exist_ok=False)
            for index, scale in enumerate(args.scales):
                image = render(pipe, slider, args.prompt, scale=scale, seed=args.seed,
                               width=args.width, height=args.height, steps=args.steps,
                               guidance=args.guidance)
                image.save(args.output / f"{index:02d}_scale_{scale:+g}.png")
            (args.output / "generation.json").write_text(json.dumps(
                {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
