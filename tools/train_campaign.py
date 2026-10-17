"""Explicit compatibility command for the shared GAN cache trainer.

The original active campaign checkout is never dispatched by this derivative.
"""
import argparse
from pathlib import Path
from moss_sliders.cli import main as slider_main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=3840)
    args = parser.parse_args(argv)
    if args.recipe.name != "recipe.json" or not args.recipe.is_file():
        raise ValueError("A verified frozen campaign recipe.json is required")
    command = ["train-campaign", "--cache", str(args.recipe.resolve().parent),
               "--output", str(args.output), "--steps", str(args.steps)]
    if args.resume:
        command += ["--resume", str(args.resume)]
    slider_main(command)


if __name__ == "__main__":
    main()
