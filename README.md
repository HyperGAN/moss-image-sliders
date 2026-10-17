# MOSS-Image Sliders

A separately maintained HyperGAN project for signed concept LoRA controls on
[OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT).
Adapted from [particle-sliders](https://github.com/HyperGAN/particle-sliders), with
a MOSS-specific backend, trainable attention adapters, reproducible ramp inference,
and a local Gradio control panel. The original repository is untouched.

**Status:** CPU integration validated on the actual MOSS implementation at tiny random
dimensions. Full SFT checkpoint training, GPU memory use and visual concept quality still
need a GPU acceptance run. Example configs are starting recipes, not trained adapters.
This project does not implement the separate ParticleGAN develop BISPF upgrade.

## Install

Python 3.10–3.12, recommended 3.12; uv 0.13.1. CUDA builds use the PyTorch 2.9.1
runtime selected by `uv.lock`; verify compatibility with your installed driver first.

```bash
git clone https://github.com/HyperGAN/moss-image-sliders.git
cd moss-image-sliders
uv sync --frozen --extra dev --extra ui
uv run --frozen moss-sliders doctor
```

For CPU-only development, use the separately hashed CPU lock (Linux x86_64 / Python 3.12).
It avoids downloading CUDA runtimes. Both locks pin all transitive dependencies.

```bash
uv venv --python 3.12
uv pip sync requirements-cpu.lock --torch-backend cpu --require-hashes
uv pip install --no-deps -e .
.venv/bin/pytest -q
.venv/bin/moss-sliders smoke --receipt outputs/smoke.json
```

Do not use `uv run` with the CPU environment: it would synchronize the CUDA lock.
CPU smoke tests use no network, no model cache and no pretrained weights.
The same checks are provided in [a CI workflow example](docs/ci-workflow.example.yml).
Automatic Actions installation awaits a maintainer with workflow permissions; the
publishing token's permissions were preserved.

## Train a concept

Inspect and edit `examples/warm-light.json` or `examples/watercolor.json`. Each recipe
specifies neutral/positive/negative descriptions, LoRA rank, learning rate, training
steps, rollout steps, resolution, seed, direction strength and token budget.
The default example is a bounded 20-step pilot at 512px, not a quality claim.

```bash
# Optional: point this at a verified writable cache with >=40 GB free.
export HF_HOME=/path/to/writable/cache
uv run --frozen moss-sliders train --config examples/warm-light.json \
  --output checkpoints/warm-light-pilot --device cuda:0 --download
```

`--download` explicitly allows the pinned model download (about 28 GB); otherwise
loading is local-cache-only. It never selects or modifies storage automatically.
Output directories must be new. Training saves only `adapter.safetensors`,
`slider.json` (base revision, target names and recipe), and `metrics.json`.
There is no automatic long training job, tracking upload, or checkpoint publication.
Interrupted runs do not have resumable optimizer checkpoints in v0.1.

The frozen neutral model first denoises a shared noise sample to one of the official
scheduler timesteps. At this same latent and time, frozen positive and negative prompts
define a velocity difference. The neutral-caption student learns
`v_base + scale * strength * (v_positive - v_negative)` at scales +1 and -1.
Only low-rank q/k/v/o residuals in the main DiT blocks receive gradients. The text
encoder and base transformer stay frozen. This is a contrastive concept distillation
baseline; it reuses particle-sliders' signed adapter mechanism, not its full experimental
particle objectives. Scale zero bypasses every adapter exactly. Intermediate and
extrapolated scale behavior must be inspected; a linear residual need not yield a
linear perceptual change. Wider prompt generalization needs multiple training scenes
and a subsequent evaluation extension; this release trains one prompt triplet per run.

## Render a matched-seed ramp

```bash
uv run --frozen moss-sliders infer --adapter checkpoints/warm-light-pilot \
  --prompt 'A ceramic teapot on a wooden table, studio photograph.' \
  --scales -1 -0.5 0 0.5 1 --seed 42 --steps 50 --guidance 3 \
  --width 1024 --height 1024 --output outputs/warm-light-ramp --offload
```

Each image uses a fresh generator with the same seed; the output folder includes
generation parameters. Adapter/model revision mismatches fail before attachment.
Different CFG values and layer skipping can change slider behavior: start with the
official defaults (50 steps, CFG 3, shift 5, negative branch skip at layer 22).
The training velocity teachers do not apply sampling CFG or layer skipping.

## Local interface

```bash
uv run --frozen moss-sliders ui --adapter checkpoints/warm-light-pilot --offload
```

Controls include signed concept strength, seed, dimensions, steps and CFG. Requests
are serialized because scale and scheduler state are mutable. The UI binds to
127.0.0.1 and does not create a public share link. Training controls live in the
explicit JSON recipe and CLI to keep expensive runs deliberate.

## Resource planning and acceptance

The 5.08B DiT alone is about 10.16 GB of stored BF16 weights. The bundled Qwen3.5
encoder contributes about 16.79 GB; the FP32 VAE adds more. A full resident pipeline
therefore needs considerably more than 27 GB VRAM including activations and workspaces.
48 GB A6000 inference is a plausible starting point, not measured here; component
offload lowers resident use. Training caches the three prompt encodings, moves the
encoder to CPU, and loads only the DiT on the selected GPU (no VAE). Main block
activation checkpointing is enabled. Use one GPU for the first pilot; this release
does not pool two GPUs or offer distributed training. Budget at least 40 GB of cache
disk and ample CPU RAM (64 GB or more recommended). Check free VRAM before launching;
512px is the pilot resolution. Do not infer 96 GB pooled VRAM from two 48 GB cards.

After CUDA is available, run the 20-step pilot, inspect finite losses, load its saved
adapter in a fresh process, then compare -1/0/+1 ramps on the training prompt and held-out
scenes. Check that the zero image matches an adapter-free run with identical sampler,
seed and prompt, and inspect preservation of unrelated composition. GPU timing, peak
memory and visual quality should be recorded before increasing training steps.
No pretrained weight download or real GPU training was performed in the initial release.

See [model integration](docs/model-integration.md), [test receipts](docs/validation.md),
[NOTICE](NOTICE), and retained [licenses](LICENSES).
