# MOSS-Image Sliders

A separately maintained HyperGAN project for signed concept LoRA controls on
[OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT).
Adapted from [particle-sliders](https://github.com/HyperGAN/particle-sliders), with
a MOSS-specific backend, trainable attention adapters, reproducible ramp inference,
and a local Gradio control panel. The original repository is untouched.

**Status:** A real 60-update MOSS-Image-SFT GPU pilot produced the lighting examples
below. Moderate strengths give a visible cool-to-warm change with better shape preservation;
full strengths can change geometry. Trained adapter weights remain private. The pilot uses
signed LoRA velocity distillation. This project does not implement the separate ParticleGAN
BISPF upgrade.

## Samples

**Start around magnitude 0.35 for this lighting pilot.** Each row keeps its prompt,
seed, 512 × 512 resolution, 50 Euler steps and CFG 3 fixed; only signed adapter
strength changes. The controls run **cool → off → warm**. These are actual model
outputs from the same trained adapter, not color-filtered or blended previews.

### Lighting warmth · gentle

**Cool · −0.35** → **Off · 0** → **Warm · +0.35**

![Lighting warmth: cool minus 0.35, base zero, warm plus 0.35](assets/lighting-warmth/gentle.png)

Full resolution: [−0.35](samples/lighting-warmth/teapot/minus-0.35.png) ·
[Off](samples/lighting-warmth/teapot/off.png) ·
[+0.35](samples/lighting-warmth/teapot/plus-0.35.png)

<details><summary>Prompt and seed</summary>

A ceramic teapot on a wooden table, studio photograph.

Seed: `42`. This is the neutral caption from the training triplet.

</details>

### Lighting warmth · full strength and held-out scene

**Cool · −1** → **Off · 0** → **Warm · +1**

![Full-strength lighting warmth: teapot seed 42 above and held-out cup and book seed 84 below](assets/lighting-warmth/full-strength.png)

Full resolution, teapot: [−1](samples/lighting-warmth/teapot/minus-1.png) ·
[Off](samples/lighting-warmth/teapot/off.png) ·
[+1](samples/lighting-warmth/teapot/plus-1.png)

Full resolution, held-out cup/book: [−1](samples/lighting-warmth/held-out-cup-book/minus-1.png) ·
[Off](samples/lighting-warmth/held-out-cup-book/off.png) ·
[+1](samples/lighting-warmth/held-out-cup-book/plus-1.png)

<details><summary>Prompts and seeds</summary>

Top row: A ceramic teapot on a wooden table, studio photograph.

Seed: `42`.

Bottom row: A white porcelain cup beside an open book on a wooden desk, studio photograph.

Seed: `84`. This scene was not used in training.

</details>

The cool-to-warm direction is visible in both scenes. **Full strength also changes
geometry and composition:** the +1 teapot changes its handle/spout, and the −1
held-out cup loses its handle. This is not clean relighting. The moderate ramp retains
the teapot's shape much better, with small texture changes. These are development
examples from one training triplet, not a final-test benchmark or a general quality claim.
Zero strength matches the separately rendered [adapter-free baseline](samples/lighting-warmth/teapot/adapter-free.png)
pixel for pixel on the teapot scene.

<details><summary>Training, inference and provenance</summary>

Base: [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), pinned to
`fc5de0d56fc910cbca6f6625a70f5f582b26deb1`. Training and rendering code:
[`a6f5cd0`](https://github.com/HyperGAN/moss-image-sliders/commit/a6f5cd06aa561d0edd495656509a712c6c4b8ead).
Generated 2026-10-17 on one RTX A6000.

Training: 60 updates; rank/alpha 4; LR 0.0003; direction strength 1; seed 42;
512px; four-step frozen neutral rollouts; token cap 256; frozen DiT base and text
encoder. Neutral caption: `A ceramic teapot on a wooden table, studio photograph.`
Positive caption adds `warm golden lighting.`; negative adds `cool blue lighting.`
Exact strings and file SHA256 values are in the [provenance manifest](samples/lighting-warmth/provenance.json).

The executed objective is signed LoRA contrastive velocity distillation. At the same
latent and timestep, frozen teachers give `delta = v(positive) - v(negative)` and
`base = v(neutral)`. The neutral-caption student at signs +1 and −1 minimizes the mean
of the two velocity MSEs against `base + delta` and `base - delta` respectively.
There is **no discriminator, adversarial loss, or ParticleGAN/BISPF call** in this
pilot's executed training path. Scale zero bypasses the adapter exactly.

Inference: fresh matched-seed CPU generator for each strength; 50 steps; CFG 3;
scheduler shift 5; negative prompt empty; negative branch skips layer 22. PNGs are
published unchanged, with contact-sheet labels added outside the generated images.
Only examples and their documentation/provenance are published here; the trained
adapter and private result bundle remain private.

</details>

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
