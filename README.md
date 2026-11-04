# MOSS-Image Sliders

Signed concept LoRA controls for pinned OpenMOSS MOSS-Image-SFT. The current
`moss-image-sliders 0.3.0.dev2` package and `moss-sliders train` use official
**ParticleGAN 0.9.4**, shared **particle-sliders-core 0.12.0** at
`ddb41423bcefc1fe76cd94d0f9b367353b3ea137`, HyperGAN develop builder
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, and HNDL 0.10.0. Actual HNDL
attention LoRA and the shared spatial field critic execute the public
`bispf_optimized` adapter GAN; the shared core owns losses, optimizers,
sampling and full-state checkpoint validation. **There is no particle bank**:
`particles=None` is the legitimate adapter-only generator arm.

CPU software checks use the real tiny random MOSS architecture. Pretrained CUDA,
BF16, offload, real VAE output, performance and media quality remain unqualified
for this new source cohort. Model loading still requires CUDA and defaults to
local cached weights. Old weights, four historical controls, MSE pilots and
PG0.9.0/0.9.1 run records retain their original objective/version and are not
retrained by this package update. [Migration and exact limits](docs/pg094-current.md).

## Install the current CPU runtime

The package supports Python 3.11–3.12. This hash-checked runtime lock is specifically
for CPython 3.12/Linux x86-64 and uses the official Torch 2.9.1+cpu archive.
Create a new environment; never repin an active campaign:

```bash
python3.12 -m venv .venv-cpu
. .venv-cpu/bin/activate
python -m pip install --require-hashes -r requirements-runtime-cpu.lock
python -m pip install --no-deps 'particle-sliders-core[serialization,hypergan-hndl] @ git+https://github.com/HyperGAN/particle-sliders.git@ddb41423bcefc1fe76cd94d0f9b367353b3ea137#subdirectory=packages/particle-sliders-core'
python -m pip install --no-deps 'hypergan @ git+https://github.com/HyperGAN/HyperGAN.git@5f368a3770cf5bdae0448995c9f76e29af8049f7'
python -m pip install --no-deps .
python -m pip check
moss-sliders --help
moss-sliders smoke --receipt outputs/gan-smoke.json
```

`moss-hndl` remains a compatibility command for the same current native path.
`profiles/hypergan-hndl` no longer selects a different core from the base package.
The old root `uv.lock`/`requirements-cpu.lock` and base PG0.9.1 lock are archived
unchanged in [profiles/legacy-pg091](profiles/legacy-pg091/README.md).

**UI installation remains blocked**: Gradio 6.30 requires Pillow<13 while this
package keeps Pillow>=13. Do not add `.[ui]` or use the archived full/UI lock as a
current environment. The security floor is preserved.

## One training command

The merged core frontend discovers the installed product profiles:

```bash
particle-slider train --config examples/warm-light.json --profile moss --check --no-reporting
particle-slider train --config examples/warm-light.json --profile moss --out /new/run
```

The native profile is explicitly `source_only` until the new model cohort is
qualified. `--check` validates recipe/runtime sources without loading weights.
Execution uses this product's native factory and existing CUDA/local-cache gate.
To run the bounded CPU architecture fixture, explicitly select its profile and
`dummy=true` recipe:

```bash
particle-slider train --config examples/cpu-fixture.json --profile moss-cpu-fixture --out /new/cpu-fixture --no-reporting
```

Resume accepts a complete source-compatible checkpoint directory using `--resume`.
The original `moss-sliders train --config ... --output ...` interface remains.
Neither training command searches learning rates, renders images or publishes
weights. [Unified execution details](TRAINING.md).

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

## Train a concept

The example JSON recipes explicitly name `pg09_gan_only`, ParticleGAN `0.9.1`,
`bispf_optimized`, and batch size. Recipes from the old MSE trainer fail validation.
Training and inference preflight the full native chat template before any
encoder forward. An over-budget prompt fails instead of being silently truncated.

```bash
moss-sliders train --config examples/warm-light.json \
  --output checkpoints/new-gan-run --device cuda:0
```

Model loading is local-only unless `--download` is explicitly supplied. Real runs
require coordinator GPU admission. For each frozen neutral rollout state the two
signed teachers are `base +/- strength * (positive - negative)`. The student uses
the neutral caption. Critic conditions contain mask-weighted normalized neutral
Qwen features plus native time, sign and strength. Frozen edit RMS from training
states sets the response coordinate scale; no reconstruction/MSE loss is trained.
The shared recipe owns both optimizers and its public adversarial/penalty terms.
The four-step rollout uses the actual SFT scheduler grid approximately
`[1000, 910, 717, 24]`; rendering below uses a distinct 50-step grid. See
[the verified scheduler review](docs/sft-scheduler-review-20261031.md) for both
grids, historical provenance and the remaining native quality gap.

The output is a new directory containing `run.json`, metrics, immutable full-game
checkpoints and an `adapter/` portable v2 export. For an interrupted run, use a new
output directory and the original full checkpoint and unchanged recipe:

```bash
moss-sliders train --config examples/warm-light.json \
  --resume checkpoints/new-gan-run/checkpoints/step-000010 \
  --output checkpoints/resumed-gan-run --device cuda:0
```

Resume restores actual completed steps, both optimizers, critic, adapter, model
RNG, independent noise/row streams, shuffled orders, cursors and exposure counts.
The total budget remains the original recipe's budget. Model, prompt, conditioning,
source, runtime, recipe or objective changes fail closed. Adapter-only exports and
legacy MSE optimizer packets cannot resume a GAN run. Existing source-bound native
campaign checkpoints must continue under their original admitted runtime; package
relocation is a new lineage and requires a new cohort or reviewed migration.

## Historical frozen campaign caches

Cached campaigns require their original PG/core/product source cohort. Current
`train-campaign` fails before any model load; its old shared campaign module is
not part of the current core. Continue historical campaigns from the exact
recorded environment or [legacy reproduction instructions](profiles/legacy-pg091/README.md).
The current native factory builds signed frozen rollouts through the supported
shared API. No cache, run directory or old artifact was modified.

## Render a matched-seed ramp

```bash
moss-sliders infer --adapter checkpoints/new-gan-run/adapter \
  --prompt 'A ceramic teapot on a wooden table, studio photograph.' \
  --scales -1 -0.5 0 0.5 1 --seed 42 --steps 50 --guidance 3 \
  --width 1024 --height 1024 --output outputs/gan-ramp --offload
```

The render default stays at 50 steps. To exercise the same four-step scheduler
grid used by the example's frozen training rollouts, use a separate diagnostic
output directory:

```bash
moss-sliders infer --adapter checkpoints/new-gan-run/adapter \
  --prompt 'A ceramic teapot on a wooden table, studio photograph.' \
  --scales -1 -0.5 0 0.5 1 --seed 42 --steps 4 --guidance 3 \
  --width 1024 --height 1024 --output outputs/gan-ramp-four-step --offload
```

This reproduces the scheduler grid, not a pretrained quality result or the neutral
training trajectory. The 50-step example and all strength defaults remain unchanged.

Inference and UI execute the shared MOSS response backend. CFG adapts only the
conditional branch, keeps the empty-prompt branch frozen, and disables negative
layer skipping. Each strength gets fresh identical-seed FP32 noise and a fresh
copy of the native scheduler. The shared decoder retains native VAE normalization
and unpacking. At zero, latents match an unattached native baseline under this
same no-skip protocol. Historical v1 adapters remain readable; their appearance
can differ under the corrected CFG protocol. Raw source-bound campaign exports
require validation with their original `MossCampaignRun.load_adapter` and explicit
conversion to portable v2, rather than guessing their parameter mapping.

The UI is currently blocked by the dependency conflict described above. Its existing
code serializes requests, binds to 127.0.0.1, and creates no public share link; this
is not a current installation or UI qualification claim.
Native CUDA/BF16, real VAE output, offload hooks, peak memory and visual quality
remain separate coordinator-admitted gates. No trained weights are published.
