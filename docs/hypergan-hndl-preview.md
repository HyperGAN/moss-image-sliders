# Isolated HyperGAN/HNDL preview

This preview executes official PG0.9.4 through shared core0.12.0, the exact
HyperGAN develop builder `5f368a3770cf5bdae0448995c9f76e29af8049f7` and HNDL0.10.0.
Both attention LoRA projections and the spatial field critic execute actual HNDL
graphs. The critic graph, GAN losses, optimizers, checkpoint validation and native
response admission come from the installed shared package. The MOSS loader,
frozen rollout, signed adapter hooks and sampler remain product owned.

The default distribution remains the historical core0.6.0/CDF/PG0.9.1 cohort.
The new `moss-image-sliders-hndl` variant is built from the same source files in
a temporary staging directory, using its own dependency metadata and `moss-hndl`
entry point. Install each variant in its own virtual environment: both provide
`moss_sliders`, and they cannot share one environment. An optional extra requiring
two incompatible exact core versions would be unsatisfiable.

## Build and admission

The preview's packaged `hndl_cohort.json` binds the complete core Python/data/graph
source closure to local candidate commit
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` and wheel SHA256
`9afad3afc1f6c653a574074dc14f0da1f1c43975c31ca45c4eac049df0b0b4ad`.
Its variant metadata requires core0.12.0; that version check alone cannot admit a
different source closure. That candidate is not yet published:
normal network installation of this profile is blocked until the central owner
publishes and reviews the exact cohort. Do not substitute a similarly numbered
release or overwrite the historical environment. Local qualification installs the
owner's exact candidate wheel in a second isolated environment and verifies the
official PG sources and HyperGAN/HNDL builder closure at runtime.

With the declared Hatchling backend installed, build the separate wheel:

```bash
python tools/build_hndl_profile.py --output dist/hndl
```

After that exact core candidate is available, a fresh supported Python3.12 CPU
environment can install the preview wheel and its dependencies normally. Resolve
Torch against the official CPU artifact and retain Pillow>=13. No UI extra is
defined for this variant. `moss-hndl --help` discovers commands without loading a
model; `moss-hndl smoke --steps 4` exercises tiny random native MOSS weights only.
The `train` command requires an explicit `pg_version: "0.9.4"` recipe; use
`examples/warm-light.json` as a structural reference, keeping its historical file
unchanged. Native training remains unqualified and was not run for this task.

Programmatic callers use `moss_sliders.native_hndl.build_hndl_run` with their
existing MOSS pipeline and frozen encoded contexts, followed by the existing
shared-game checkpoint/export helpers. Historical cached campaign providers are
absent from the clean core0.12 lane, so `train-campaign` refuses before loading a
model. UI is also refused. The historical `moss-sliders` command remains unchanged
in its original environment.

## Provenance and compatibility

Native image velocity responses stay `[B,C,H,W]`. The critic also accepts an
explicit singleton temporal axis `[B,C,1,H,W]`, squeezing only that axis at the
conv boundary. It never flattens fields to bypass the continuous-domain guard.
The signed adapter remains an adapter GAN, without inventing a particle bank.
Scale zero bypasses every residual graph, preserving the frozen response exactly.

Preview checkpoints and portable exports contain the actual PG0.9.4 source
profile, verified HyperGAN/HNDL identity and product binding source hash. Loading
an export checks that identity before attaching hooks. Old PG0.9.1 full game
packets are rejected without mutation; restore those with the original source
and dependency cohort. Historical v1 MSE artifacts retain their original format
and meaning. Existing sample images and old receipts do not qualify this preview.

CPU evidence covers actual HNDL projection output/input/parameter-gradient parity
at matched weights, native field gradient ownership, real shared GAN updates,
unchanged frozen base, exact full-game resume, incompatible checkpoint rejection,
portable actual-weight reload and zero-strength behavior. Pretrained weights,
CUDA/BF16, native image quality, offload, memory and performance are unrun. No
model downloads, GPU training, package publication or deployment are performed.
