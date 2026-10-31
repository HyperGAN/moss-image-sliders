# Isolated HyperGAN/HNDL preview

This preview executes official PG 0.9.4 through shared core 0.12.0, the exact
HyperGAN develop builder `5f368a3770cf5bdae0448995c9f76e29af8049f7` and HNDL 0.10.0.
Both attention LoRA projections and the spatial field critic execute actual HNDL
graphs. The critic graph, GAN losses, optimizers, checkpoint validation and native
response admission come from the installed shared package. The MOSS loader,
frozen rollout, signed adapter hooks and sampler remain product owned.

The default distribution remains the historical core 0.6.0/CDF/PG 0.9.1 cohort.
The new `moss-image-sliders-hndl` variant is built from the same source files in
a temporary staging directory, using its own dependency metadata and `moss-hndl`
entry point. Install each variant in its own virtual environment: both provide
`moss_sliders`, and they cannot share one environment. An optional extra requiring
two incompatible exact core versions would be unsatisfiable.

## Build and admission

The exact core candidate
[`a794e3b4a8813b012f2c4faa3da82b78d4a90214`](https://github.com/HyperGAN/particle-sliders/commit/a794e3b4a8813b012f2c4faa3da82b78d4a90214)
is now publicly reachable as [draft PR148](https://github.com/HyperGAN/particle-sliders/pull/148).
The preview metadata pins that full Git commit, including the serialization and
HyperGAN/HNDL extras. This is source availability, not a merged release or
pretrained/GPU qualification. The complete core Python/data/graph closure still
has to match the packaged `hndl_cohort.json`; another core 0.12 build cannot bypass
that check by sharing its version number.

This modern preview is a new cohort, version `0.3.0.dev1`. The recursive manifest
now covers every package source/data file and the compatibility namespace. Its
qualification build wheel SHA256 is
`46fbe99cf0cf9e4cbd7143dd220c479ccbb8820dc348ee850da04047f1cea1ec`.
The old `4abc` manifest is preserved under
`profiles/hypergan-hndl/history/core-4abc-cohort.json`. Old preview full packets
and portable exports are not relabelled or migrated: source/config/runtime
identity mismatches fail closed. Restore those with their original source and
core cohort. Default core 0.6.0/PG 0.9.1 and historical locks are unchanged.

### Clean CPU installation

This sequence is for Python 3.12 / Linux x86-64. Build the variant in a separate
builder environment, then install its 45 hash-checked runtime archives and the
exact public VCS packages into a fresh runtime environment:

```bash
python3.12 -m venv .venv-hndl-build
.venv-hndl-build/bin/python -m pip install 'hatchling==1.28.0'
.venv-hndl-build/bin/python tools/build_hndl_profile.py --sdist --output dist/hndl
python3.12 -m venv .venv-hndl
. .venv-hndl/bin/activate
python -m pip install --require-hashes -r requirements-hndl-cpu.lock
python -m pip install --no-deps 'hypergan @ git+https://github.com/HyperGAN/HyperGAN.git@5f368a3770cf5bdae0448995c9f76e29af8049f7'
python -m pip install --no-deps 'particle-sliders-core[serialization,hypergan-hndl] @ git+https://github.com/HyperGAN/particle-sliders.git@a794e3b4a8813b012f2c4faa3da82b78d4a90214#subdirectory=packages/particle-sliders-core'
python -m pip install --no-deps dist/hndl/moss_image_sliders_hndl-0.3.0.dev1-py3-none-any.whl
python -m pip check
moss-hndl --help
moss-hndl smoke --steps 4
```

The separate `requirements-hndl-cpu.lock` was resolved from this profile and the
exact official CPU Torch archive in `requirements-runtime-cpu.in`. It keeps
Pillow 13, official PG 0.9.4 and HNDL 0.10. VCS packages are installed in separate
full-commit steps because pip archive-hash mode cannot hash VCS requirements;
their transitive archives are already in the lock. No UI, CUDA dependency or
model weights are installed. The old `uv.lock`, `requirements-cpu.lock`, base
`requirements-runtime-cpu.lock` and checkpoint source manifest are preserved.
Regenerate only this candidate lock with
`python tools/lock_runtime_cpu.py --profile hndl --uv /path/to/uv`.

The clean public-source environment passed `pip check`, source/builder admission,
installed CLI discovery, tiny native CPU smoke and the focused installed-package
suite. The public source availability and CPU installation blocker is closed;
[the independent receipts](receipts/core-a794-migration/receipt.json)
record the exact commits, commands, archive count and qualification scope.
No UI extra is defined. The `train` command still requires an explicit
`pg_version: "0.9.4"` recipe; use `examples/warm-light.json` as a structural
reference without changing that historical file. Native training remains
unqualified and was not run for this task.

Programmatic callers use `moss_sliders.native_hndl.build_hndl_run` with their
existing MOSS pipeline and frozen encoded contexts, followed by the existing
shared-game checkpoint/export helpers. Historical cached campaign providers are
absent from the clean core 0.12 lane, so `train-campaign` refuses before loading a
model. UI is also refused. The historical `moss-sliders` command remains unchanged
in its original environment.

## Provenance and compatibility

Native image velocity responses stay `[B,C,H,W]`. The critic also accepts an
explicit singleton temporal axis `[B,C,1,H,W]`, squeezing only that axis at the
conv boundary. It never flattens fields to bypass the continuous-domain guard.
The signed adapter remains an adapter GAN, without inventing a particle bank.
Scale zero bypasses every residual graph, preserving the frozen response exactly.

Preview checkpoints and portable exports contain the actual PG 0.9.4 source
profile, verified HyperGAN/HNDL identity and product binding source hash. Loading
an export checks that identity before attaching hooks. Old PG 0.9.1 full game
packets are rejected without mutation; restore those with the original source
and dependency cohort. Historical v1 MSE artifacts retain their original format
and meaning. Existing sample images and old receipts do not qualify this preview.

New full checkpoints use `moss-pg09-checkpoint-v2` JSON with explicit `pg_version`,
`pg_source_profile` and `penalty_input_domain`. A mismatched or incomplete header
rejects before tensor loading. Original v1 checkpoint headers remain supported
inside the original PG 0.9.1 cohort; this preview rejects unidentified legacy
headers before tensor loading and directs them to their original source environment.

CPU evidence covers actual HNDL projection output/input/parameter-gradient parity
at matched weights, native field gradient ownership, real shared GAN updates,
unchanged frozen base, exact full-game resume, incompatible checkpoint rejection,
portable actual-weight reload and zero-strength behavior. Pretrained weights,
CUDA/BF16, native image quality, offload, memory and performance are unrun. No
model downloads, GPU training, package publication or deployment are performed.

## Scope of this migration

The parent-qualified core includes synthetic CUDA software evidence. This MOSS
migration adds only CPU installed-package evidence and does not qualify native
MOSS weights, GPU training, media quality or a new response domain. No domain or
width guard is relaxed. The MOSS binding exposes raw adapter evaluation; it does
not advertise EMA selection. Existing zero-strength, exact-resume, failure/retry
and source-tamper tests remain the relevant product contract. Full migration
identity and old-packet rejection receipts are in `receipts/core-a794-migration/`.
