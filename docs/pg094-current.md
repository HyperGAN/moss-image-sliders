# Current MOSS PG0.9.4 source cohort

The canonical `moss-image-sliders 0.3.0.dev2` distribution and `moss-sliders`
training path now use official ParticleGAN0.9.4 through shared core0.12.0 at
`ddb41423bcefc1fe76cd94d0f9b367353b3ea137`. This includes the merged unified
training/isolated execution API. HyperGAN's native graph builder remains pinned
to `5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL0.10.0, and PG's upstream
release is `330eb94ee748cdcad131b0275d8de75fd5a7f96f` (official wheel SHA256
`ad21eeaea861cf0582fc39aa96eb3fd8ab8cf21e8e0f0179f0597a9c42eb99b6`).
Python>=3.11,<3.13 is required. This promotes the existing HNDL implementation
from an optional preview to the supported base path; a version bump of the old
base PG0.9.1 path alone would not make its campaign APIs compatible.

The product owns its pinned MOSS loader, native attention LoRA/frozen rollout
factory and velocity-field layout. The shared core owns `ConditionalPG09`, the
actual spatial HNDL critic, optimizer lifecycle, balanced independent generator/
critic sampling, losses, regularizers and full-state replay. The generator arm
is adapter-only, with `particles=None`; no fake particle bank or local GAN loss
is introduced. The default recipe is public `bispf_optimized`.

## Source admission and installation

`src/moss_sliders/hndl_cohort.json` locks the exact built wheel's 573 files under
particle_sliders and one concept_slider_core compatibility namespace file: 574
source/data files total. The wheel-root legacy import startup hook is a separate
artifact covered by the wheel hash. Runtime admission compares the complete
recursive package closures and delegates official PG/HNDL/HyperGAN source
qualification to the shared core. New files, altered source or graph data, wrong
version, dirty HyperGAN build provenance and cross-cohort packet headers refuse.
The previous a794 manifest is archived unchanged in profiles/hypergan-hndl/history.

Use README's clean install in a new environment. The current CPU lock has 45
hash-checked runtime packages for CPython3.12/Linux x86-64, including Torch2.9.1+cpu,
Pillow13, official PG0.9.4 and HNDL0.10.0. Install the exact core and HyperGAN source
commits separately with --no-deps, then install this product. Both source commits
must remain reachable; dependency installation is not a model qualification.
A cached uv VCS build of HyperGAN was observed with dirty=true and correctly
rejected; a clean `pip install --no-cache-dir --no-deps` of the same exact commit
produced the required clean provenance. Do not alter the provenance JSON or weaken
the gate to admit an invalid build.

## Checkpoints, weights and historical paths

Source/profile identities deliberately change. Historical signed-MSE pilots,
four controls, PG0.9.0/0.9.1 records and existing local runs keep their recorded
lineage; they have not been retrained or relabelled. Legacy portable v1 adapters
remain readable. New GAN exports/checkpoints record PG0.9.4 and exact runtime
identity. Source-bound old GAN packets/exports require their original environment;
unidentified v1 training headers and cross-cohort packets reject before tensor
loading/state mutation. The old cached-campaign API is absent from the latest
shared core, so current train-campaign refuses before any model loading. Use
[the original exact source/lock cohort](../profiles/legacy-pg091/README.md) for
historical reproduction, including those campaigns.

The concurrent feat/unified-training-command-20261104 branch and its machine-local
PG094 source-bound bridge remain separate. Neither it nor its runs/checkpoints
were changed or promoted to portable native qualification.

## Qualification limits

The installed CPU checks exercise actual random tiny MOSS and HNDL graphs,
optimizer ownership, frozen base/OFF/zero behavior, signed balanced rows, exact
resume, portable reload, malformed/history checkpoint rejection, complete source
closure tampering, scheduler and RNG isolation, BF16 CPU arithmetic, and rollback.
The merged `particle-slider` dispatcher can plan/check/train/resume the explicit
`moss-cpu-fixture` profile with dummy=true and emit completed native checkpoints
and adapters. The portable pretrained profile remains source_only; --check loads
no model. It preserves the CUDA-only and pinned local-cache gate.

No pretrained weights were downloaded, no GPU training ran, and no existing runs
were modified. Pretrained CUDA/BF16/offload/VAE/media quality/performance remain
unqualified for this new source cohort. UI is still blocked by the patched
Pillow>=13/Gradio<13 constraint. No deployment, training-weight publication or
merge is part of this change. Historical receipts remain historical.

Installed artifact evidence: [receipt and exact source/artifact hashes](receipts/pg094-current/receipt.json),
[wheel suite](receipts/pg094-current/wheel-tests.xml) (114 passed, one optional UI
skip), [sdist rebuilt-install native/dispatcher lane](receipts/pg094-current/sdist-tests.xml)
(22 passed), and [installed CLI smoke](receipts/pg094-current/cli-smoke.json).
Wheel and sdist pip checks pass. All installed product package bytes match the
final wheel and checkout; all 574 core namespace files match the exact ddb source.
