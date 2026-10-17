# MOSS shared ParticleGAN integration audit

The inspected public repository is `HyperGAN/moss-image-sliders`, public main
`477278a2e87f77d8a56a38bd6b952741231f1b3e`. The campaign source baseline is
`0aa3867965aad20f3ee653dcf14e49a1fbca06d9`. Work is isolated on
`audit/moss-pg09-shared-integration-20261017`. Original campaign worktrees, pinned
sources, resident services, credentials, permissions and GPU processes are untouched.

## Executed entrypoints

| Entrypoint | Executed implementation |
| --- | --- |
| `moss-sliders train` | `gan.build_run` calls installed `particle_sliders.pg09.ConditionalPG09`, shared MOSS response adapter, field critic and scale-aware checkpointing |
| `moss-sliders train-campaign` | Shared `pg09_moss_campaign.MossCampaignRows` and `build_campaign_run`; original caches read-only |
| `tools/train_campaign.py` | Explicit forward to `train-campaign`; requires a new output directory |
| `moss-sliders smoke` | Same shared GAN on tiny real random MOSS, full-game disk replay and portable adapter reload |
| `moss-sliders infer` / UI | Shared `pg09_backends` CFG, native scheduler and decoder; actual loaded LoRA leaves |
| Historical `training.train_step` / `campaign.cached_backward` | Reject before update; no MSE optimizer path |
| Old monitor / autonomous CUDA smoke | Reject before GPU queries, locks or process dispatch |

The shared package alone owns GAN losses, public recipe selection, optimizers,
regularization, independent noise streams and balanced sampler state. This MOSS
adapter has no routed particle bank, so no unused particle prior is created.
Explicit new recipes require ParticleGAN 0.9.1 and `bispf_optimized`. The old native
0.9.0 runs and their source-bound checkpoints retain their original runtime.

## Demonstrated defects and corrections

The retained [legacy probe](receipts/shared-pg09-audit/probe_legacy_defects.py)
loads the exact original blobs and reproduces delayed signed checkpoint backward
failing with 62 forward tensors versus 46 recomputed tensors. The new path uses
the shared per-forward scale capture, including repeated protected-gradient replay.
The [JSON result](receipts/shared-pg09-audit/legacy-defects.json) also records the
old whole-pipeline adapter scope and negative layer 22 skipping. Shared inference
now scales the conditional branch only and keeps the empty negative branch frozen,
with no negative layer skipping. The vendored official model source is unchanged.

The old CLI/cached/smoke trainers optimized local MSE and imported no shared GAN.
All current training now executes the actual installed conditional game. Paired
signs share latent/time and frozen teachers. Generic critic conditions include
mask-weighted normalized Qwen features, native timestep, sign and strength;
cached campaigns reuse the exact shared condition policy. Scale calibration uses
only training edit RMS. Held-out data never fits the scale or training objective.

Full game checkpoints keep adapter, critic, both optimizers, all RNG and shuffled
row/cursor/exposure state. Checksums and core identities are verified on CPU before
restore. Exact continuation retains the original budget and step counter. Portable
v2 inference exports carry verified shared provenance and strict FP32 finite actual
adapter leaves; checksum/shape/target/config mismatches fail before attaching hooks.
Legacy v1 adapters remain readable; MSE optimizer packets are never accepted as GAN
resumes. Shared inference exports cannot be reserialized into a full checkpoint.
Failed generic construction restores model flags, forwards and adapter hooks.
Training and inference tokenize the complete native chat template without
truncation before encoding. Initialization seeds CPU and only the fork-owned
CUDA generator, preserving other generators. Portable export prevalidates
provenance JSON before filesystem writes and publishes its manifest last;
ordinary failures clean the newly reserved output and allow retry.

## Validation and remaining gates

The sole packaging owner's final immutable proposed dependency is
`cdfb80a76b882b4b73861b12f555263bd64ffeb7`, `particle-sliders-core` 0.6.0,
released ParticleGAN 0.9.1. It is pending Ada approval and merge. Its producer wheel
SHA256 is `4abc83988fdeb6d7478de830e7ee8ec5d9bd99c20d52f89e4ce594d50dfd39a3`.
The official ParticleGAN wheel SHA256 is
`b01b7957169cb3de45887e22b4d525fd54071577e1de58089707cf05fe6992ec`.
Final source/wheel receipts accompany this exact dependency pin. The original 42a2e371 installed-wheel consumer suite passed **48 tests in 7.43
seconds**, with no skips; these original receipts remain historical and unchanged. Its separate isolated `-I` process outside both checkouts imported the three
wheels and passed the four-update GAN/resume/reload smoke with `conceptmod` absent.
The optimizer-state P1 review superseded that dependency. A separate replacement
receipt records the earlier 4be-wheel rerun: **46 installed-wheel CPU tests in
13.413 seconds**, with no failures or skips, including product RNG scope,
inference-token preflight and export failure cleanup. The replacement isolated
four-update/replay/reload smoke also passed with all three imports from installed
wheels, no `conceptmod`, and CUDA uninitialized. All 11 active source pins match.
See the [replacement receipt](receipts/shared-pg09-replacement/receipt.json) for
source commit `3d4bad7df2ee5527bb3de24248715c0ee424f8df`, wheel hashes and the
29-source map. Those results remain scoped to 4be. The final CDF qualification
is recorded separately under `receipts/shared-pg09-cdf/`, using the exact producer
wheel and a narrow MOSS restore/update/export check. CDF adds Adam-specific clock
bounds and an Iris metadata fix; MOSS uses the unchanged optimized non-Adam
wrappers. No image payload or existing source cohort is adopted. These CPU receipts
do not qualify pretrained image quality.

The [CDF receipt](receipts/shared-pg09-cdf/receipt.json) records **18 narrow
installed-wheel CPU tests in 10.770 seconds**, zero failures/skips, plus an
isolated installed-origin/metadata and four-update/export/exact-resume smoke.
The product pin commit is `b247a239d3b73b75b32291a85e068c1ec57ef258`; its exact
Git-archive wheel SHA256 is
`04fc39472bd551055f5bb37af4c80e279dce7925666b06b32f02458dbe4193be`.
Product code/tests/tools and the 42/4be receipts remain unchanged. Only this
narrow CDF result qualifies the final dependency pin; CUDA remains uninitialized.

Before any new native run the parent must route exact dependency and product heads
through Ada, check source pins and use coordinator admission. Package relocation
changes module/source identity: old source-bound cohorts require the original
runtime or an explicitly reviewed migration, rather than an automatic resume.

Remaining native gates are fresh-process CUDA/BF16 full-state replay, actual model
memory and timing, text encoder/offload hooks, finite native VAE decoding, matched
unattached-base versus off images/latents, signed/moderate ramps, and separate
interim/final visual qualification. No GPU job, model load/download, image/model
publication, resident-service change, merge or active-source edit was performed.
