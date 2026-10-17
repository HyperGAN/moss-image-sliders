# Shared PG0.9.1 integration validation

The current derivative uses the proposed shared package commit
`4be0b5b29d2e2c741431f468d71ba8a7c884b724` (0.6.0) and released ParticleGAN 0.9.1.
The original 42a2e371 candidate passed 48 installed-wheel CPU checks. Its receipts
are preserved under `receipts/shared-pg09-audit/` as historical evidence. That
candidate was superseded after an optimizer-state P1 review. Replacement
qualification and product RNG, inference-token and export-retry corrections are
recorded separately under `receipts/shared-pg09-replacement/`; only the replacement
receipt qualifies the current dependency pin. See [the audit](shared-pg09-audit.md).

No GPU/native model run or quality claim belongs to this audit. Existing native
0.9.0 campaigns and their sources are untouched. The receipts below are historical
baseline validation and must not be read as qualification of the new GAN objective.

# Initial validation receipts — 2026-10-17

## Environment inspected

Host-level read-only `nvidia-smi` returned two NVIDIA RTX A6000 cards, each
49,140 MiB total. At inspection GPU 0 had 46,903 MiB free (21% utilization),
GPU 1 had 31,365 MiB free (0% utilization). These are snapshots, not reservations.
`findmnt /mnt/nvme` confirmed a writable ext4 mount; `df -h` showed about 3.6 TB
free. The restricted execution sandbox masks GPU access and makes this mount
read-only; the host-level check resolved that discrepancy. No system settings,
driver settings, boot settings or mount settings were changed. No NVMe files
were written and no model weights were downloaded.

GitHub read checks confirmed account `mikkel`, active admin membership in `HyperGAN`,
and no existing `moss-image-sliders` repository before creation. Existing
`/ml2/hypergan/particle-sliders` and `/ml2/sliders/particle-sliders` remained clean;
no edits or branches were created in either. The new project has its own Git
repository and feature branch `feat/moss-image-sliders`.

The local Babel onboarding helper was inspected as requested by machine instructions;
setup of the separate `moss-image-sliders` instance returned `OSError` inside the
restricted environment. No enrollment, external coordination message, or permission
change was completed. This does not block the project or authorized GitHub publishing.

## Bounded verification

Python 3.12.13, torch 2.9.1+cpu, diffusers 0.38.0, transformers 5.3.0,
safetensors 0.8.0. Installed with the hashed CPU lock, without CUDA runtime packages.
The vendored official pipeline's SHA256 is checked by the suite.

- `pytest -q`: integration suite passes; see [JUnit receipt](receipts/pytest.xml).
- `ruff check src tests`: passed (vendor source excluded and preserved unchanged).
- `python -m compileall -q src`: passed.
- `uv lock --check --offline`: 98 packages resolved; lock current.
- `uv build --offline`: wheel and sdist built. Inspected wheel includes project MIT,
  particle-sliders MIT, OpenMOSS Apache-2.0 and NOTICE files.
- `moss-sliders smoke --receipt docs/receipts/cpu-smoke.json`: four optimizer steps on
  the real official transformer at tiny random dimensions, checkpoint round trip,
  and Euler latent inference at -1/0/+1. No image decoder or model download.

The [CPU smoke receipt](receipts/cpu-smoke.json) records losses decreasing from
0.00126879 to 0.000173493. This validates code behavior, not pretrained image quality.
Tests also verify exact zero-scale preservation, unchanged base parameters, signed
Linear residuals, deterministic CFG sampling, frozen teachers, nonzero adapter
gradients, checkpointed and ordinary backward, BF16 base with FP32 adapters,
strict metadata and state rejection, duplicate attachment prevention, exception-safe
scale reset, scheduler reset, pinned loader calls, config guards and UI construction.

## Remaining acceptance work

GitHub rejected the initial push because the existing OAuth token lacks `workflow`
scope for `.github/workflows/tests.yml`. No refs were created by that rejected push.
Rather than expanding permissions, the workflow is distributed as
[`ci-workflow.example.yml`](ci-workflow.example.yml). An authorized maintainer can
install it under `.github/workflows/` later. GitHub Actions has not run; all receipts
above are local. The unpublished initial commit was amended to remove the active
workflow from repository history before retrying publication.

No full MOSS checkpoint was downloaded, and no real GPU training or decoded image
quality evaluation ran. The 20-step pilot and visual held-out ramp acceptance are
documented in README. The host has suitable GPUs, but existing usage must be
rechecked and a writable cache selected before the approximately 28 GB download.
GPU peak memory, runtime and semantic slider efficacy are unmeasured. This release
does not claim trained weights, visual quality, multi-GPU support or compatibility
with third-party LoRA loaders.
